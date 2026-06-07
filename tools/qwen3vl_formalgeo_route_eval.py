#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


VARIANTS = ("image_only", "image_oracle_theorem_route")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_simple_answer(answer: str) -> bool:
    answer = answer.strip()
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?|-?\d+/\d+", answer))


def to_float(answer: str) -> float | None:
    answer = answer.strip()
    try:
        if "/" in answer and re.fullmatch(r"-?\d+/\d+", answer):
            a, b = answer.split("/")
            return float(a) / float(b)
        return float(answer)
    except Exception:
        return None


def normalize_text_answer(text: str) -> str:
    text = (text or "").strip()
    matches = re.findall(r"ANSWER\s*[:：]\s*([^\n\r]+)", text, flags=re.I)
    if matches:
        return matches[-1].strip().rstrip(".。")
    return text.splitlines()[0].strip().rstrip(".。") if text else ""


def numeric_prediction(text: str) -> float | None:
    ans = normalize_text_answer(text)
    if not ans:
        return None
    # Prefer a bare fraction/number immediately after ANSWER.
    m = re.search(r"-?\d+/\d+|-?\d+(?:\.\d+)?", ans)
    if not m:
        return None
    return to_float(m.group(0))


def answer_correct(pred_text: str, gold: str, tol: float = 1e-3) -> bool:
    pred = numeric_prediction(pred_text)
    target = to_float(gold)
    if pred is None or target is None or not math.isfinite(pred):
        return False
    return abs(pred - target) <= max(tol, abs(target) * 1e-4)


def theorem_name(seq: str) -> str:
    return seq.split("(", 1)[0].strip()


def theorem_to_english(seq: str) -> str:
    name = theorem_name(seq)
    text = name
    for old in [
        "_property",
        "_judgment",
        "_determination",
        "_definition",
        "_algebraic",
        "_with_common_vertex",
    ]:
        text = text.replace(old, "")
    text = text.replace("_", " ")
    return text.strip()


def theorem_route_text(theorem_seqs: list[str]) -> str:
    lines = [
        "The following theorem route is annotated by the dataset. Treat it as a solving hint, not as the final answer.",
        "",
        "[THEOREM ROUTE]",
    ]
    for i, seq in enumerate(theorem_seqs, 1):
        lines.append(f"{i}. Use {theorem_to_english(seq)}.")
    lines.extend(["", "[RAW THEOREM SEQUENCE]"])
    for i, seq in enumerate(theorem_seqs, 1):
        lines.append(f"{i}. {seq}")
    lines.extend(
        [
            "",
            "[USE POLICY]",
            "- Verify each theorem against the diagram and problem text.",
            "- Do not copy numbers from this route; compute the requested value from the image and question.",
        ]
    )
    return "\n".join(lines)


def resized_image_path(image_path: Path, cache_dir: Path, max_side: int) -> Path:
    out = cache_dir / f"{image_path.stem}.jpg"
    if out.exists():
        return out
    from PIL import Image

    cache_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path)
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    img.save(out, "JPEG", quality=92, optimize=True)
    return out


def build_records(dataset_name: str, datasets_path: str, limit: int, stratified: bool = True, resize_max_side: int = 768) -> list[dict[str, Any]]:
    from formalgeo.data import DatasetLoader

    loader = DatasetLoader(dataset_name, datasets_path)
    root = Path(datasets_path) / dataset_name
    diagrams = root / "diagrams"
    resized_dir = root / f"diagrams_resized_{resize_max_side}"
    n = int(loader.info.get("problem_number", 7000))
    rows: list[dict[str, Any]] = []
    for pid in range(1, n + 1):
        try:
            p = loader.get_problem(pid)
        except Exception:
            continue
        answer = str(p.get("problem_answer", "")).strip()
        theorem_seqs = [str(x) for x in p.get("theorem_seqs", []) if str(x).strip()]
        image_name = str(p.get("problem_img", "")).strip()
        original_image_path = diagrams / image_name
        question = str(p.get("problem_text_en", "")).strip()
        if not question or not theorem_seqs or not original_image_path.exists() or not is_simple_answer(answer):
            continue
        rows.append(
            {
                "id": str(pid),
                "question": question,
                "image": str(original_image_path),
                "original_image": str(original_image_path),
                "answer": answer,
                "theorem_seqs": theorem_seqs,
                "goal_cdl": p.get("goal_cdl", ""),
                "source": p.get("source", ""),
            }
        )
        if not stratified and len(rows) >= limit:
            break
    if not stratified or len(rows) <= limit:
        selected = rows[:limit]
        for row in selected:
            row["image"] = str(resized_image_path(Path(row["original_image"]), resized_dir, resize_max_side))
        return selected

    def bucket(row: dict[str, Any]) -> str:
        k = len(row["theorem_seqs"])
        if k <= 1:
            return "easy_len1"
        if k <= 3:
            return "medium_len2_3"
        return "hard_len4plus"

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[bucket(row)].append(row)
    names = ["easy_len1", "medium_len2_3", "hard_len4plus"]
    quota = {name: limit // len(names) for name in names}
    for name in names[: limit % len(names)]:
        quota[name] += 1
    selected: list[dict[str, Any]] = []
    for name in names:
        selected.extend(groups[name][: quota[name]])
    # If one bucket is short, fill from the remaining records deterministically.
    chosen = {row["id"] for row in selected}
    for row in rows:
        if len(selected) >= limit:
            break
        if row["id"] not in chosen:
            selected.append(row)
            chosen.add(row["id"])
    selected = sorted(selected, key=lambda x: int(x["id"]))
    for row in selected:
        row["image"] = str(resized_image_path(Path(row["original_image"]), resized_dir, resize_max_side))
    return selected


def build_messages(row: dict[str, Any], variant: str, image_path: Path, max_pixels: int) -> list[dict[str, Any]]:
    extra = "No extra theorem route is provided."
    if variant == "image_oracle_theorem_route":
        extra = theorem_route_text(row["theorem_seqs"])
    prompt = f"""Solve this geometry problem from the image and text.

Question:
{row["question"]}

Additional context:
{extra}

Return exactly one line and nothing else:
ANSWER: <numeric value>
"""
    image_item = {"type": "image", "image": str(image_path.resolve()), "max_pixels": max_pixels}
    return [{"role": "user", "content": [image_item, {"type": "text", "text": prompt}]}]


def summarize(results: list[dict[str, Any]], variants: list[str]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in results:
        by_variant[r["variant"]].append(r)
        by_id[r["id"]][r["variant"]] = r
    out: dict[str, Any] = {"variants": {}, "pairwise_vs_image_only": {}, "n_unique_ids": len(by_id)}
    for variant in variants:
        items = by_variant.get(variant, [])
        complete = [r for r in items if r.get("complete_response") and not r.get("error")]
        correct = [r for r in complete if r.get("correct")]
        out["variants"][variant] = {
            "n": len(items),
            "complete": len(complete),
            "correct": len(correct),
            "accuracy": len(correct) / len(items) if items else 0.0,
            "accuracy_on_complete": len(correct) / len(complete) if complete else 0.0,
            "avg_duration_s": sum(float(r.get("duration_s", 0.0)) for r in items) / len(items) if items else 0.0,
            "errors": sum(1 for r in items if r.get("error")),
            "missing_numeric_answer": sum(1 for r in items if r.get("prediction_value") is None),
        }
    base_name = "image_only"
    for variant in variants:
        if variant == base_name:
            continue
        win = loss = both_correct = both_wrong = incomplete = 0
        examples = {"win": [], "loss": []}
        for pid, pair in by_id.items():
            base = pair.get(base_name)
            other = pair.get(variant)
            if not base or not other or not base.get("complete_response") or not other.get("complete_response"):
                incomplete += 1
            elif other.get("correct") and not base.get("correct"):
                win += 1
                if len(examples["win"]) < 10:
                    examples["win"].append(pid)
            elif base.get("correct") and not other.get("correct"):
                loss += 1
                if len(examples["loss"]) < 10:
                    examples["loss"].append(pid)
            elif base.get("correct") and other.get("correct"):
                both_correct += 1
            else:
                both_wrong += 1
        out["pairwise_vs_image_only"][variant] = {
            "win": win,
            "loss": loss,
            "net": win - loss,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "incomplete": incomplete,
            "examples": examples,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", default="formalgeo7k_v2")
    ap.add_argument("--datasets-path", default="datasets/formalgeo")
    ap.add_argument("--qwen-vl-model", default="~/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--no-stratified", action="store_true")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", default="qwen3vl8b_formalgeo_route300.jsonl")
    ap.add_argument("--summary", default="qwen3vl8b_formalgeo_route300_summary.json")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-pixels", type=int, default=262144)
    ap.add_argument("--resize-max-side", type=int, default=768)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    rows = build_records(
        args.dataset_name,
        args.datasets_path,
        args.limit,
        stratified=not args.no_stratified,
        resize_max_side=args.resize_max_side,
    )
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    lengths = defaultdict(int)
    for row in rows:
        k = len(row["theorem_seqs"])
        lengths["len1" if k <= 1 else "len2_3" if k <= 3 else "len4plus"] += 1
    print(f"Loaded {len(rows)} simple-answer FormalGeo records for {variants}. theorem_len_buckets={dict(lengths)}", flush=True)

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True, "torch_dtype": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            compute_dtype = torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model_name = str(Path(args.qwen_vl_model).expanduser()) if args.qwen_vl_model.startswith("~") else args.qwen_vl_model
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, max_pixels=args.max_pixels)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    model.eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else ({int(eos)} if eos is not None else set())

    out_path = Path(args.out)
    done: set[tuple[str, str]] = set()
    results: list[dict[str, Any]] = []
    for item in read_jsonl(out_path):
        results.append(item)
        done.add((item["id"], item["variant"]))

    for row in rows:
        image_path = Path(row["image"])
        for variant in variants:
            if (row["id"], variant) in done:
                print("SKIP", row["id"], variant, flush=True)
                continue
            started = time.time()
            text = ""
            error = None
            new_len = 0
            truncated = False
            try:
                messages = build_messages(row, variant, image_path, args.max_pixels)
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
                new_tokens = trimmed[0].tolist()
                new_len = len(new_tokens)
                truncated = new_len >= args.max_new_tokens and (not eos_ids or new_tokens[-1] not in eos_ids)
                text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            pred_value = numeric_prediction(text)
            complete = error is None and not truncated and pred_value is not None
            correct = complete and answer_correct(text, row["answer"])
            rec = {
                "dataset": "FormalGeo7K-v2",
                "id": row["id"],
                "source": row.get("source", ""),
                "variant": variant,
                "question": row["question"],
                "image": row["image"],
                "original_image": row.get("original_image", ""),
                "goal_cdl": row.get("goal_cdl", ""),
                "answer": row["answer"],
                "prediction_text": normalize_text_answer(text),
                "prediction_value": pred_value,
                "correct": correct,
                "complete_response": complete,
                "truncated": truncated,
                "new_tokens": new_len,
                "duration_s": time.time() - started,
                "raw_response": text,
                "error": error,
                "theorem_seqs": row["theorem_seqs"],
                "theorem_route_text": theorem_route_text(row["theorem_seqs"]) if variant == "image_oracle_theorem_route" else "",
            }
            append_jsonl(out_path, rec)
            results.append(rec)
            done.add((row["id"], variant))
            print(
                f"{'OK' if correct else 'FAIL'}\t{row['id']}\t{variant}\tpred={pred_value}\tgold={row['answer']}\tcomplete={complete}\tduration_s={rec['duration_s']:.2f}",
                flush=True,
            )

    summary = summarize(results, variants)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
