#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


VARIANTS = ("route_minimal", "route_cautious", "route_target_check")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def normalize_answer(text: str, valid_letters: str = "ABCDE") -> str:
    cls = re.escape(valid_letters)
    matches = re.findall(rf"ANSWER\s*[:：]\s*([{cls}])\b", text or "", flags=re.I)
    if matches:
        return matches[-1].upper()
    stripped = (text or "").strip()
    if re.fullmatch(rf"[{cls}]", stripped, flags=re.I):
        return stripped.upper()
    prefix = re.match(rf"^\s*([{cls}])\s*[.)．。:：]", stripped, flags=re.I)
    return prefix.group(1).upper() if prefix else ""


def choices_text(row: dict[str, Any]) -> str:
    return "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(row.get("choices") or []))


def theorem_route_only(route: str) -> str:
    m = re.search(r"\[THEOREM ROUTE\]\s*(.*?)(?=\n\[[A-Z_ ]+\]|\Z)", route or "", flags=re.S)
    text = m.group(1).strip() if m else (route or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[:6])


def build_extra(variant: str, route: str) -> str:
    minimal = theorem_route_only(route)
    if variant == "route_minimal":
        return (
            "Possible theorem-route hint. It may be incomplete.\n"
            "Use only the theorem names below as a weak hint; do not rely on the raw theorem sequence.\n"
            f"<THEOREM_HINT>\n{minimal}\n</THEOREM_HINT>"
        )
    if variant == "route_cautious":
        return (
            "Low-confidence theorem-route hypothesis. The hint may be wrong or irrelevant.\n"
            "Use it only if every theorem is directly supported by visible marks, labels, and the problem text. "
            "If any theorem is unsupported, ignore the whole hint and solve from the image first.\n"
            f"<THEOREM_HINT>\n{minimal}\n</THEOREM_HINT>"
        )
    if variant == "route_target_check":
        return (
            "Low-confidence theorem-route hypothesis. Before using it, internally check target binding:\n"
            "1. Does the route compute exactly the requested quantity, not a nearby angle/length/area?\n"
            "2. Are the required givens visible in the image and stated in the text?\n"
            "3. For area, arc, sine, similarity, and trapezoid/parallelogram formulas, ignore the route unless the needed dimensions or angle are explicit.\n"
            "If these checks fail, solve image-first and disregard the route.\n"
            f"<THEOREM_HINT>\n{minimal}\n</THEOREM_HINT>"
        )
    raise ValueError(variant)


def build_messages(row: dict[str, Any], variant: str, route: str, image_path: Path) -> list[dict[str, Any]]:
    choices = row.get("choices") or []
    valid_letters = "/".join(chr(65 + i) for i in range(len(choices)))
    prompt = f"""Solve this geometry multiple-choice problem.

Question:
{row.get("question", "")}

Choices:
{choices_text(row)}

Additional context:
{build_extra(variant, route)}

Return exactly one line and nothing else:
ANSWER: X

X must be one of {valid_letters}."""
    return [{"role": "user", "content": [{"type": "image", "image": str(image_path.resolve())}, {"type": "text", "text": prompt}]}]


def summarize(results: list[dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_id: dict[str, dict[str, Any]] = defaultdict(dict)
    for r in results:
        by_variant[r["variant"]].append(r)
        by_id[str(r["id"])][r["variant"]] = r
    out: dict[str, Any] = {"variants": {}, "pairwise_vs_image_only": {}, "n_unique_ids": len(by_id)}
    for variant in VARIANTS:
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
        }
        win = loss = both_correct = both_wrong = incomplete = 0
        for qid, other in by_id.items():
            base = baseline.get(qid)
            item = other.get(variant)
            if not base or not item or not base.get("complete_response") or not item.get("complete_response"):
                incomplete += 1
            elif item.get("correct") and not base.get("correct"):
                win += 1
            elif base.get("correct") and not item.get("correct"):
                loss += 1
            elif base.get("correct") and item.get("correct"):
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
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("multidataset_2x300_eval.jsonl"))
    ap.add_argument("--route-cache", type=Path, required=True)
    ap.add_argument("--test-ids", type=Path, required=True)
    ap.add_argument("--baseline-results", type=Path, required=True)
    ap.add_argument("--qwen-vl-model", default="~/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--answer-max-new-tokens", type=int, default=32)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    wanted = set(json.loads(args.test_ids.read_text(encoding="utf-8"))) if args.test_ids.exists() else set()
    rows_all = [r for r in read_jsonl(args.dataset) if r.get("dataset") == "Geometry3K-300"]
    rows = [r for r in rows_all if str(r["id"]) in wanted][: args.limit]
    if len(rows) < args.limit:
        raise RuntimeError(f"Need {args.limit} test rows, got {len(rows)}")

    route_by_id = {str(r["id"]): r.get("generated_route", "") for r in read_jsonl(args.route_cache)}
    baseline = {
        str(r["id"]): r
        for r in read_jsonl(args.baseline_results)
        if r.get("variant") == "image_only"
    }

    model_name = str(Path(args.qwen_vl_model).expanduser()) if args.qwen_vl_model.startswith("~") else args.qwen_vl_model
    kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True, "torch_dtype": "auto"}
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    vlm.eval()
    eos = vlm.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else ({int(eos)} if eos is not None else set())

    done: set[tuple[str, str]] = set()
    results: list[dict[str, Any]] = []
    if args.out.exists():
        for item in read_jsonl(args.out):
            results.append(item)
            done.add((str(item["id"]), item["variant"]))

    for row in rows:
        rid = str(row["id"])
        image_path = Path(str(row["image"]))
        route = route_by_id.get(rid, "")
        for variant in VARIANTS:
            if (rid, variant) in done:
                continue
            started = time.time()
            text = ""
            error = None
            new_len = 0
            truncated = False
            try:
                messages = build_messages(row, variant, route, image_path)
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(vlm.device)
                with torch.inference_mode():
                    generated = vlm.generate(**inputs, max_new_tokens=args.answer_max_new_tokens, do_sample=False)
                trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
                new_tokens = trimmed[0].tolist()
                new_len = len(new_tokens)
                truncated = new_len >= args.answer_max_new_tokens and (not eos_ids or new_tokens[-1] not in eos_ids)
                text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            valid_letters = "".join(chr(65+i) for i in range(len(row.get("choices") or []))) or "ABCDE"
            pred = normalize_answer(text, valid_letters)
            complete = error is None and not truncated and pred != ""
            correct = complete and pred == str(row["answer"]).strip().upper()
            rec = {
                "dataset": "Geometry3K-route-harm-200",
                "id": rid,
                "variant": variant,
                "answer": row["answer"],
                "prediction": pred,
                "correct": correct,
                "complete_response": complete,
                "truncated": truncated,
                "new_tokens": new_len,
                "duration_s": time.time() - started,
                "raw_response": text,
                "error": error,
            }
            append_jsonl(args.out, rec)
            results.append(rec)
            done.add((rid, variant))
            print(f"{'OK' if correct else 'FAIL'}\t{rid}\t{variant}\tpred={pred}\tgold={row['answer']}\tduration_s={rec['duration_s']:.2f}", flush=True)

    summary = summarize(results, baseline)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
