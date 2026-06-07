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


VARIANTS = ("image_only", "image_generated_theorem_route", "image_oracle_theorem_route")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


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
    matches = re.findall(r"ANSWER\s*[:：]\s*([^\n\r]+)", text or "", flags=re.I)
    if matches:
        return matches[-1].strip().rstrip(".。")
    return (text or "").splitlines()[0].strip().rstrip(".。") if text else ""


def numeric_prediction(text: str) -> float | None:
    ans = normalize_text_answer(text)
    m = re.search(r"-?\d+/\d+|-?\d+(?:\.\d+)?", ans)
    return to_float(m.group(0)) if m else None


def answer_correct(pred_text: str, gold: str, tol: float = 1e-3) -> bool:
    pred = numeric_prediction(pred_text)
    target = to_float(gold)
    if pred is None or target is None or not math.isfinite(pred):
        return False
    return abs(pred - target) <= max(tol, abs(target) * 1e-4)


def resize_image(image_path: Path, cache_dir: Path, max_side: int) -> Path:
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


def generate_route(row: dict[str, Any], tok: Any, model: Any, max_new_tokens: int) -> str:
    prompt = f"<|im_start|>user\n{row['prompt'].strip()}<|im_end|>\n<|im_start|>assistant\n"
    enc = tok(prompt, return_tensors="pt").to(model.device)
    import torch

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)


def build_messages(row: dict[str, Any], variant: str, route: str, image_path: Path, max_pixels: int) -> list[dict[str, Any]]:
    extra = "No theorem route is provided."
    if variant == "image_generated_theorem_route":
        extra = "Generated theorem-route hypothesis. Use it only after checking the image.\n\n" + route
    elif variant == "image_oracle_theorem_route":
        extra = "Dataset-annotated oracle theorem route. Use it only after checking the image.\n\n" + row["target"]
    prompt = f"""Solve this geometry problem from the image and text.

Question:
{row["question"]}

Additional context:
{extra}

Return exactly one line and nothing else:
ANSWER: <numeric value>
"""
    return [{"role": "user", "content": [{"type": "image", "image": str(image_path.resolve()), "max_pixels": max_pixels}, {"type": "text", "text": prompt}]}]


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
        }
    for variant in variants:
        if variant == "image_only":
            continue
        win = loss = both_correct = both_wrong = incomplete = 0
        for _, pair in by_id.items():
            base, other = pair.get("image_only"), pair.get(variant)
            if not base or not other or not base.get("complete_response") or not other.get("complete_response"):
                incomplete += 1
            elif other.get("correct") and not base.get("correct"):
                win += 1
            elif base.get("correct") and not other.get("correct"):
                loss += 1
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
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, default=Path("formalgeo_route_data/test.jsonl"))
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--qwen-vl-model", default="~/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--route-cache", type=Path, default=Path("formalgeo_generated_route_cache.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("qwen3vl8b_formalgeo_generated_route_eval.jsonl"))
    ap.add_argument("--summary", type=Path, default=Path("qwen3vl8b_formalgeo_generated_route_eval_summary.json"))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--variants", default="image_only,image_generated_theorem_route,image_oracle_theorem_route")
    ap.add_argument("--route-max-new-tokens", type=int, default=220)
    ap.add_argument("--answer-max-new-tokens", type=int, default=32)
    ap.add_argument("--resize-max-side", type=int, default=768)
    ap.add_argument("--max-pixels", type=int, default=262144)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.test)[: args.limit]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    route_by_id: dict[str, str] = {}
    if args.route_cache.exists():
        for r in read_jsonl(args.route_cache):
            route_by_id[str(r["id"])] = r.get("generated_route", "")

    if any(v == "image_generated_theorem_route" for v in variants):
        tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True, local_files_only=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        base = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True, quantization_config=quant, device_map="auto")
        gen_model = PeftModel.from_pretrained(base, args.adapter)
        gen_model.eval()
        for row in rows:
            if row["id"] in route_by_id:
                continue
            started = time.time()
            route = generate_route(row, tok, gen_model, args.route_max_new_tokens)
            route_by_id[row["id"]] = route
            append_jsonl(args.route_cache, {"id": row["id"], "generated_route": route, "duration_s": time.time() - started})
            print(f"ROUTE\t{row['id']}\tchars={len(route)}", flush=True)
        del gen_model, base
        torch.cuda.empty_cache()

    model_name = str(Path(args.qwen_vl_model).expanduser()) if args.qwen_vl_model.startswith("~") else args.qwen_vl_model
    kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True, "torch_dtype": "auto"}
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, max_pixels=args.max_pixels)
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    vlm.eval()
    eos = vlm.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else ({int(eos)} if eos is not None else set())

    done: set[tuple[str, str]] = set()
    results: list[dict[str, Any]] = []
    if args.out.exists():
        for item in read_jsonl(args.out):
            results.append(item)
            done.add((item["id"], item["variant"]))

    cache_dir = Path("formalgeo_route_data/resized_eval_768")
    for row in rows:
        image_path = resize_image(Path(row["image"]), cache_dir, args.resize_max_side)
        route = route_by_id.get(row["id"], "")
        for variant in variants:
            if (row["id"], variant) in done:
                continue
            started = time.time()
            text = ""
            error = None
            new_len = 0
            truncated = False
            try:
                messages = build_messages(row, variant, route, image_path, args.max_pixels)
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
            pred = numeric_prediction(text)
            complete = error is None and not truncated and pred is not None
            correct = complete and answer_correct(text, row["answer"])
            rec = {
                "dataset": "FormalGeo7K-v2",
                "id": row["id"],
                "variant": variant,
                "question": row["question"],
                "answer": row["answer"],
                "prediction_text": normalize_text_answer(text),
                "prediction_value": pred,
                "correct": correct,
                "complete_response": complete,
                "truncated": truncated,
                "new_tokens": new_len,
                "duration_s": time.time() - started,
                "raw_response": text,
                "error": error,
                "generated_route": route if variant == "image_generated_theorem_route" else "",
                "oracle_route": row["target"] if variant == "image_oracle_theorem_route" else "",
            }
            append_jsonl(args.out, rec)
            results.append(rec)
            done.add((row["id"], variant))
            print(f"{'OK' if correct else 'FAIL'}\t{row['id']}\t{variant}\tpred={pred}\tgold={row['answer']}\tduration_s={rec['duration_s']:.2f}", flush=True)

    summary = summarize(results, variants)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
