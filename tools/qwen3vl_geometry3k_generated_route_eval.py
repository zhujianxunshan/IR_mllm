#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


VARIANTS = ("image_only", "image_generated_theorem_route")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def normalize_answer(text: str, valid_letters: str = "ABCDE") -> str:
    letter_class = re.escape(valid_letters)
    matches = re.findall(rf"ANSWER\s*[:：]\s*([{letter_class}])\b", text or "", flags=re.I)
    if matches:
        return matches[-1].upper()
    matches = re.findall(rf"(?:option|choice|answer)\s*(?:is|:|：)?\s*([{letter_class}])\b", text or "", flags=re.I)
    if matches:
        return matches[-1].upper()
    stripped = (text or "").strip()
    if re.fullmatch(rf"[{letter_class}]", stripped, flags=re.I):
        return stripped.upper()
    prefix = re.match(rf"^\s*([{letter_class}])\s*[.)．。:：]", stripped, flags=re.I)
    return prefix.group(1).upper() if prefix else ""


def route_prompt(row: dict[str, Any], gdp_text: str) -> str:
    choices = row.get("choices") or []
    choice_text = "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices))
    return f"""You are a theorem-route generator for geometry problems.
Given the problem text, answer choices, and an automatic diagram parse, predict a compact theorem route that may help a multimodal model solve the problem.

Important:
- Do not choose an option.
- Do not solve the problem numerically.
- Do not output the final answer.
- Output only [THEOREM ROUTE], optional [RAW THEOREM SEQUENCE], and [USE POLICY].

Question:
{row.get("question", "")}

Choices:
{choice_text}

GDP-4B automatic diagram parse:
{gdp_text}
"""


def generate_route(row: dict[str, Any], gdp_text: str, tok: Any, model: Any, max_new_tokens: int) -> str:
    prompt = f"<|im_start|>user\n{route_prompt(row, gdp_text).strip()}<|im_end|>\n<|im_start|>assistant\n"
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


def build_messages(row: dict[str, Any], variant: str, generated_route: str, image_path: Path) -> list[dict[str, Any]]:
    choices = row.get("choices") or []
    choice_text = "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices))
    valid_letters = "/".join(chr(65+i) for i in range(len(choices)))
    extra = "No theorem route is provided."
    if variant == "image_generated_theorem_route":
        extra = (
            "Generated theorem-route hypothesis. Use it only after checking the image and question; ignore it if it conflicts with visible geometry.\n\n"
            + generated_route
        )
    prompt = f"""Solve this geometry multiple-choice problem.

Question:
{row.get("question", "")}

Choices:
{choice_text}

Additional context:
{extra}

Return exactly one line and nothing else:
ANSWER: X

X must be one of {valid_letters}."""
    return [{"role": "user", "content": [{"type": "image", "image": str(image_path.resolve())}, {"type": "text", "text": prompt}]}]


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
    ap.add_argument("--dataset", type=Path, default=Path("multidataset_2x300_eval.jsonl"))
    ap.add_argument("--gdp", type=Path, default=Path("gdp4b_geometry300_parse.jsonl"))
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--qwen-vl-model", default="~/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--route-cache", type=Path, default=Path("geometry3k_generated_route_cache_150.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("qwen3vl8b_geometry3k_generated_route150.jsonl"))
    ap.add_argument("--summary", type=Path, default=Path("qwen3vl8b_geometry3k_generated_route150_summary.json"))
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--route-max-new-tokens", type=int, default=220)
    ap.add_argument("--answer-max-new-tokens", type=int, default=32)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.dataset) if r.get("dataset") == "Geometry3K-300"]
    gdp_by_id = {}
    for item in read_jsonl(args.gdp):
        if item.get("error") is None:
            gdp_by_id[str(item["id"])] = item.get("gdp_text") or item.get("gdp_response") or ""
    rows = [r for r in rows if str(r["id"]) in gdp_by_id][: args.limit]
    variants = list(VARIANTS)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    route_by_id: dict[str, str] = {}
    if args.route_cache.exists():
        for item in read_jsonl(args.route_cache):
            route_by_id[str(item["id"])] = item.get("generated_route", "")

    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True, quantization_config=quant, device_map="auto")
    gen_model = PeftModel.from_pretrained(base, args.adapter)
    gen_model.eval()
    for row in rows:
        rid = str(row["id"])
        if rid in route_by_id:
            continue
        started = time.time()
        route = generate_route(row, gdp_by_id[rid], tok, gen_model, args.route_max_new_tokens)
        route_by_id[rid] = route
        append_jsonl(args.route_cache, {"id": rid, "generated_route": route, "duration_s": time.time() - started})
        print(f"ROUTE\t{rid}\tchars={len(route)}", flush=True)
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
        for variant in variants:
            if (rid, variant) in done:
                continue
            started = time.time()
            text = ""
            error = None
            new_len = 0
            truncated = False
            try:
                messages = build_messages(row, variant, route_by_id.get(rid, ""), image_path)
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
                "dataset": "Geometry3K-150",
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
                "generated_route": route_by_id.get(rid, "") if variant == "image_generated_theorem_route" else "",
            }
            append_jsonl(args.out, rec)
            results.append(rec)
            done.add((rid, variant))
            print(f"{'OK' if correct else 'FAIL'}\t{rid}\t{variant}\tpred={pred}\tgold={row['answer']}\tduration_s={rec['duration_s']:.2f}", flush=True)

    summary = summarize(results, variants)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
