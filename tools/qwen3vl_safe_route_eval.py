#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


VARIANTS = (
    "image_only",
    "old_route",
    "compact_route",
    "compact_route_qc",
    "confidence_route",
    "confidence_route_qc",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def choices_text(row: dict[str, Any]) -> str:
    return "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(row.get("choices") or []))


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


def clip(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + "\n...[truncated]"


def route_prompt(row: dict[str, Any], gdp_text: str, mode: str) -> str:
    return f"""You are a conservative theorem-route generator for geometry MLLM solving.
Given the problem text, answer choices, and an automatic diagram parse, output a short route hint only when it is likely to help.

Rules:
- Never solve numerically and never choose the final answer.
- Never repeat the same theorem.
- Prefer 1-3 target-matched theorem hints.
- If the route is uncertain, mark it LOW_CONFIDENCE.
- If no theorem is clearly supported by the parse and target, output DISCARD.

Generation mode: {mode}

Question:
{row.get("question", "")}

Choices:
{choices_text(row)}

GDP-4B automatic diagram parse:
{gdp_text}
"""


def generate_route(row: dict[str, Any], gdp_text: str, tok: Any, model: Any, mode: str, max_new_tokens: int) -> str:
    import torch

    prompt = f"<|im_start|>user\n{route_prompt(row, gdp_text, mode).strip()}<|im_end|>\n<|im_start|>assistant\n"
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=5,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)


def theorem_steps(route: str) -> list[str]:
    if not route:
        return []
    m = re.search(r"\[THEOREM ROUTE\]\s*(.*?)(?=\n\[[A-Z_ ]+\]|\Z)", route, flags=re.S)
    body = m.group(1) if m else route
    steps = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.upper().startswith("DISCARD"):
            continue
        if re.search(r"\bUse\b", line, flags=re.I):
            text = re.sub(r"^\d+\.\s*", "", line)
            text = re.sub(r"^Use\s+", "", text, flags=re.I).strip().rstrip(".")
            if text:
                steps.append(text)
    return steps


def route_qc(route: str) -> dict[str, Any]:
    quality = ""
    m = re.search(r"\[ROUTE_QUALITY\]\s*(.*?)(?=\n\[|\Z)", route or "", flags=re.S)
    if m:
        quality = m.group(1).strip().splitlines()[0].strip().upper()
    steps = theorem_steps(route)
    norm = [re.sub(r"\s+", " ", s.lower()).strip() for s in steps]
    max_rep = max([norm.count(x) for x in set(norm)] or [0])
    dup_ratio = 1.0 - len(set(norm)) / len(norm) if norm else 0.0
    discard = (
        quality == "DISCARD"
        or "DISCARD:" in (route or "")
        or len(steps) == 0
        or len(steps) > 5
        or max_rep >= 3
        or dup_ratio >= 0.5
    )
    low = quality == "LOW_CONFIDENCE" or len(steps) > 3 or max_rep >= 2
    return {"quality": quality, "n_steps": len(steps), "max_repeat": max_rep, "dup_ratio": dup_ratio, "discard": discard, "low_confidence": low}


def qc_route_text(route: str) -> str:
    qc = route_qc(route)
    if qc["discard"]:
        return ""
    label = "[THEOREM ROUTE - LOW CONFIDENCE]" if qc["low_confidence"] else "[THEOREM ROUTE - TRUSTED]"
    warning = (
        "The following steps are uncertain and may be incorrect. Treat them as weak suggestions only."
        if qc["low_confidence"]
        else "The following compact route passed basic quality checks. Still verify it against the image."
    )
    return f"{label}\n{warning}\n\n{clip(route, 1200)}"


def build_extra(variant: str, route: str) -> str:
    if variant == "image_only":
        return "No additional structured context is provided."
    if variant.endswith("_qc"):
        routed = qc_route_text(route)
        if not routed:
            return "No route is provided because the generated route failed quality control. Solve from image and question only."
        return routed
    return "Generated theorem-route hypothesis. Use it only after checking the image and question; ignore it if unsupported.\n\n" + clip(route, 1400)


def build_messages(row: dict[str, Any], variant: str, route: str, image_path: Path) -> list[dict[str, Any]]:
    valid_letters = "/".join(chr(65+i) for i in range(len(row.get("choices") or [])))
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


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in results:
        by_variant[row["variant"]].append(row)
        by_id[str(row["id"])][row["variant"]] = row
    out: dict[str, Any] = {"variants": {}, "pairwise_vs_image_only": {}, "route_qc": {}, "n_unique_ids": len(by_id)}
    for variant in VARIANTS:
        items = by_variant.get(variant, [])
        complete = [r for r in items if r.get("complete_response") and not r.get("error")]
        correct = [r for r in complete if r.get("correct")]
        out["variants"][variant] = {
            "n": len(items),
            "complete": len(complete),
            "correct": len(correct),
            "accuracy": len(correct) / len(items) if items else 0.0,
            "avg_duration_s": sum(float(r.get("duration_s", 0.0)) for r in items) / len(items) if items else 0.0,
        }
    for variant in VARIANTS:
        if variant == "image_only":
            continue
        win = loss = both_correct = both_wrong = 0
        for pair in by_id.values():
            base, item = pair.get("image_only"), pair.get(variant)
            if not base or not item:
                continue
            if item.get("correct") and not base.get("correct"):
                win += 1
            elif base.get("correct") and not item.get("correct"):
                loss += 1
            elif base.get("correct") and item.get("correct"):
                both_correct += 1
            else:
                both_wrong += 1
        out["pairwise_vs_image_only"][variant] = {"win": win, "loss": loss, "net": win - loss, "both_correct": both_correct, "both_wrong": both_wrong}
    for variant in ["old_route", "compact_route", "confidence_route"]:
        items = [r for r in by_variant.get(variant, []) if r.get("route_qc")]
        if items:
            out["route_qc"][variant] = {
                "discard": sum(1 for r in items if r["route_qc"].get("discard")),
                "low_confidence": sum(1 for r in items if r["route_qc"].get("low_confidence")),
                "avg_steps": sum(r["route_qc"].get("n_steps", 0) for r in items) / len(items),
                "max_steps": max(r["route_qc"].get("n_steps", 0) for r in items),
            }
    return out


def main() -> None:
    global VARIANTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("multidataset_2x300_eval.jsonl"))
    ap.add_argument("--dataset-label", default="Geometry3K-300", help="Dataset label to keep; use ALL to disable filtering.")
    ap.add_argument("--gdp", type=Path, default=Path("gdp4b_geometry300_parse.jsonl"))
    ap.add_argument("--test-ids", type=Path, default=Path("data/tcrp_route_policy/test_ids.json"))
    ap.add_argument("--old-route-cache", type=Path, default=Path("results/geometry3k_trust_selection_route_cache_300.jsonl"))
    ap.add_argument("--compact-adapter", type=Path, required=True)
    ap.add_argument("--confidence-adapter", type=Path, required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--qwen-vl-model", default="/home/ubuntu/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--compact-cache", type=Path, required=True)
    ap.add_argument("--confidence-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--variants", default=",".join(VARIANTS), help="Comma-separated variants to run.")
    ap.add_argument("--route-max-new-tokens", type=int, default=160)
    ap.add_argument("--answer-max-new-tokens", type=int, default=32)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    selected = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    valid = set(VARIANTS)
    unknown = [v for v in selected if v not in valid]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}; valid={VARIANTS}")
    VARIANTS = selected

    wanted = set(json.loads(args.test_ids.read_text(encoding="utf-8"))) if args.test_ids.exists() else set()
    rows_all = read_jsonl(args.dataset)
    if args.dataset_label.upper() != "ALL":
        rows_all = [r for r in rows_all if r.get("dataset") == args.dataset_label]
    rows = [r for r in rows_all if not wanted or str(r["id"]) in wanted][: args.limit]
    if len(rows) < args.limit:
        raise RuntimeError(f"Need {args.limit} rows, got {len(rows)}")

    gdp_by_id = {str(r["id"]): (r.get("gdp_text") or r.get("gdp_response") or "") for r in read_jsonl(args.gdp) if not r.get("error")}
    old_route = {str(r["id"]): r.get("generated_route", "") for r in read_jsonl(args.old_route_cache)}
    compact_route = {str(r["id"]): r.get("generated_route", "") for r in read_jsonl(args.compact_cache)}
    confidence_route = {str(r["id"]): r.get("generated_route", "") for r in read_jsonl(args.confidence_cache)}

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    for mode, adapter, cache, route_map in [
        ("compact_sft", args.compact_adapter, args.compact_cache, compact_route),
        ("confidence_sft", args.confidence_adapter, args.confidence_cache, confidence_route),
    ]:
        missing = [r for r in rows if str(r["id"]) not in route_map]
        if not missing:
            continue
        tok = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True, local_files_only=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True, quantization_config=quant, device_map="auto")
        model = PeftModel.from_pretrained(base, adapter)
        model.eval()
        for row in missing:
            rid = str(row["id"])
            started = time.time()
            route = generate_route(row, gdp_by_id.get(rid, ""), tok, model, mode, args.route_max_new_tokens)
            route_map[rid] = route
            append_jsonl(cache, {"id": rid, "generated_route": route, "duration_s": time.time() - started, "mode": mode, "route_qc": route_qc(route)})
            print(f"ROUTE\t{mode}\t{rid}\tsteps={route_qc(route)['n_steps']}\tdiscard={route_qc(route)['discard']}", flush=True)
        del model, base
        torch.cuda.empty_cache()

    kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True, "torch_dtype": "auto"}
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    processor = AutoProcessor.from_pretrained(args.qwen_vl_model, trust_remote_code=True)
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(args.qwen_vl_model, **kwargs)
    vlm.eval()
    eos = vlm.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else ({int(eos)} if eos is not None else set())

    route_for_variant = {
        "old_route": old_route,
        "compact_route": compact_route,
        "compact_route_qc": compact_route,
        "confidence_route": confidence_route,
        "confidence_route_qc": confidence_route,
    }
    done: set[tuple[str, str]] = set()
    results: list[dict[str, Any]] = []
    for r in read_jsonl(args.out):
        results.append(r)
        done.add((str(r["id"]), r["variant"]))

    for row in rows:
        rid = str(row["id"])
        for variant in VARIANTS:
            if (rid, variant) in done:
                continue
            route = route_for_variant.get(variant, {}).get(rid, "")
            started = time.time()
            text = ""
            error = None
            new_len = 0
            truncated = False
            try:
                messages = build_messages(row, variant, route, Path(str(row["image"])))
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
                "dataset": "Geometry3K-safe-route-200",
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
                "route": route if variant != "image_only" else "",
                "route_qc": route_qc(route) if variant != "image_only" else {},
                "error": error,
            }
            append_jsonl(args.out, rec)
            results.append(rec)
            done.add((rid, variant))
            print(f"{'OK' if correct else 'FAIL'}\t{rid}\t{variant}\tpred={pred}\tgold={row['answer']}", flush=True)

    summary = summarize(results)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
