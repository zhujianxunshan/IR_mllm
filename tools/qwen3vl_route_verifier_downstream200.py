#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


VARIANTS = ("image_only", "image_route", "verifier_filter", "verifier_rewrite")


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


def clip(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + "\n...[truncated]"


def extract_section(text: str, name: str) -> str:
    pattern = rf"\[{re.escape(name)}\]\s*(.*?)(?=\n\[[A-Z_]+\]|\Z)"
    m = re.search(pattern, text or "", flags=re.S)
    return m.group(1).strip() if m else ""


def make_verifier_prompt(row: dict[str, Any], gdp: str, route: str) -> str:
    return f"""You are a fine-grained route verifier and target-binding checker for geometry MLLM solving.
Given a geometry multiple-choice problem, GDP-4B automatic diagram parse, and a candidate theorem route, decide whether the route is target-matched, condition-supported, and safe to show to a downstream multimodal solver.

Do not solve the problem numerically. Do not choose the final answer.

Question:
{row.get("question", "")}

Choices:
{choices_text(row)}

GDP-4B automatic parse:
{clip(gdp, 2000)}

Candidate theorem route:
{clip(route, 1600)}

Output exactly these sections:
[ROUTE_VALIDITY]
[TARGET_BINDING]
[CONDITION_CHECK]
[USE_DECISION]
[REVISED_HINT]
"""


def generate_verifier(row: dict[str, Any], gdp: str, route: str, tok: Any, model: Any, max_new_tokens: int) -> str:
    import torch

    prompt = f"<|im_start|>user\n{make_verifier_prompt(row, gdp, route).strip()}<|im_end|>\n<|im_start|>assistant\n"
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)


def build_extra(variant: str, route: str, report: str) -> str:
    if variant == "image_only":
        return "No additional structured context is provided."
    if variant == "image_route":
        return (
            "Generated theorem-route hypothesis. Use it only after checking the image and question; ignore it if unsupported.\n"
            f"<THEOREM_ROUTE>\n{route}\n</THEOREM_ROUTE>"
        )

    decision = extract_section(report, "USE_DECISION").lower()
    revised = extract_section(report, "REVISED_HINT")
    validity = extract_section(report, "ROUTE_VALIDITY")
    target = extract_section(report, "TARGET_BINDING")
    cond = extract_section(report, "CONDITION_CHECK")

    # Strict filter: if the verifier is not confidently positive, fall back to
    # the exact raw-image setting. Do not expose rejection rationales, GDP text,
    # or low-confidence route text to the downstream MLLM.
    accept_filter = decision.strip() == "use_route"
    if variant == "verifier_filter":
        if not accept_filter:
            return "No additional structured context is provided."
        return (
            "Route verifier accepted this as a low-confidence route hint. Verify it against the image before use.\n"
            f"[VERIFIER_VALIDITY]\n{validity}\n\n[SAFE_ROUTE_HINT]\n{clip(revised or route, 1200)}"
        )
    if variant == "verifier_rewrite":
        if revised:
            return (
                "Verifier-rewritten target-bound hint. Use only after checking the image and problem text.\n"
                f"[VERIFIER_VALIDITY]\n{validity}\n\n[TARGET_BINDING]\n{clip(target, 700)}\n\n[CONDITION_CHECK]\n{clip(cond, 700)}\n\n[REVISED_HINT]\n{clip(revised, 1200)}"
            )
        return "No reliable route rewrite is available. Solve from the image and question."
    raise ValueError(variant)


def build_messages(row: dict[str, Any], variant: str, route: str, report: str, image_path: Path) -> list[dict[str, Any]]:
    choices = row.get("choices") or []
    valid_letters = "/".join(chr(65 + i) for i in range(len(choices)))
    prompt = f"""Solve this geometry multiple-choice problem.

Question:
{row.get("question", "")}

Choices:
{choices_text(row)}

Additional context:
{build_extra(variant, route, report)}

Return exactly one line and nothing else:
ANSWER: X

X must be one of {valid_letters}."""
    return [{"role": "user", "content": [{"type": "image", "image": str(image_path.resolve())}, {"type": "text", "text": prompt}]}]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in results:
        by_variant[r["variant"]].append(r)
        by_id[str(r["id"])][r["variant"]] = r
    out: dict[str, Any] = {"variants": {}, "pairwise_vs_image_only": {}, "route_harm_reduction": {}, "n_unique_ids": len(by_id)}
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
    pairwise: dict[str, dict[str, int]] = {}
    for variant in VARIANTS:
        if variant == "image_only":
            continue
        win = loss = both_correct = both_wrong = incomplete = 0
        for pair in by_id.values():
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
        pairwise[variant] = {
            "win": win,
            "loss": loss,
            "net": win - loss,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "incomplete": incomplete,
        }
    out["pairwise_vs_image_only"] = pairwise
    route = pairwise.get("image_route", {})
    for variant in ["verifier_filter", "verifier_rewrite"]:
        cur = pairwise.get(variant, {})
        if route and cur:
            out["route_harm_reduction"][variant] = {
                "route_win_retained": cur.get("win", 0),
                "route_loss_reduced_by": route.get("loss", 0) - cur.get("loss", 0),
                "net_delta_vs_route": cur.get("net", 0) - route.get("net", 0),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("multidataset_2x300_eval.jsonl"))
    ap.add_argument("--dataset-name", default="Geometry3K-300")
    ap.add_argument("--gdp", type=Path, default=Path("gdp4b_geometry300_parse.jsonl"))
    ap.add_argument("--route-cache", type=Path, required=True)
    ap.add_argument("--test-ids", type=Path, required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--qwen-vl-model", default="~/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--verifier-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--verifier-max-new-tokens", type=int, default=420)
    ap.add_argument("--answer-max-new-tokens", type=int, default=32)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    wanted = set(json.loads(args.test_ids.read_text(encoding="utf-8"))) if args.test_ids.exists() else set()
    rows_all = [r for r in read_jsonl(args.dataset) if r.get("dataset") == args.dataset_name]
    rows = [r for r in rows_all if str(r["id"]) in wanted][: args.limit]
    if len(rows) < args.limit:
        raise RuntimeError(f"Need {args.limit} test rows, got {len(rows)}")

    gdp_by_id = {str(r["id"]): (r.get("gdp_text") or r.get("gdp_response") or "") for r in read_jsonl(args.gdp) if not r.get("error")}
    route_by_id = {str(r["id"]): r.get("generated_route", "") for r in read_jsonl(args.route_cache)}

    verifier_by_id: dict[str, str] = {}
    if args.verifier_cache.exists():
        for r in read_jsonl(args.verifier_cache):
            verifier_by_id[str(r["id"])] = r.get("verifier_report", "")

    missing = [r for r in rows if str(r["id"]) not in verifier_by_id]
    if missing:
        tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True, local_files_only=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        base = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True, quantization_config=quant, device_map="auto")
        verifier_model = PeftModel.from_pretrained(base, args.adapter)
        verifier_model.eval()
        for row in missing:
            rid = str(row["id"])
            started = time.time()
            report = generate_verifier(row, gdp_by_id.get(rid, ""), route_by_id.get(rid, ""), tok, verifier_model, args.verifier_max_new_tokens)
            verifier_by_id[rid] = report
            append_jsonl(args.verifier_cache, {"id": rid, "verifier_report": report, "duration_s": time.time() - started})
            print(f"VERIFY\t{rid}\tchars={len(report)}", flush=True)
        del verifier_model, base
        torch.cuda.empty_cache()

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
        for variant in VARIANTS:
            if (rid, variant) in done:
                continue
            started = time.time()
            text = ""
            error = None
            new_len = 0
            truncated = False
            try:
                messages = build_messages(row, variant, route_by_id.get(rid, ""), verifier_by_id.get(rid, ""), image_path)
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
                "dataset": "Geometry3K-route-verifier-200",
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
                "verifier_report": verifier_by_id.get(rid, "") if variant.startswith("verifier") else "",
            }
            append_jsonl(args.out, rec)
            results.append(rec)
            done.add((rid, variant))
            print(f"{'OK' if correct else 'FAIL'}\t{rid}\t{variant}\tpred={pred}\tgold={row['answer']}\tduration_s={rec['duration_s']:.2f}", flush=True)

    summary = summarize(results)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
