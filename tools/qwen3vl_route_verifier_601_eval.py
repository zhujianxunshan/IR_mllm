#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VARIANTS = ("image_only", "compact_route", "verifier_gate", "verifier_rewrite")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def clip(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + "\n...[truncated]"


def choices_text(row: dict[str, Any]) -> str:
    return "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(row.get("choices") or []))


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


def extract_section(text: str, name: str) -> str:
    pattern = rf"\[{re.escape(name)}\]\s*(.*?)(?=\n\[[A-Z_]+\]|\Z)"
    m = re.search(pattern, text or "", flags=re.S)
    return m.group(1).strip() if m else ""


def verifier_prompt(row: dict[str, Any], gdp: str, route: str) -> str:
    return f"""You are a route verifier and target-binding checker for multimodal geometry solving.
Given a geometry problem, a GDP-4B automatic diagram parse, and a candidate theorem route, decide whether the route is safe to show to a downstream multimodal solver.

Do not solve the problem. Do not choose the final answer.

Question:
{row.get("question", "")}

Choices:
{choices_text(row)}

GDP-4B automatic parse:
{clip(gdp, 2200)}

Candidate theorem route:
{clip(route, 1800)}

Output exactly these sections:
[ROUTE_VALIDITY]
[TARGET_BINDING]
[CONDITION_CHECK]
[USE_DECISION]
[REVISED_HINT]
"""


def generate_verifier(row: dict[str, Any], gdp: str, route: str, tok: Any, model: Any, max_new_tokens: int) -> str:
    import torch

    text = f"<|im_start|>user\n{verifier_prompt(row, gdp, route).strip()}<|im_end|>\n<|im_start|>assistant\n"
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)


def build_extra(variant: str, route: str, report: str) -> str:
    if variant == "image_only":
        return "No additional structured context is provided."
    if variant == "compact_route":
        return (
            "Generated theorem-route hypothesis. Treat it as a weak hint and ignore it if unsupported by the image.\n"
            f"<THEOREM_ROUTE>\n{clip(route, 1400)}\n</THEOREM_ROUTE>"
        )
    decision = extract_section(report, "USE_DECISION").strip().lower()
    revised = extract_section(report, "REVISED_HINT")
    validity = extract_section(report, "ROUTE_VALIDITY")
    target = extract_section(report, "TARGET_BINDING")
    condition = extract_section(report, "CONDITION_CHECK")
    if variant == "verifier_gate":
        if decision not in {"use_route", "use_route_cautious"}:
            return "No additional structured context is provided."
        return (
            "A verifier accepted this route as a weak hint. Verify it against the image before use.\n"
            f"[VERIFIER_VALIDITY]\n{clip(validity, 500)}\n\n[SAFE_ROUTE_HINT]\n{clip(revised or route, 1200)}"
        )
    if variant == "verifier_rewrite":
        if decision in {"reject_route", "use_image_first"} or not revised:
            return "No additional structured context is provided."
        return (
            "Verifier-rewritten target-bound hint. Use only after checking the image and problem text.\n"
            f"[VERIFIER_VALIDITY]\n{clip(validity, 500)}\n\n[TARGET_BINDING]\n{clip(target, 600)}\n\n[CONDITION_CHECK]\n{clip(condition, 600)}\n\n[REVISED_HINT]\n{clip(revised, 1200)}"
        )
    raise ValueError(variant)


def build_messages(row: dict[str, Any], variant: str, route: str, report: str, image_path: Path) -> list[dict[str, Any]]:
    letters = "/".join(chr(65 + i) for i in range(len(row.get("choices") or [])))
    prompt = f"""Solve this geometry multiple-choice problem.

Question:
{row.get("question", "")}

Choices:
{choices_text(row)}

Additional context:
{build_extra(variant, route, report)}

Return exactly one line and nothing else:
ANSWER: X

X must be one of {letters}."""
    return [{"role": "user", "content": [{"type": "image", "image": str(image_path.resolve())}, {"type": "text", "text": prompt}]}]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    decisions: Counter[str] = Counter()

    for r in results:
        by_variant[r["variant"]].append(r)
        by_id[str(r["id"])][r["variant"]] = r
        if r["variant"].startswith("verifier"):
            decisions[extract_section(r.get("verifier_report", ""), "USE_DECISION").strip().lower() or "missing"] += 1
    out: dict[str, Any] = {"variants": {}, "pairwise_vs_image_only": {}, "route_harm_reduction": {}, "verifier_decisions": dict(decisions), "n_unique_ids": len(by_id)}
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
            "errors": sum(1 for r in items if r.get("error")),
        }
    pairwise = {}
    for variant in VARIANTS:
        if variant == "image_only":
            continue
        win = loss = both_correct = both_wrong = incomplete = 0
        for pair in by_id.values():
            base, cur = pair.get("image_only"), pair.get(variant)
            if not base or not cur or not base.get("complete_response") or not cur.get("complete_response"):
                incomplete += 1
            elif cur.get("correct") and not base.get("correct"):
                win += 1
            elif base.get("correct") and not cur.get("correct"):
                loss += 1
            elif base.get("correct") and cur.get("correct"):
                both_correct += 1
            else:
                both_wrong += 1
        pairwise[variant] = {"win": win, "loss": loss, "net": win - loss, "both_correct": both_correct, "both_wrong": both_wrong, "incomplete": incomplete}
    out["pairwise_vs_image_only"] = pairwise
    route = pairwise.get("compact_route", {})
    for variant in ("verifier_gate", "verifier_rewrite"):
        cur = pairwise.get(variant, {})
        out["route_harm_reduction"][variant] = {
            "route_win_retained": cur.get("win", 0),
            "route_loss_reduced_by": route.get("loss", 0) - cur.get("loss", 0),
            "net_delta_vs_compact_route": cur.get("net", 0) - route.get("net", 0),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("geometry3k_test_full.jsonl"))
    ap.add_argument("--test-ids", type=Path, default=Path("data/route_verifier_601/test_ids.json"))
    ap.add_argument("--gdp", type=Path, default=Path("results/gdp4b_geometry3k_test601_parse.jsonl"))
    ap.add_argument("--route-cache", type=Path, default=Path("results/safe_route_compact_geometry3k_test601.jsonl"))
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--qwen-vl-model", default="~/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--verifier-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--variants", default="image_only,compact_route,verifier_gate,verifier_rewrite")
    ap.add_argument("--verifier-max-new-tokens", type=int, default=360)
    ap.add_argument("--answer-max-new-tokens", type=int, default=32)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")

    wanted = [str(x) for x in json.loads(args.test_ids.read_text(encoding="utf-8"))]
    wanted_set = set(wanted[: args.limit])
    rows = [r for r in read_jsonl(args.dataset) if str(r.get("id")) in wanted_set]
    rows.sort(key=lambda r: wanted.index(str(r["id"])) if str(r["id"]) in wanted else 10**9)
    rows = rows[: args.limit]
    if len(rows) < args.limit:
        raise RuntimeError(f"Need {args.limit} rows, got {len(rows)}")

    gdp_by_id = {str(r["id"]): (r.get("gdp_text") or r.get("gdp_response") or "") for r in read_jsonl(args.gdp) if not r.get("error")}
    route_by_id = {str(r["id"]): r.get("generated_route", "") for r in read_jsonl(args.route_cache)}

    verifier_by_id: dict[str, str] = {}
    if args.verifier_cache.exists():
        for r in read_jsonl(args.verifier_cache):
            verifier_by_id[str(r["id"])] = r.get("verifier_report", "")

    need_verifier = any(v.startswith("verifier") for v in variants)
    missing = [r for r in rows if need_verifier and str(r["id"]) not in verifier_by_id]
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
            rep = generate_verifier(row, gdp_by_id.get(rid, ""), route_by_id.get(rid, ""), tok, verifier_model, args.verifier_max_new_tokens)
            verifier_by_id[rid] = rep
            append_jsonl(args.verifier_cache, {"id": rid, "verifier_report": rep, "duration_s": time.time() - started})
            print(f"VERIFY\t{rid}\tdecision={extract_section(rep, 'USE_DECISION')}", flush=True)
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
        for variant in variants:
            if (rid, variant) in done:
                continue
            started = time.time()
            text = ""
            error = None
            new_len = 0
            truncated = False
            try:
                messages = build_messages(row, variant, route_by_id.get(rid, ""), verifier_by_id.get(rid, ""), Path(str(row["image"])))
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(vlm.device)
                with torch.inference_mode():
                    generated = vlm.generate(**inputs, max_new_tokens=args.answer_max_new_tokens, do_sample=False)
                trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
                toks = trimmed[0].tolist()
                new_len = len(toks)
                truncated = new_len >= args.answer_max_new_tokens and (not eos_ids or toks[-1] not in eos_ids)
                text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            valid_letters = "".join(chr(65 + i) for i in range(len(row.get("choices") or []))) or "ABCDE"
            pred = normalize_answer(text, valid_letters)
            complete = error is None and not truncated and pred != ""
            correct = complete and pred == str(row["answer"]).strip().upper()
            rec = {
                "dataset": "Geometry3K-route-verifier-601-heldout",
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
                "route": route_by_id.get(rid, "") if variant == "compact_route" else "",
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
