#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def choices_text(row: dict[str, Any]) -> str:
    return "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(row.get("choices") or []))


def route_prompt(row: dict[str, Any], gdp_text: str) -> str:
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
{choices_text(row)}

GDP-4B automatic diagram parse:
{gdp_text}
"""


def generate_route(row: dict[str, Any], gdp_text: str, tok: Any, model: Any, max_new_tokens: int) -> str:
    import torch

    prompt = f"<|im_start|>user\n{route_prompt(row, gdp_text).strip()}<|im_end|>\n<|im_start|>assistant\n"
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--gdp", type=Path, required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = read_jsonl(args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    gdp_by_id = {str(r["id"]): (r.get("gdp_text") or r.get("gdp_response") or "") for r in read_jsonl(args.gdp) if not r.get("error")}
    done = {str(r["id"]) for r in read_jsonl(args.out)} if args.out.exists() else set()

    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True, quantization_config=quant, device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    for row in rows:
        rid = str(row["id"])
        if rid in done:
            print("SKIP", rid, flush=True)
            continue
        if rid not in gdp_by_id:
            print("MISS_GDP", rid, flush=True)
            continue
        started = time.time()
        route = generate_route(row, gdp_by_id[rid], tok, model, args.max_new_tokens)
        append_jsonl(args.out, {"id": rid, "generated_route": route, "duration_s": time.time() - started})
        print(f"ROUTE\t{rid}\tchars={len(route)}\tduration_s={time.time() - started:.2f}", flush=True)


if __name__ == "__main__":
    main()
