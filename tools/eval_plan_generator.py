#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SECTIONS = ["[KNOWN FACTS]", "[AUXILIARY CONSTRUCTION CANDIDATES]", "[THEOREM ROUTE]", "[EXECUTION]"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def section_score(text: str) -> float:
    return sum(1 for s in SECTIONS if s in text) / len(SECTIONS)


def cue_overlap(pred: str, target: str) -> float:
    keys = ["circle", "tangent", "arc", "right", "perpendicular", "parallel", "similar", "congruent", "area", "perimeter", "proportion", "Pythagorean"]
    p = {k for k in keys if re.search(k, pred, re.I)}
    t = {k for k in keys if re.search(k, target, re.I)}
    if not t:
        return 1.0 if not p else 0.0
    return len(p & t) / len(t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=420)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True, quantization_config=quant, device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    rows = read_jsonl(args.val)[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scores = []
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            prompt = f"<|im_start|>user\n{row['prompt'].strip()}<|im_end|>\n<|im_start|>assistant\n"
            enc = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            gen = tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
            rec = {
                "id": row["id"],
                "prediction": gen,
                "target": row["target"],
                "section_score": section_score(gen),
                "cue_overlap": cue_overlap(gen, row["target"]),
            }
            scores.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {
        "n": len(scores),
        "avg_section_score": sum(x["section_score"] for x in scores) / max(1, len(scores)),
        "avg_cue_overlap": sum(x["cue_overlap"] for x in scores) / max(1, len(scores)),
    }
    (args.out.with_suffix(".summary.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
