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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--model", default="PeijieWang/GDP-4B")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    done = {str(r["id"]) for r in read_jsonl(args.out)} if args.out.exists() else set()

    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    kwargs: dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto", "trust_remote_code": True}
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, **kwargs)
    model.eval()

    prompt = "Please parse the geometric diagram and provide its formal description."
    for row in rows:
        rid = str(row["id"])
        if rid in done:
            print("SKIP", rid, flush=True)
            continue
        messages = [{"role": "user", "content": [{"type": "image", "image": str(Path(row["image"]).resolve())}, {"type": "text", "text": prompt}]}]
        started = time.time()
        text = ""
        error = None
        new_tokens = 0
        try:
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
            new_tokens = len(trimmed[0])
            text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        append_jsonl(args.out, {"id": rid, "image": row["image"], "gdp_prompt": prompt, "gdp_text": text, "new_tokens": new_tokens, "duration_s": time.time() - started, "error": error})
        print(f"{'ERR' if error else 'OK'}\t{rid}\ttokens={new_tokens}\tduration_s={time.time() - started:.2f}", flush=True)


if __name__ == "__main__":
    main()
