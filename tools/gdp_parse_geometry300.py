#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="multidataset_2x300_eval.jsonl")
    parser.add_argument("--model", default="PeijieWang/GDP-4B")
    parser.add_argument("--out", default="gdp4b_geometry300_parse.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    rows = [r for r in read_jsonl(Path(args.dataset)) if r.get("dataset") == "Geometry3K-300"]
    if args.limit:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for row in read_jsonl(out_path):
            done.add(row["id"])

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    kwargs = {"torch_dtype": "auto", "device_map": "auto", "trust_remote_code": True}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, **kwargs)
    model.eval()

    prompt = "Please parse the geometric diagram and provide its formal description."
    for row in rows:
        if row["id"] in done:
            print("SKIP", row["id"], flush=True)
            continue
        image_path = Path(row["image"]).resolve()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        started = time.time()
        text = ""
        error = None
        new_tokens = 0
        try:
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
            new_tokens = len(trimmed[0])
            text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        result = {
            "id": row["id"],
            "image": row["image"],
            "gdp_prompt": prompt,
            "gdp_text": text,
            "new_tokens": new_tokens,
            "duration_s": time.time() - started,
            "error": error,
        }
        append_jsonl(out_path, result)
        print(
            f"{'ERR' if error else 'OK'}\t{row['id']}\ttokens={new_tokens}\tduration_s={result['duration_s']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
