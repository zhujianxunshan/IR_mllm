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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--model", default="PeijieWang/GDP-4B")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--resize-max-side", type=int, default=768)
    ap.add_argument("--max-pixels", type=int, default=262144)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    done = set()
    if args.out.exists():
        for item in read_jsonl(args.out):
            done.add(str(item["id"]))

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, max_pixels=args.max_pixels, local_files_only=True)
    kwargs: dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto", "trust_remote_code": True, "local_files_only": True}
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

    cache_dir = args.out.parent / "gdp_resized_images"
    prompt = (
        "Please parse the geometric diagram and provide its formal description. "
        "List visible points, lines, circles, length labels, angle labels, and geometric relations. "
        "Do not solve the problem."
    )
    for idx, row in enumerate(rows, 1):
        rid = str(row["id"])
        if rid in done:
            print(f"SKIP\t{idx}/{len(rows)}\t{rid}", flush=True)
            continue
        started = time.time()
        text = ""
        error = None
        new_tokens = 0
        image_path = Path(row["image"])
        try:
            image_path = resize_image(image_path, cache_dir, args.resize_max_side)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path.resolve()), "max_pixels": args.max_pixels},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
            new_tokens = len(trimmed[0])
            text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        rec = {
            "id": rid,
            "image": row["image"],
            "question": row.get("question", ""),
            "gdp_prompt": prompt,
            "gdp_text": text,
            "new_tokens": new_tokens,
            "duration_s": time.time() - started,
            "error": error,
        }
        append_jsonl(args.out, rec)
        print(f"{'ERR' if error else 'OK'}\t{idx}/{len(rows)}\t{rid}\ttokens={new_tokens}\tduration_s={rec['duration_s']:.2f}", flush=True)


if __name__ == "__main__":
    main()
