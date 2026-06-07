#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def normalize_answer(text: str, valid_letters: str) -> str:
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


def build_messages(row: dict, gdp_text: str, image_path: Path):
    choices = row.get("choices", [])
    choice_text = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
    valid_letters = "/".join(chr(65 + i) for i in range(len(choices)))
    prompt = f"""Solve this geometry multiple-choice problem.

Question:
{row["question"]}

Choices:
{choice_text}

Additional context:
GDP-4B automatic geometric parsing output. Use it only as auxiliary evidence; trust the image and question if there is conflict.
<GDP_PARSE>
{gdp_text}
</GDP_PARSE>

Return exactly one line and nothing else:
ANSWER: X

X must be one of {valid_letters}."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def summarize(rows: list[dict]):
    items = [r for r in rows if r.get("variant") == "image_gdp"]
    complete = [r for r in items if r.get("complete_response") and not r.get("error")]
    correct = [r for r in complete if r.get("correct")]
    return {
        "variant": "image_gdp",
        "n": len(items),
        "complete": len(complete),
        "correct": len(correct),
        "accuracy": len(correct) / len(items) if items else 0.0,
        "accuracy_on_complete": len(correct) / len(complete) if complete else 0.0,
        "errors": sum(1 for r in items if r.get("error")),
        "missing_answer_line": sum(1 for r in items if r.get("missing_answer_line")),
        "avg_duration_s": sum(float(r.get("duration_s", 0.0)) for r in items) / len(items) if items else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="multidataset_2x300_eval.jsonl")
    parser.add_argument("--gdp", default="gdp4b_geometry300_parse.jsonl")
    parser.add_argument("--model", default="~/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--out", default="qwen3vl8b_geometry300_image_gdp.jsonl")
    parser.add_argument("--summary", default="qwen3vl8b_geometry300_image_gdp_summary.json")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = [r for r in read_jsonl(Path(args.dataset)) if r.get("dataset") == "Geometry3K-300"]
    if args.limit:
        rows = rows[: args.limit]
    gdp_by_id = {r["id"]: r for r in read_jsonl(Path(args.gdp))}

    out_path = Path(args.out)
    done = set()
    results = []
    if out_path.exists():
        for row in read_jsonl(out_path):
            results.append(row)
            done.add(row["id"])

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_path = str(Path(args.model).expanduser()) if args.model.startswith("~") else args.model
    kwargs = {"device_map": "auto", "trust_remote_code": True, "torch_dtype": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    model.eval()

    for row in rows:
        if row["id"] in done:
            print("SKIP", row["id"], flush=True)
            continue
        gdp = gdp_by_id.get(row["id"], {})
        gdp_text = gdp.get("gdp_text", "")
        started = time.time()
        text = ""
        error = None
        try:
            messages = build_messages(row, gdp_text, Path(row["image"]))
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
            text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        valid_letters = "".join(chr(65 + i) for i in range(len(row.get("choices", []))))
        pred = normalize_answer(text, valid_letters or "ABCDE")
        complete = error is None and pred != ""
        correct = complete and pred == str(row["answer"]).strip().upper()
        result = {
            "dataset": "Geometry3K-300",
            "id": row["id"],
            "variant": "image_gdp",
            "answer": row["answer"],
            "prediction": pred,
            "correct": correct,
            "complete_response": complete,
            "missing_answer_line": pred == "",
            "duration_s": time.time() - started,
            "raw_response": text,
            "gdp_error": gdp.get("error"),
            "gdp_text_chars": len(gdp_text),
            "error": error,
        }
        append_jsonl(out_path, result)
        results.append(result)
        print(
            f"{'OK' if correct else 'FAIL'}\t{row['id']}\tpred={pred!r}\tgold={row['answer']!r}\t"
            f"complete={complete}\tduration_s={result['duration_s']:.2f}",
            flush=True,
        )

    Path(args.summary).write_text(json.dumps(summarize(results), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summarize(results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
