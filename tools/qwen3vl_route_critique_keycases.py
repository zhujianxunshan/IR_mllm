#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


VARIANTS = ("critique_first",)


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


def theorem_route_only(route: str, limit: int = 8) -> str:
    match = re.search(r"\[THEOREM ROUTE\]\s*(.*?)(?=\n\[[A-Z_ ]+\]|\Z)", route or "", flags=re.S)
    text = match.group(1).strip() if match else (route or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[:limit])


def build_extra(route: str) -> str:
    minimal = theorem_route_only(route)
    return f"""[THEOREM ROUTE]
{minimal}

[USE POLICY]
- Critique before using: for each step in the theorem route, first check whether it is consistent with the diagram and the problem text.
- If a step is inconsistent, unsupported, irrelevant to the requested target, or nonsensical, you MUST discard it and rely on your own geometric knowledge from the image.
- Your first sentence must be either "Route step 1 is valid because ..." or "Route step 1 is invalid because ...".
- Treat the route as a weak hypothesis, not as an authoritative solution path."""


def build_messages(row: dict[str, Any], route: str, image_path: Path) -> list[dict[str, Any]]:
    choices = row.get("choices") or []
    valid_letters = "/".join(chr(65 + i) for i in range(len(choices)))
    prompt = f"""Solve this geometry multiple-choice problem.

Question:
{row.get("question", "")}

Choices:
{choices_text(row)}

Additional context:
{build_extra(route)}

First decide whether route step 1 is usable. Keep the reason under 12 words.
Then solve. Do not show calculations.

Output exactly two lines:
Route step 1 is valid/invalid because ...
ANSWER: X

X must be one of {valid_letters}."""
    return [{"role": "user", "content": [{"type": "image", "image": str(image_path.resolve())}, {"type": "text", "text": prompt}]}]


def summarize(results: list[dict[str, Any]], baseline: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    by_id = {str(row["id"]): row for row in results}
    out: dict[str, Any] = {
        "n": len(by_id),
        "overall": {},
        "by_case_type": {},
        "pairwise_vs_image_only": defaultdict(int),
        "pairwise_vs_image_route": defaultdict(int),
        "recover_route_losses": {},
        "preserve_route_wins": {},
    }
    complete = [row for row in by_id.values() if row.get("complete_response") and not row.get("error")]
    correct = [row for row in complete if row.get("correct")]
    out["overall"] = {
        "complete": len(complete),
        "correct": len(correct),
        "accuracy": len(correct) / len(by_id) if by_id else 0.0,
        "accuracy_on_complete": len(correct) / len(complete) if complete else 0.0,
        "avg_duration_s": sum(float(row.get("duration_s", 0.0)) for row in by_id.values()) / len(by_id) if by_id else 0.0,
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in by_id.values():
        grouped[row.get("case_type", "unknown")].append(row)
    for case_type, rows in grouped.items():
        ok = [row for row in rows if row.get("correct")]
        out["by_case_type"][case_type] = {
            "n": len(rows),
            "correct": len(ok),
            "accuracy": len(ok) / len(rows) if rows else 0.0,
        }

    for qid, row in by_id.items():
        base = baseline.get(qid, {})
        image_only = base.get("image_only")
        image_route = base.get("image_route")
        if image_only and image_only.get("correct") and not row.get("correct"):
            out["pairwise_vs_image_only"]["loss"] += 1
        elif image_only and (not image_only.get("correct")) and row.get("correct"):
            out["pairwise_vs_image_only"]["win"] += 1
        elif image_only and image_only.get("correct") and row.get("correct"):
            out["pairwise_vs_image_only"]["both_correct"] += 1
        elif image_only:
            out["pairwise_vs_image_only"]["both_wrong"] += 1

        if image_route and image_route.get("correct") and not row.get("correct"):
            out["pairwise_vs_image_route"]["loss"] += 1
        elif image_route and (not image_route.get("correct")) and row.get("correct"):
            out["pairwise_vs_image_route"]["win"] += 1
        elif image_route and image_route.get("correct") and row.get("correct"):
            out["pairwise_vs_image_route"]["both_correct"] += 1
        elif image_route:
            out["pairwise_vs_image_route"]["both_wrong"] += 1

    loss_cases = [row for row in by_id.values() if row.get("case_type") == "route_loss"]
    win_cases = [row for row in by_id.values() if row.get("case_type") == "route_win"]
    out["recover_route_losses"] = {
        "n": len(loss_cases),
        "recovered_to_correct": sum(1 for row in loss_cases if row.get("correct")),
        "still_wrong": sum(1 for row in loss_cases if not row.get("correct")),
        "recovery_rate": sum(1 for row in loss_cases if row.get("correct")) / len(loss_cases) if loss_cases else 0.0,
    }
    out["preserve_route_wins"] = {
        "n": len(win_cases),
        "still_correct": sum(1 for row in win_cases if row.get("correct")),
        "lost": sum(1 for row in win_cases if not row.get("correct")),
        "preservation_rate": sum(1 for row in win_cases if row.get("correct")) / len(win_cases) if win_cases else 0.0,
    }
    out["pairwise_vs_image_only"] = dict(out["pairwise_vs_image_only"])
    out["pairwise_vs_image_route"] = dict(out["pairwise_vs_image_route"])
    out["pairwise_vs_image_only"]["net"] = out["pairwise_vs_image_only"].get("win", 0) - out["pairwise_vs_image_only"].get("loss", 0)
    out["pairwise_vs_image_route"]["net"] = out["pairwise_vs_image_route"].get("win", 0) - out["pairwise_vs_image_route"].get("loss", 0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dataset", type=Path, default=Path("multidataset_2x300_eval.jsonl"))
    parser.add_argument("--extra-dataset", type=Path, default=Path("data/geometry3k_extra100_eval.jsonl"))
    parser.add_argument("--main-route-cache", type=Path, default=Path("results/geometry3k_trust_selection_route_cache_300.jsonl"))
    parser.add_argument("--extra-route-cache", type=Path, default=Path("results/geometry3k_extra100_route_cache.jsonl"))
    parser.add_argument("--main-baseline", type=Path, default=Path("results/qwen3vl8b_route_verifier_downstream200_4var.jsonl"))
    parser.add_argument("--extra-baseline", type=Path, default=Path("results/qwen3vl8b_route_verifier_extra100_4var.jsonl"))
    parser.add_argument("--case-ids", type=Path, required=True)
    parser.add_argument("--qwen-vl-model", default="/home/ubuntu/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--answer-max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    cases = json.loads(args.case_ids.read_text(encoding="utf-8"))
    wanted: dict[str, dict[str, str]] = {}
    for case_type in ("route_loss", "route_win"):
        for qid in cases.get(case_type, []):
            wanted[str(qid)] = {"case_type": case_type}

    rows: dict[str, dict[str, Any]] = {}
    for path, set_name in [(args.main_dataset, "main200"), (args.extra_dataset, "extra100")]:
        for row in read_jsonl(path):
            qid = str(row.get("id"))
            if qid in wanted:
                item = dict(row)
                item["set_name"] = set_name
                item["case_type"] = wanted[qid]["case_type"]
                rows[qid] = item

    route_by_id: dict[str, str] = {}
    for path in (args.main_route_cache, args.extra_route_cache):
        for row in read_jsonl(path):
            route_by_id[str(row.get("id"))] = row.get("generated_route", "")

    baseline: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in (args.main_baseline, args.extra_baseline):
        for row in read_jsonl(path):
            if row.get("variant") in {"image_only", "image_route"}:
                baseline[str(row.get("id"))][str(row.get("variant"))] = row

    missing = sorted(set(wanted) - set(rows))
    if missing:
        raise RuntimeError(f"Missing requested rows: {missing}")

    model_kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True, "torch_dtype": "auto"}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    processor = AutoProcessor.from_pretrained(args.qwen_vl_model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.qwen_vl_model, **model_kwargs)
    model.eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else ({int(eos)} if eos is not None else set())

    done = set()
    results: list[dict[str, Any]] = []
    if args.out.exists():
        for item in read_jsonl(args.out):
            results.append(item)
            done.add(str(item.get("id")))

    ordered = sorted(rows.values(), key=lambda row: (row["case_type"], row["set_name"], row["id"]))
    for row in ordered:
        qid = str(row["id"])
        if qid in done:
            continue
        started = time.time()
        text = ""
        error = None
        new_len = 0
        truncated = False
        try:
            image_path = Path(str(row["image"]))
            messages = build_messages(row, route_by_id.get(qid, ""), image_path)
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.answer_max_new_tokens, do_sample=False)
            trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
            new_tokens = trimmed[0].tolist()
            new_len = len(new_tokens)
            truncated = new_len >= args.answer_max_new_tokens and (not eos_ids or new_tokens[-1] not in eos_ids)
            text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        valid_letters = "".join(chr(65 + i) for i in range(len(row.get("choices") or []))) or "ABCDE"
        pred = normalize_answer(text, valid_letters)
        complete = error is None and not truncated and pred != ""
        correct = complete and pred == str(row["answer"]).strip().upper()
        record = {
            "dataset": "Geometry3K-route-critique-keycases",
            "set_name": row["set_name"],
            "case_type": row["case_type"],
            "id": qid,
            "question": row.get("question", ""),
            "choices": row.get("choices", []),
            "answer": row["answer"],
            "prediction": pred,
            "correct": correct,
            "complete_response": complete,
            "truncated": truncated,
            "new_tokens": new_len,
            "duration_s": time.time() - started,
            "raw_response": text,
            "route_input": route_by_id.get(qid, ""),
            "baseline_image_only": baseline.get(qid, {}).get("image_only", {}),
            "baseline_image_route": baseline.get(qid, {}).get("image_route", {}),
            "error": error,
        }
        append_jsonl(args.out, record)
        results.append(record)
        done.add(qid)
        print(
            f"{'OK' if correct else 'FAIL'}\t{row['case_type']}\t{row['set_name']}\t{qid}\t"
            f"pred={pred}\tgold={row['answer']}\tduration_s={record['duration_s']:.2f}",
            flush=True,
        )

    summary = summarize(results, baseline)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
