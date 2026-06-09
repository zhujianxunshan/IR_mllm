#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def theorem_lines(route: str) -> list[str]:
    match = re.search(r"\[THEOREM ROUTE\]\s*(.*?)(?=\n\[[A-Z_ ]+\]|\Z)", route or "", flags=re.S)
    body = match.group(1).strip() if match else (route or "").strip()
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    clean = []
    for line in lines:
        line = re.sub(r"^\d+\.\s*Use\s+", "", line, flags=re.I).strip().rstrip(".")
        if line and line.lower() not in {x.lower() for x in clean}:
            clean.append(line)
    return clean[:6]


def candidate_bank(route: str) -> list[str]:
    text = " ".join(theorem_lines(route)).lower()
    candidates: list[str] = []

    def add(item: str) -> None:
        if item not in candidates:
            candidates.append(item)

    if "parallelogram" in text:
        add("Candidate A: In a parallelogram, opposite sides are equal. Apply if the unknown is a side length.")
        add("Candidate B: In a parallelogram, opposite angles are equal. Apply if the unknown is an angle.")
        add("Candidate C: In a parallelogram, diagonals bisect each other. Apply if diagonal parts are given.")
        add("Candidate D: Consecutive angles in a parallelogram are supplementary. Apply if adjacent angles are involved.")
    if "parallel" in text:
        add("Candidate A: Corresponding angles formed by parallel lines are equal. Apply only if parallel lines are marked or stated.")
        add("Candidate B: Alternate interior angles formed by parallel lines are equal. Apply only if the target angle is alternate interior.")
        add("Candidate C: Same-side interior angles formed by parallel lines are supplementary. Apply only if the two angles lie on the same side of a transversal.")
    if "midsegment" in text:
        add("Candidate A: A trapezoid midsegment equals the average of the two bases. Apply if midpoints of legs and two bases are involved.")
        add("Candidate B: A triangle midsegment is parallel to the third side and half its length. Apply if the figure is a triangle.")
    if "similar" in text:
        add("Candidate A: Corresponding angles of similar polygons are equal. Apply if the unknown is an angle.")
        add("Candidate B: Corresponding side lengths of similar polygons are proportional. Apply if the unknown is a length.")
        add("Candidate C: Perimeters of similar polygons follow the same scale factor. Apply if perimeter is involved.")
    if "sine" in text or "area" in text:
        add("Candidate A: Triangle area is 1/2 ab sin(C). Apply if two sides and included angle are given.")
        add("Candidate B: Parallelogram area is ab sin(C) or base times height. Apply if solving area or height of a parallelogram.")
        add("Candidate C: Trapezoid area is (b1+b2)h/2. Apply if solving area or height of a trapezoid.")
        add("Candidate D: The sine ratio in a right triangle is opposite/hypotenuse. Apply if the target is a side or angle in a right triangle.")
    if "circle" in text or "arc" in text or "tangent" in text:
        add("Candidate A: A tangent is perpendicular to the radius at the point of tangency. Apply if a radius to tangent point is shown.")
        add("Candidate B: An exterior secant/tangent angle equals half the difference of intercepted arcs. Apply if the target is an angle.")
        add("Candidate C: A central angle has the same measure as its intercepted arc. Apply if the center is the angle vertex.")
        add("Candidate D: Circle area is pi r^2. Apply only if the target is an area.")
        add("Candidate E: Tangent-secant power theorem relates external segments. Apply if segment lengths from an external point are given.")
    if "line addition" in text:
        add("Candidate A: Segment addition: if B lies between A and C, then AB + BC = AC. Apply if collinear segment parts are given.")
        add("Candidate B: Angle addition: adjacent angles sum to the whole angle. Apply if adjacent angle parts are given.")
    if "equilateral" in text:
        add("Candidate A: All sides of an equilateral triangle are equal. Apply if the unknown is a side length.")
        add("Candidate B: All angles of an equilateral triangle are 60 degrees. Apply if the unknown is an angle.")
    if "complementary" in text:
        add("Candidate A: Complementary angles sum to 90 degrees. Apply only if a right angle or complementary relation is visible/stated.")
        add("Candidate B: Supplementary linear-pair angles sum to 180 degrees. Apply if angles form a straight line.")
    if not candidates:
        for i, line in enumerate(theorem_lines(route)[:4], 1):
            add(f"Candidate {chr(64+i)}: {line}. Apply only if its required conditions are visible or stated.")

    add("Candidate Z: None of the above. Use the image and problem text only if no candidate directly matches the target.")
    return candidates[:8]


def build_messages(row: dict[str, Any], route: str, image_path: Path) -> list[dict[str, Any]]:
    choices = row.get("choices") or []
    valid_letters = "/".join(chr(65 + i) for i in range(len(choices)))
    candidates = "\n".join(candidate_bank(route))
    prompt = f"""Solve this geometry multiple-choice problem.

Question:
{row.get("question", "")}

Choices:
{choices_text(row)}

[THEOREM CANDIDATES]
Your task is to select the most appropriate theorem, not to blindly execute a route.
{candidates}

Instruction:
1. Analyze the question target and the given information.
2. Select the candidate that directly helps solve the unknown. If none matches, select Candidate Z.
3. Keep your selection reason very short, then solve.

Output exactly two lines:
SELECTED: Candidate X because ...
ANSWER: Y

Y must be one of {valid_letters}."""
    return [{"role": "user", "content": [{"type": "image", "image": str(image_path.resolve())}, {"type": "text", "text": prompt}]}]


def summarize(results: list[dict[str, Any]], baseline: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    by_id = {str(row["id"]): row for row in results}
    out: dict[str, Any] = {"n": len(by_id), "overall": {}, "by_case_type": {}, "pairwise_vs_image_only": defaultdict(int), "pairwise_vs_image_route": defaultdict(int)}
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
        out["by_case_type"][case_type] = {"n": len(rows), "correct": len(ok), "accuracy": len(ok) / len(rows) if rows else 0.0}
    for qid, row in by_id.items():
        base = baseline.get(qid, {})
        for variant, key in [("image_only", "pairwise_vs_image_only"), ("image_route", "pairwise_vs_image_route")]:
            old = base.get(variant)
            if not old:
                continue
            if old.get("correct") and not row.get("correct"):
                out[key]["loss"] += 1
            elif (not old.get("correct")) and row.get("correct"):
                out[key]["win"] += 1
            elif old.get("correct") and row.get("correct"):
                out[key]["both_correct"] += 1
            else:
                out[key]["both_wrong"] += 1
    for key in ["pairwise_vs_image_only", "pairwise_vs_image_route"]:
        out[key] = dict(out[key])
        out[key]["net"] = out[key].get("win", 0) - out[key].get("loss", 0)
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
    parser.add_argument("--answer-max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    cases = json.loads(args.case_ids.read_text(encoding="utf-8"))
    wanted = {str(qid): "route_loss" for qid in cases.get("route_loss", [])}
    wanted.update({str(qid): "route_win" for qid in cases.get("route_win", [])})

    rows: dict[str, dict[str, Any]] = {}
    for path, set_name in [(args.main_dataset, "main200"), (args.extra_dataset, "extra100")]:
        for row in read_jsonl(path):
            qid = str(row.get("id"))
            if qid in wanted:
                item = dict(row)
                item["set_name"] = set_name
                item["case_type"] = wanted[qid]
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
            messages = build_messages(row, route_by_id.get(qid, ""), Path(str(row["image"])))
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.answer_max_new_tokens, do_sample=False)
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated)]
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
            "dataset": "Geometry3K-theorem-candidates-keycases",
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
            "candidate_list": candidate_bank(route_by_id.get(qid, "")),
            "route_input": route_by_id.get(qid, ""),
            "baseline_image_only": baseline.get(qid, {}).get("image_only", {}),
            "baseline_image_route": baseline.get(qid, {}).get("image_route", {}),
            "error": error,
        }
        append_jsonl(args.out, record)
        results.append(record)
        done.add(qid)
        print(f"{'OK' if correct else 'FAIL'}\t{row['case_type']}\t{row['set_name']}\t{qid}\tpred={pred}\tgold={row['answer']}\tduration_s={record['duration_s']:.2f}", flush=True)

    summary = summarize(results, baseline)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
