#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


VARIANTS = [
    "raw",
    "structured_facts",
    "perception_graph",
    "construction_plan",
    "geoguide_solver",
    "geoguide_cautious",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def normalize_answer(text: str, valid_letters: str = "ABCDE") -> str:
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


def formal(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("formal_logic") or {}


def all_predicates(row: dict[str, Any]) -> list[str]:
    preds: list[str] = []
    for key in ("dissolved_text_logic_form", "text_logic_form", "diagram_logic_form"):
        value = formal(row).get(key) or []
        if isinstance(value, list):
            preds.extend(str(x) for x in value if str(x).strip())
    seen: set[str] = set()
    out: list[str] = []
    for pred in preds:
        if pred not in seen:
            seen.add(pred)
            out.append(pred)
    return out


def symbols(text: str) -> set[str]:
    return set(re.findall(r"\b[A-Z]\b|[a-z]\b", text or ""))


def pred_score(pred: str, row: dict[str, Any]) -> tuple[int, int, int, int]:
    q_syms = symbols(row.get("question", ""))
    p_syms = symbols(pred)
    return (
        int("Find(" in pred),
        len(q_syms & p_syms),
        int(bool(re.search(r"(?<![A-Za-z])-?\d+(?:\.\d+)?", pred))),
        int(any(k in pred for k in ("Equals", "Parallel", "Perpendicular", "Similar", "Congruent", "Tangent", "Bisects", "IsMidpoint"))),
    )


def ranked_predicates(row: dict[str, Any], k: int = 10) -> list[str]:
    return sorted(all_predicates(row), key=lambda p: pred_score(p, row), reverse=True)[:k]


def simplify_predicate(pred: str) -> str:
    text = pred
    for old, new in [
        ("LengthOf(Line(", "length("),
        ("MeasureOf(Angle(", "angle("),
        ("RadiusOf(Circle(", "radius("),
        ("Line(", ""),
        ("Angle(", "angle("),
        ("Circle(", "circle("),
    ]:
        text = text.replace(old, new)
    return text.replace("))", ")")


def extracted_question_facts(row: dict[str, Any]) -> list[str]:
    question = row.get("question", "")
    facts: list[str] = []
    nums = re.findall(r"(?<![A-Za-z])-?\d+(?:\.\d+)?\s*(?:°|cm|m|in|ft|%|π)?", question)
    if nums:
        facts.append("Numbers/measurements mentioned: " + ", ".join(dict.fromkeys(nums[:10])))
    syms = sorted(symbols(question))
    if syms:
        facts.append("Named diagram symbols: " + ", ".join(syms[:16]))
    low = question.lower()
    cues = []
    for key in ["triangle", "angle", "parallel", "perpendicular", "midpoint", "bisect", "circle", "radius", "diameter", "tangent", "similar", "congruent", "area", "perimeter", "slope", "function", "chart", "table"]:
        if key in low:
            cues.append(key)
    if cues:
        facts.append("Detected relation words: " + ", ".join(cues))
    md = row.get("metadata") or {}
    meta_bits = [str(md.get(k)) for k in ("task", "context", "source", "grade") if md.get(k)]
    if meta_bits:
        facts.append("Dataset metadata: " + " | ".join(meta_bits))
    skills = md.get("skills")
    if isinstance(skills, list) and skills:
        facts.append("Skill tags: " + ", ".join(str(x) for x in skills[:6]))
    return facts or ["No reliable structured facts were extracted beyond the question text."]


def theorem_cues(row: dict[str, Any], preds: list[str]) -> list[str]:
    text = " ".join([row.get("question", ""), row.get("context", ""), row.get("task", "")] + preds)
    low = text.lower()
    cues: list[str] = []
    if "right" in low or "perpendicular" in low or "90" in text:
        cues.append("Check right-triangle, perpendicular-angle, Pythagorean, or trig relations.")
    if "similar" in low or "proportion" in low:
        cues.append("Match corresponding parts and set a proportion.")
    if "congruent" in low:
        cues.append("Transfer equal sides or angles between corresponding parts.")
    if any(x in low for x in ["circle", "chord", "arc", "tangent", "radius", "diameter"]):
        cues.append("Use circle angle, radius, chord, tangent, or arc rules.")
    if any(x in low for x in ["parallel", "parallelogram", "rectangle", "rhombus", "square"]):
        cues.append("Use parallel-line angle rules or quadrilateral side/diagonal properties.")
    if "area" in low:
        cues.append("Identify base-height, decomposition, or subtraction of regions.")
    if "perimeter" in low:
        cues.append("Solve missing side lengths before summing the boundary.")
    if any(x in low for x in ["chart", "plot", "table", "figure question answering"]):
        cues.append("Read the relevant visual marks carefully before doing arithmetic.")
    if not cues:
        cues.append("Translate the visible relations into the shortest equation for the target.")
    return cues[:4]


def structured_facts(row: dict[str, Any]) -> str:
    if formal(row):
        preds = ranked_predicates(row, k=12)
        body = "\n".join(f"- {simplify_predicate(p)}" for p in preds) or "none"
        return "[STRUCTURED FACTS]\nGeometry3K oracle formal predicates, shortened and answer-neutral:\n" + body
    return "[STRUCTURED FACTS]\n" + "\n".join(f"- {fact}" for fact in extracted_question_facts(row))


def bucket_predicates(preds: list[str]) -> dict[str, list[str]]:
    buckets = {"target": [], "visual": [], "metric": [], "relation": [], "shape": [], "other": []}
    for pred in preds:
        if "Find(" in pred:
            buckets["target"].append(pred)
        elif re.search(r"(?<![A-Za-z])-?\d+(?:\.\d+)?", pred):
            buckets["metric"].append(pred)
        elif any(key in pred for key in ("Triangle", "Circle", "Quadrilateral", "Rectangle", "Square", "Polygon", "Arc", "Line(", "Angle(")):
            buckets["shape"].append(pred)
        elif any(key in pred for key in ("Equals", "Parallel", "Perpendicular", "Similar", "Congruent", "Tangent", "Bisects", "IsMidpoint", "Collinear")):
            buckets["relation"].append(pred)
        elif any(key in pred for key in ("PointLiesOn", "Intersect", "Endpoint", "Center")):
            buckets["visual"].append(pred)
        else:
            buckets["other"].append(pred)
    return buckets


def bullet_block(title: str, items: list[str], limit: int = 6) -> str:
    if not items:
        return f"{title}\n- none"
    return f"{title}\n" + "\n".join(f"- {simplify_predicate(item)}" for item in items[:limit])


def perception_graph(row: dict[str, Any]) -> str:
    """Perception-first notes inspired by GeoPQA/GDP-style perception repair."""
    preds = ranked_predicates(row, k=16)
    if preds:
        buckets = bucket_predicates(preds)
        blocks = [
            "[PERCEPTION GRAPH: TARGET]",
            bullet_block("Target / unknown:", buckets["target"], limit=3),
            bullet_block("Visible objects and shapes:", buckets["shape"] + buckets["visual"], limit=7),
            bullet_block("Measured labels:", buckets["metric"], limit=6),
            bullet_block("Spatial / geometric relations:", buckets["relation"], limit=7),
        ]
    else:
        facts = extracted_question_facts(row)
        blocks = [
            "[PERCEPTION GRAPH: TARGET]",
            "Target / unknown:\n- infer from the question wording",
            "Visible objects and labels:\n" + "\n".join(f"- {fact}" for fact in facts),
            "Grounding rule:\n- re-read the image for the exact visual mark, label, count, chart value, or relation before solving",
        ]
    return (
        "Perception-first intermediate language. Its role is to make the model inspect visual evidence before algebra.\n\n"
        + "\n\n".join(blocks)
        + "\n\n[USE]\n- First verify which listed facts are visible in the image.\n- Then solve using only verified facts.\n- If this graph misses something visible, use the image and question."
    )


def auxiliary_candidates(row: dict[str, Any]) -> list[str]:
    text = " ".join([row.get("question", ""), row.get("context", ""), row.get("task", "")] + ranked_predicates(row, k=12))
    low = text.lower()
    out: list[str] = []
    if any(key in low for key in ("circle", "radius", "diameter", "tangent", "chord", "arc")):
        out.append("Circle construction: connect center to tangent point/chord endpoint; mark equal radii; use perpendicular radius-to-tangent/chord if applicable.")
    if any(key in low for key in ("similar", "proportion", "scale")):
        out.append("Similarity construction: identify corresponding vertices and align matching sides before setting a proportion.")
    if any(key in low for key in ("right", "perpendicular", "height", "altitude", "distance")):
        out.append("Right-triangle construction: drop or identify a perpendicular/altitude and use Pythagorean or trig relations.")
    if any(key in low for key in ("parallel", "angle", "transversal")):
        out.append("Angle construction: trace parallel/transversal angle pairs, vertical angles, supplements, and triangle angle sums.")
    if any(key in low for key in ("midpoint", "bisect", "median")):
        out.append("Bisection construction: split the relevant segment/angle into equal parts and transfer equality.")
    if "area" in low:
        out.append("Area construction: choose base-height or decompose the region into standard shapes.")
    if any(key in low for key in ("chart", "plot", "table", "bar", "line graph")):
        out.append("Visual-reading construction: locate the exact axis/table/category entry, then compare or compute.")
    if not out:
        out.append("Minimal construction: avoid adding extra lines unless a visible relation cannot be used directly.")
    return out[:4]


def construction_plan(row: dict[str, Any]) -> str:
    """Construction-aware plan inspired by Geoint-R1 and interleaved visual CoT work."""
    facts = structured_facts(row)
    candidates = "\n".join(f"- {item}" for item in auxiliary_candidates(row))
    cues = "\n".join(f"- {cue}" for cue in theorem_cues(row, ranked_predicates(row, k=12)))
    return (
        "Construction-aware intermediate language. It proposes answer-neutral helper constructions and a theorem route.\n\n"
        "[KNOWN FACTS]\n"
        + facts
        + "\n\n[AUXILIARY CONSTRUCTION CANDIDATES]\n"
        + candidates
        + "\n\n[THEOREM ROUTE]\n"
        + cues
        + "\n\n[EXECUTION]\n"
        "- Pick at most one useful construction; do not invent unsupported geometry.\n"
        "- After each visual/theorem step, re-check the relevant image mark or label.\n"
        "- Form the equation or comparison, compute the target, then choose one of the available options."
    )


def geoguide(row: dict[str, Any], cautious: bool) -> str:
    preds = ranked_predicates(row, k=10)
    facts = structured_facts(row)
    cues = "\n".join(f"- {cue}" for cue in theorem_cues(row, preds))
    checklist = (
        "- Identify the target quantity before looking at choices.\n"
        "- Use only facts supported by the image and question.\n"
        "- Build the shortest theorem/arithmetic relation for the target.\n"
        "- Compute or infer the target, then match it to one of the available options."
    )
    guard = ""
    if cautious:
        guard = (
            "\n\n[TRUST GUARD]\n"
            "- Treat the structured facts as auxiliary, not authoritative.\n"
            "- If a structured fact conflicts with the image or question, ignore that fact.\n"
            "- Do not pick an option only because it appears in the structured notes."
        )
    return f"GeoGuide-IL solver notes.\n\n{facts}\n\n[THEOREM / READING CUES]\n{cues}\n\n[SOLVING CHECKLIST]\n{checklist}{guard}"


def variant_context(row: dict[str, Any], variant: str) -> str:
    if variant == "raw":
        return "No extra structured representation is provided."
    if variant == "structured_facts":
        return structured_facts(row)
    if variant == "perception_graph":
        return perception_graph(row)
    if variant == "construction_plan":
        return construction_plan(row)
    if variant == "geoguide_solver":
        return geoguide(row, cautious=False)
    if variant == "geoguide_cautious":
        return geoguide(row, cautious=True)
    raise ValueError(variant)


def build_messages(row: dict[str, Any], variant: str, image_path: Path) -> list[dict[str, Any]]:
    choices = row.get("choices", [])
    choice_text = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
    valid_letters = "/".join(chr(65 + i) for i in range(len(choices)))
    prompt = f"""Solve this visual math multiple-choice problem.

Question:
{row["question"]}

Choices:
{choice_text}

Additional context:
{variant_context(row, variant)}

Return exactly one line and nothing else:
ANSWER: X

X must be one of {valid_letters}."""
    return [{"role": "user", "content": [{"type": "image", "image": str(image_path.resolve())}, {"type": "text", "text": prompt}]}]


def summarize(rows: list[dict[str, Any]], variants: list[str]) -> dict[str, Any]:
    by_dataset_variant: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_dataset_id: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        dataset = row.get("dataset", "unknown")
        by_dataset_variant[(dataset, row["variant"])].append(row)
        by_dataset_id[(dataset, row["id"])][row["variant"]] = row
    datasets = sorted({row.get("dataset", "unknown") for row in rows})
    summary: dict[str, Any] = {"datasets": {}, "overall": {}}
    for dataset in datasets:
        summary["datasets"][dataset] = {"variants": {}, "pairwise_vs_raw": {}}
        for variant in variants:
            items = by_dataset_variant.get((dataset, variant), [])
            complete = [r for r in items if r.get("complete_response") and not r.get("error")]
            correct = [r for r in complete if r.get("correct")]
            summary["datasets"][dataset]["variants"][variant] = {
                "n": len(items),
                "complete": len(complete),
                "correct": len(correct),
                "accuracy": len(correct) / len(items) if items else 0.0,
                "avg_duration_s": sum(float(r.get("duration_s", 0.0)) for r in items) / len(items) if items else 0.0,
                "errors": sum(1 for r in items if r.get("error")),
            }
        for variant in variants:
            if variant == "raw":
                continue
            win = loss = both_correct = both_wrong = incomplete = 0
            examples = {"win": [], "loss": []}
            for (ds, _), item in by_dataset_id.items():
                if ds != dataset:
                    continue
                raw = item.get("raw")
                other = item.get(variant)
                if not raw or not other or not raw.get("complete_response") or not other.get("complete_response"):
                    incomplete += 1
                elif other.get("correct") and not raw.get("correct"):
                    win += 1
                    if len(examples["win"]) < 6:
                        examples["win"].append(raw["id"])
                elif raw.get("correct") and not other.get("correct"):
                    loss += 1
                    if len(examples["loss"]) < 6:
                        examples["loss"].append(raw["id"])
                elif raw.get("correct") and other.get("correct"):
                    both_correct += 1
                else:
                    both_wrong += 1
            summary["datasets"][dataset]["pairwise_vs_raw"][variant] = {
                "win": win,
                "loss": loss,
                "net": win - loss,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "incomplete": incomplete,
                "examples": examples,
            }
    for variant in variants:
        items = [r for r in rows if r["variant"] == variant]
        correct = [r for r in items if r.get("correct")]
        summary["overall"][variant] = {"n": len(items), "correct": len(correct), "accuracy": len(correct) / len(items) if items else 0.0}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dataset = read_jsonl(Path(args.dataset))
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    out_path = Path(args.out)
    done: set[tuple[str, str]] = set()
    results: list[dict[str, Any]] = []
    if out_path.exists():
        for row in read_jsonl(out_path):
            results.append(row)
            done.add((row["id"], row["variant"]))

    kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True, "torch_dtype": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            compute_dtype = torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, **kwargs)
    model.eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else ({int(eos)} if eos is not None else set())

    for row in dataset:
        image_path = Path(str(row["image"]))
        for variant in variants:
            if (row["id"], variant) in done:
                print("SKIP", row["id"], variant, flush=True)
                continue
            print("CALL", row.get("dataset", "unknown"), row["id"], variant, flush=True)
            error = None
            text = ""
            new_len = 0
            truncated = False
            started = time.time()
            try:
                messages = build_messages(row, variant, image_path)
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
                new_tokens = trimmed[0].tolist()
                new_len = len(new_tokens)
                truncated = new_len >= args.max_new_tokens and (not eos_ids or new_tokens[-1] not in eos_ids)
                text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            duration = time.time() - started
            valid_letters = "".join(chr(65 + i) for i in range(len(row.get("choices", []))))
            pred = normalize_answer(text, valid_letters=valid_letters or "ABCDE")
            complete = error is None and not truncated and pred != ""
            correct = complete and pred == str(row["answer"]).strip().upper()
            result = {
                "dataset": row.get("dataset", "unknown"),
                "id": row["id"],
                "variant": variant,
                "answer": row["answer"],
                "prediction": pred,
                "correct": correct,
                "complete_response": complete,
                "truncated": truncated,
                "missing_answer_line": pred == "",
                "new_tokens": new_len,
                "duration_s": duration,
                "raw_response": text,
                "error": error,
                "source": row.get("source"),
                "task": row.get("task"),
                "context": row.get("context"),
            }
            append_jsonl(out_path, result)
            results.append(result)
            done.add((row["id"], variant))
            print(
                f"{'OK' if correct else 'FAIL'}\t{row.get('dataset')}\t{row['id']}\t{variant}\t"
                f"pred={pred!r}\tgold={row['answer']!r}\tcomplete={complete}\tduration_s={duration:.2f}",
                flush=True,
            )

    summary = summarize(results, variants)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
