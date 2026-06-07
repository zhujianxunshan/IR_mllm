#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def formal(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("formal_logic") or {}


def symbols(text: str) -> set[str]:
    return set(re.findall(r"\b[A-Z]\b|[a-z]\b", text or ""))


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


def pred_score(pred: str, row: dict[str, Any]) -> tuple[int, int, int, int]:
    q_syms = symbols(row.get("question", ""))
    p_syms = symbols(pred)
    return (
        int("Find(" in pred),
        len(q_syms & p_syms),
        int(bool(re.search(r"(?<![A-Za-z])-?\d+(?:\.\d+)?", pred))),
        int(any(k in pred for k in ("Equals", "Parallel", "Perpendicular", "Similar", "Congruent", "Tangent", "Bisects", "IsMidpoint"))),
    )


def ranked_predicates(row: dict[str, Any], k: int = 12) -> list[str]:
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


def structured_facts(row: dict[str, Any]) -> str:
    preds = ranked_predicates(row, k=12)
    body = "\n".join(f"- {simplify_predicate(p)}" for p in preds) or "- none"
    return "[STRUCTURED FACTS]\nGeometry3K oracle formal predicates, shortened and answer-neutral:\n" + body


def theorem_cues(row: dict[str, Any], preds: list[str]) -> list[str]:
    text = " ".join([row.get("question", "")] + preds)
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
    if not cues:
        cues.append("Translate the visible relations into the shortest equation for the target.")
    return cues[:4]


def auxiliary_candidates(row: dict[str, Any]) -> list[str]:
    text = " ".join([row.get("question", "")] + ranked_predicates(row, k=12))
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
    if not out:
        out.append("Minimal construction: avoid adding extra lines unless a visible relation cannot be used directly.")
    return out[:4]


def construction_plan(row: dict[str, Any]) -> str:
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


def prompt_for(row: dict[str, Any], gdp_text: str | None, mode: str) -> str:
    choices = row.get("choices") or []
    choice_text = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    parts = [
        "You are a geometry intermediate-language generator.",
        "Given a geometry problem, produce a construction-aware intermediate language.",
        "The language must be answer-neutral: do not choose an option and do not solve numerically unless it is a stated fact.",
        "",
        f"Question: {row.get('question','')}",
        "Choices:",
        choice_text,
    ]
    if mode == "gdp_to_plan":
        parts += ["", "GDP automatic diagram parse:", (gdp_text or "No GDP parse available.")]
    elif mode == "text_to_plan":
        parts += ["", "Diagram parse: unavailable. Infer only high-level construction cues from the question and choices."]
    else:
        raise ValueError(mode)
    parts += ["", "Output only the construction-aware intermediate language."]
    return "\n".join(parts)


def make_rows(source_rows: list[dict[str, Any]], gdp_by_id: dict[str, str], mode: str) -> list[dict[str, Any]]:
    out = []
    for row in source_rows:
        rid = row["id"]
        if mode == "gdp_to_plan" and rid not in gdp_by_id:
            continue
        target = construction_plan(row)
        out.append(
            {
                "id": rid,
                "mode": mode,
                "image": row.get("image"),
                "question": row.get("question"),
                "choices": row.get("choices"),
                "prompt": prompt_for(row, gdp_by_id.get(rid), mode),
                "target": target,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", type=Path, default=Path("geometry3k_test_full.jsonl"))
    ap.add_argument("--gdp", type=Path, default=Path("gdp4b_geometry300_parse.jsonl"))
    ap.add_argument("--outdir", type=Path, default=Path("construction_plan_data"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    source_rows = read_jsonl(args.geometry)
    gdp_by_id = {}
    if args.gdp.exists():
        for row in read_jsonl(args.gdp):
            if row.get("error") is None:
                gdp_by_id[row["id"]] = row.get("gdp_text") or ""

    for mode in ("text_to_plan", "gdp_to_plan"):
        rows = make_rows(source_rows, gdp_by_id, mode)
        random.shuffle(rows)
        n_val = max(30, int(round(len(rows) * 0.12))) if len(rows) > 60 else max(1, len(rows) // 5)
        val = rows[:n_val]
        train = rows[n_val:]
        write_jsonl(args.outdir / mode / "train.jsonl", train)
        write_jsonl(args.outdir / mode / "val.jsonl", val)
        print(mode, "train", len(train), "val", len(val), "out", args.outdir / mode)


if __name__ == "__main__":
    main()
