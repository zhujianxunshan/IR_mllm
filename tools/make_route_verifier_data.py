#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_DECISIONS = {
    "use_route",
    "rewrite_route",
    "use_route_cautious",
    "use_image_first",
    "reject_route",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clip(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + "\n...[truncated]"


def choices_text(row: dict[str, Any]) -> str:
    return "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(row.get("choices") or []))


def extract_theorem_route(route: str) -> str:
    m = re.search(r"\[THEOREM ROUTE\]\s*(.*?)(?=\n\[[A-Z_ ]+\]|\Z)", route or "", flags=re.S)
    return m.group(1).strip() if m else (route or "").strip()


def theorem_names(route: str) -> list[str]:
    names: list[str] = []
    for line in extract_theorem_route(route).splitlines():
        line = line.strip(" -\t")
        line = re.sub(r"^\d+\.\s*", "", line)
        line = re.sub(r"^Use\s+", "", line, flags=re.I)
        line = line.strip(". ")
        if line:
            names.append(line)
    return names


def route_family(route: str) -> set[str]:
    low = route.lower()
    fams = set()
    for key in [
        "angle",
        "arc",
        "area",
        "circle",
        "cosine",
        "line addition",
        "midsegment",
        "parallel",
        "parallelogram",
        "pythagorean",
        "right triangle",
        "similar",
        "sine",
        "tangent",
        "trapezoid",
        "triangle",
    ]:
        if key in low:
            fams.add(key)
    return fams


def infer_target(question: str, goal_cdl: str = "") -> str:
    text = f"{question} {goal_cdl}".lower()
    if "area" in text:
        return "area"
    if "perimeter" in text:
        return "perimeter"
    if "arc" in text or "widehat" in text:
        return "arc measure"
    if "angle" in text or "∠" in text or "measureofangle" in text:
        return "angle measure"
    if "length" in text or "valueoflength" in text:
        return "length"
    if re.search(r"\bfind x\b|\bvalue of x\b", text):
        return "unknown variable x"
    return "target quantity stated in the question"


def formalgeo_prompt(row: dict[str, Any], candidate_route: str) -> str:
    return f"""You are a fine-grained route verifier and target-binding checker for geometry MLLM solving.
Given a geometry question, structured diagram facts, and a candidate theorem route, decide whether the route is target-matched, condition-supported, and safe to show to a downstream multimodal solver.

Do not solve the problem numerically. Do not choose the final answer.

Question:
{row.get("question", "")}

Structured diagram facts:
{clip(row.get("structured_parse", ""), 2200)}

Candidate theorem route:
{clip(candidate_route, 1600)}

Output exactly these sections:
[ROUTE_VALIDITY]
[TARGET_BINDING]
[CONDITION_CHECK]
[USE_DECISION]
[REVISED_HINT]
"""


def geometry3k_prompt(row: dict[str, Any], gdp: str, route: str) -> str:
    return f"""You are a fine-grained route verifier and target-binding checker for geometry MLLM solving.
Given a geometry multiple-choice problem, GDP-4B automatic diagram parse, and a candidate theorem route, decide whether the route is target-matched, condition-supported, and safe to show to a downstream multimodal solver.

Do not solve the problem numerically. Do not choose the final answer.

Question:
{row.get("question", "")}

Choices:
{choices_text(row)}

GDP-4B automatic parse:
{clip(gdp, 2000)}

Candidate theorem route:
{clip(route, 1600)}

Output exactly these sections:
[ROUTE_VALIDITY]
[TARGET_BINDING]
[CONDITION_CHECK]
[USE_DECISION]
[REVISED_HINT]
"""


def verifier_report(
    validity: str,
    target_binding: str,
    condition_check: str,
    decision: str,
    revised_hint: str,
) -> str:
    assert decision in VALID_DECISIONS, decision
    return f"""[ROUTE_VALIDITY]
{validity}

[TARGET_BINDING]
{target_binding}

[CONDITION_CHECK]
{condition_check}

[USE_DECISION]
{decision}

[REVISED_HINT]
{revised_hint}
"""


def formalgeo_valid_target(row: dict[str, Any], route: str) -> str:
    names = theorem_names(route)
    target = infer_target(row.get("question", ""), row.get("goal_cdl", ""))
    short_route = "; ".join(names[:5]) if names else "the candidate theorem route"
    return verifier_report(
        "valid",
        f"The problem target is {target}. The candidate route is derived from the reference theorem sequence and is aligned with the goal CDL `{row.get('goal_cdl', '')}`.",
        "The route is supported by the provided FormalGeo CDL facts. The downstream solver should still verify visible relations in the diagram.",
        "use_route",
        f"Use this theorem-route hypothesis after checking the image: {short_route}. Bind the route to the requested target `{row.get('goal_cdl', '')}` and do not treat the route as a final answer.",
    )


def formalgeo_wrong_target(row: dict[str, Any], route: str, wrong_source: dict[str, Any]) -> str:
    target = infer_target(row.get("question", ""), row.get("goal_cdl", ""))
    wrong_target = infer_target(wrong_source.get("question", ""), wrong_source.get("goal_cdl", ""))
    names = theorem_names(route)
    short_route = "; ".join(names[:5]) if names else "the candidate route"
    return verifier_report(
        "invalid",
        f"The current target is {target} with goal `{row.get('goal_cdl', '')}`, but the candidate route comes from a different problem whose target is {wrong_target}. The theorem names do not establish how to compute the requested quantity.",
        "Condition support is not established for this problem. A route transferred from another problem may mention unsupported points, lines, or theorem conditions.",
        "reject_route",
        f"Do not show the candidate route `{short_route}`. Solve image-first and use only relations directly supported by the current problem.",
    )


def formalgeo_partial_target(row: dict[str, Any], route: str) -> str:
    names = theorem_names(route)
    first = names[0] if names else "the theorem family"
    target = infer_target(row.get("question", ""), row.get("goal_cdl", ""))
    return verifier_report(
        "partially_valid",
        f"The theorem family may be relevant to the target {target}, but the candidate hint is too coarse: it names `{first}` without explicitly binding the route to `{row.get('goal_cdl', '')}`.",
        "Some theorem conditions are likely supported by the CDL facts, but the downstream solver must check the exact variables and corresponding objects before using the route.",
        "rewrite_route",
        f"Use `{first}` only as a low-confidence theorem family. First identify the target `{row.get('goal_cdl', '')}`, then bind each theorem variable to visible/stated objects before computing.",
    )


def ok(item: dict[str, Any] | None) -> bool:
    return bool(item and item.get("complete_response") and item.get("correct"))


def geometry_behavior_target(row: dict[str, Any], gdp: str, route: str, pair: dict[str, dict[str, Any]]) -> str:
    raw_ok = ok(pair.get("image_only"))
    route_ok = ok(pair.get("image_route"))
    target = infer_target(row.get("question", ""))
    names = theorem_names(route)
    short = "; ".join(names[:5]) if names else "No theorem route selected"
    risky_fams = {"area", "sine", "arc", "circle", "similar", "trapezoid", "parallelogram"}
    is_risky = bool(route_family(route) & risky_fams)

    if route_ok and not raw_ok:
        return verifier_report(
            "valid",
            f"The route appears helpful for the requested target `{target}` because the route-guided solver fixed the raw image-only error on the calibration split.",
            "The route should still be verified against the image. Prefer theorem names and target binding over copying the raw formal sequence.",
            "use_route",
            f"Use this route as a verifiable hint: {short}. First bind it to the requested target `{target}`, then solve from the image.",
        )
    if raw_ok and not route_ok:
        reason = "The candidate route caused a raw-correct example to become wrong on the calibration split."
        if is_risky:
            reason += " It uses a high-risk family such as area, sine, arc/circle, similarity, or quadrilateral formula, where variable and target binding are easy to mismatch."
        return verifier_report(
            "partially_valid" if names else "invalid",
            f"The target is `{target}`, but the route is unsafe: {reason}",
            "The theorem name alone is not enough. Required objects, givens, corresponding vertices, or formula variables are not explicitly bound to the problem target.",
            "rewrite_route" if names else "reject_route",
            f"Do not expose the raw route directly. If using it, rewrite it with explicit target binding for `{target}` and verify every required given in the image; otherwise solve image-first.",
        )
    if raw_ok and route_ok:
        return verifier_report(
            "valid",
            f"The route is not harmful on the calibration split and is compatible with the target `{target}`.",
            "Use it as a weak hint only; still verify the route conditions visually.",
            "use_route_cautious",
            f"Optional theorem-route hint: {short}. Use it only if it directly computes the requested target `{target}`.",
        )
    return verifier_report(
        "invalid" if is_risky else "partially_valid",
        f"Neither raw nor route-guided solving succeeded on this calibration example. The route does not provide enough target-binding information for `{target}`.",
        "The route may be irrelevant, under-specified, or missing necessary visual conditions.",
        "use_image_first",
        "No route should be trusted. Solve from the image and problem text first; use generated structure only as low-confidence background.",
    )


def build_formalgeo_rows(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    selected = rows[:n]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        fam = next(iter(route_family(r.get("target", ""))), "other")
        by_family[fam].append(r)
    out: list[dict[str, Any]] = []
    for i, row in enumerate(selected):
        route = row.get("target", "")
        out.append(
            {
                "id": f"fg_{row.get('id')}_valid",
                "source": "formalgeo_valid",
                "prompt": formalgeo_prompt(row, route),
                "target": formalgeo_valid_target(row, route),
                "label": "valid",
            }
        )
        fams = list(route_family(route))
        hard_pool = []
        for fam in fams:
            hard_pool.extend(x for x in by_family.get(fam, []) if x.get("id") != row.get("id"))
        wrong = rng.choice(hard_pool or [x for x in rows if x.get("id") != row.get("id")])
        wrong_route = wrong.get("target", "")
        out.append(
            {
                "id": f"fg_{row.get('id')}_wrong",
                "source": "formalgeo_wrong_route",
                "prompt": formalgeo_prompt(row, wrong_route),
                "target": formalgeo_wrong_target(row, wrong_route, wrong),
                "label": "invalid",
            }
        )
        if i % 2 == 0:
            names = theorem_names(route)
            partial_route = "[THEOREM ROUTE]\n1. Use " + (names[0] if names else "the likely theorem family") + "."
            out.append(
                {
                    "id": f"fg_{row.get('id')}_partial",
                    "source": "formalgeo_partial",
                    "prompt": formalgeo_prompt(row, partial_route),
                    "target": formalgeo_partial_target(row, partial_route),
                    "label": "partial",
                }
            )
    rng.shuffle(out)
    return out


def build_geometry_rows(
    dataset: list[dict[str, Any]],
    gdp_by_id: dict[str, str],
    route_by_id: dict[str, str],
    result_by_id: dict[str, dict[str, dict[str, Any]]],
    heldout_ids: set[str],
) -> list[dict[str, Any]]:
    out = []
    for row in dataset:
        rid = str(row["id"])
        if rid in heldout_ids:
            continue
        if rid not in gdp_by_id or rid not in route_by_id or rid not in result_by_id:
            continue
        target = geometry_behavior_target(row, gdp_by_id[rid], route_by_id[rid], result_by_id[rid])
        out.append(
            {
                "id": f"g3k_{rid}",
                "source": "geometry3k_behavior_calibration",
                "prompt": geometry3k_prompt(row, gdp_by_id[rid], route_by_id[rid]),
                "target": target,
                "label": "behavior",
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formalgeo-train", type=Path, default=Path("formalgeo_route_data_all/train.jsonl"))
    ap.add_argument("--formalgeo-val", type=Path, default=Path("formalgeo_route_data_all/val.jsonl"))
    ap.add_argument("--geometry-dataset", type=Path, default=Path("multidataset_2x300_eval.jsonl"))
    ap.add_argument("--gdp", type=Path, default=Path("gdp4b_geometry300_parse.jsonl"))
    ap.add_argument("--route-cache", type=Path, required=True)
    ap.add_argument("--geometry-results", type=Path, required=True)
    ap.add_argument("--heldout-test-ids", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("data/route_verifier_text"))
    ap.add_argument("--formalgeo-train-n", type=int, default=700)
    ap.add_argument("--formalgeo-val-n", type=int, default=120)
    ap.add_argument("--geometry-val-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fg_train = build_formalgeo_rows(read_jsonl(args.formalgeo_train), args.formalgeo_train_n, args.seed)
    fg_val = build_formalgeo_rows(read_jsonl(args.formalgeo_val), args.formalgeo_val_n, args.seed + 1)

    geometry_rows = [r for r in read_jsonl(args.geometry_dataset) if r.get("dataset") == "Geometry3K-300"]
    gdp_by_id = {str(r["id"]): (r.get("gdp_text") or r.get("gdp_response") or "") for r in read_jsonl(args.gdp) if not r.get("error")}
    route_by_id = {str(r["id"]): r.get("generated_route", "") for r in read_jsonl(args.route_cache)}
    heldout_ids = set(json.loads(args.heldout_test_ids.read_text(encoding="utf-8")))
    result_by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in read_jsonl(args.geometry_results):
        result_by_id[str(r["id"])][str(r["variant"])] = r
    g3k = build_geometry_rows(geometry_rows, gdp_by_id, route_by_id, result_by_id, heldout_ids)
    rng.shuffle(g3k)
    g3k_val = g3k[: args.geometry_val_n]
    g3k_train = g3k[args.geometry_val_n :]

    train = fg_train + g3k_train
    val = fg_val + g3k_val
    rng.shuffle(train)
    rng.shuffle(val)

    write_jsonl(args.outdir / "train.jsonl", train)
    write_jsonl(args.outdir / "val.jsonl", val)
    (args.outdir / "heldout_test_ids.json").write_text(json.dumps(sorted(heldout_ids), indent=2), encoding="utf-8")
    meta = {
        "train": len(train),
        "val": len(val),
        "formalgeo_train_rows_input": args.formalgeo_train_n,
        "formalgeo_val_rows_input": args.formalgeo_val_n,
        "geometry3k_calibration_total": len(g3k),
        "geometry3k_train": len(g3k_train),
        "geometry3k_val": len(g3k_val),
        "heldout_test": len(heldout_ids),
        "train_sources": dict(Counter(r["source"] for r in train)),
        "val_sources": dict(Counter(r["source"] for r in val)),
        "train_labels": dict(Counter(r["label"] for r in train)),
        "val_labels": dict(Counter(r["label"] for r in val)),
    }
    (args.outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
