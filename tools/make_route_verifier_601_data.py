#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_DECISIONS = {"use_route", "use_route_cautious", "rewrite_route", "reject_route", "use_image_first"}


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
    return "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(row.get("choices") or []))


def extract_theorem_route(route: str) -> str:
    m = re.search(r"\[THEOREM ROUTE\]\s*(.*?)(?=\n\[[A-Z_ ]+\]|\Z)", route or "", flags=re.S)
    return m.group(1).strip() if m else (route or "").strip()


def theorem_names(route: str) -> list[str]:
    names = []
    for line in extract_theorem_route(route).splitlines():
        line = re.sub(r"^\s*\d+\.\s*", "", line.strip(" -\t"))
        line = re.sub(r"^Use\s+", "", line, flags=re.I).strip(". ")
        if line:
            names.append(line)
    return names


def route_family(route: str) -> set[str]:
    low = route.lower()
    keys = [
        "angle",
        "arc",
        "area",
        "circle",
        "cosine",
        "line addition",
        "parallel",
        "parallelogram",
        "pythagorean",
        "right triangle",
        "similar",
        "sine",
        "tangent",
        "triangle",
    ]
    return {k for k in keys if k in low}


def infer_target(question: str) -> str:
    text = (question or "").lower()
    if "area" in text:
        return "area"
    if "perimeter" in text:
        return "perimeter"
    if "arc" in text or "widehat" in text:
        return "arc measure"
    if "angle" in text or "∠" in text:
        return "angle measure"
    if "length" in text or "find x" in text or "value of x" in text:
        return "length or unknown variable"
    return "the requested quantity"


def prompt(row: dict[str, Any], gdp: str, route: str) -> str:
    return f"""You are a route verifier and target-binding checker for multimodal geometry solving.
Given a geometry problem, a GDP-4B automatic diagram parse, and a candidate theorem route, decide whether the route is safe to show to a downstream multimodal solver.

Do not solve the problem. Do not choose the final answer.

Question:
{row.get("question", "")}

Choices:
{choices_text(row)}

GDP-4B automatic parse:
{clip(gdp, 2200)}

Candidate theorem route:
{clip(route, 1800)}

Output exactly these sections:
[ROUTE_VALIDITY]
[TARGET_BINDING]
[CONDITION_CHECK]
[USE_DECISION]
[REVISED_HINT]
"""


def report(validity: str, target_binding: str, condition_check: str, decision: str, revised_hint: str) -> str:
    if decision not in VALID_DECISIONS:
        raise ValueError(decision)
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


def ok(item: dict[str, Any] | None) -> bool:
    return bool(item and item.get("complete_response") and item.get("correct"))


def make_target(row: dict[str, Any], gdp: str, route: str, pair: dict[str, dict[str, Any]]) -> tuple[str, str]:
    raw_ok = ok(pair.get("image_only"))
    route_ok = ok(pair.get("compact_route"))
    target = infer_target(row.get("question", ""))
    names = theorem_names(route)
    route_short = "; ".join(names[:3]) if names else "no reliable theorem route"
    qc = (pair.get("compact_route") or {}).get("route_qc") or {}
    n_steps = int(qc.get("n_steps") or len(names) or 0)
    discard = bool(qc.get("discard"))
    risky = bool(route_family(route) & {"area", "sine", "arc", "circle", "similar", "parallelogram"})

    if route_ok and not raw_ok:
        return (
            "route_win",
            report(
                "valid",
                f"The route is useful for the target `{target}` on the calibration behavior signal: it fixes a raw image-only error.",
                "The route should still be checked against visible relations and givens, but it is safe to expose as a weak theorem hint.",
                "use_route",
                f"Use this route as a hypothesis: {route_short}. Bind each theorem to the requested target `{target}` and verify the diagram before computing.",
            ),
        )
    if raw_ok and not route_ok:
        reason = "the route causes a raw-correct case to become wrong"
        if risky:
            reason += ", and it belongs to a high-risk theorem family where variable binding is easy to mismatch"
        return (
            "route_loss",
            report(
                "unsafe",
                f"The route is not safe for the target `{target}`: {reason}.",
                "The named theorem is not enough; required objects, corresponding vertices, or formula variables are not explicitly bound to the target.",
                "reject_route" if discard or n_steps == 0 else "use_image_first",
                "Do not show the route to the downstream solver. Use the raw image, question, and choices only unless the route can be re-derived from visible conditions.",
            ),
        )
    if raw_ok and route_ok:
        return (
            "both_correct",
            report(
                "valid",
                f"The route is compatible with the target `{target}`, but the raw image-only solver already succeeds.",
                "The route is optional. It should not override the image if any condition is ambiguous.",
                "use_route_cautious",
                f"Optional weak hint: {route_short}. Use it only if it directly binds to `{target}` and all conditions are visible.",
            ),
        )
    return (
        "both_wrong",
        report(
            "uncertain",
            f"The route does not provide enough reliable target-binding information for `{target}`.",
            "Neither raw nor route-guided behavior succeeds on the calibration signal, so the route should be treated as unreliable.",
            "use_image_first",
            "No theorem route is trusted. Solve from the image and problem statement first.",
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("geometry3k_test_full.jsonl"))
    ap.add_argument("--gdp", type=Path, default=Path("results/gdp4b_geometry3k_test601_parse.jsonl"))
    ap.add_argument("--route-cache", type=Path, default=Path("results/safe_route_compact_geometry3k_test601.jsonl"))
    ap.add_argument("--downstream-results", type=Path, default=Path("results/qwen3vl8b_safe_route_test601_raw_old_compact.jsonl"))
    ap.add_argument("--ids", type=Path, default=Path("data/tcrp_route_policy/geometry3k_test601_ids.json"))
    ap.add_argument("--outdir", type=Path, default=Path("data/route_verifier_601"))
    ap.add_argument("--calibration-n", type=int, default=401)
    ap.add_argument("--val-n", type=int, default=80)
    ap.add_argument("--route-loss-repeat", type=int, default=8)
    ap.add_argument("--route-win-repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=61)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    id_order = [str(x) for x in json.loads(args.ids.read_text(encoding="utf-8"))]
    rng.shuffle(id_order)
    calib_ids = set(id_order[: args.calibration_n])
    test_ids = set(id_order[args.calibration_n :])

    dataset = {str(r["id"]): r for r in read_jsonl(args.dataset)}
    gdp = {str(r["id"]): (r.get("gdp_text") or r.get("gdp_response") or "") for r in read_jsonl(args.gdp) if not r.get("error")}
    routes = {str(r["id"]): r.get("generated_route", "") for r in read_jsonl(args.route_cache)}
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in read_jsonl(args.downstream_results):
        by_id[str(r["id"])][str(r["variant"])] = r

    rows = []
    for rid in id_order:
        if rid not in calib_ids:
            continue
        if rid not in dataset or rid not in gdp or rid not in routes or rid not in by_id:
            continue
        label, target = make_target(dataset[rid], gdp[rid], routes[rid], by_id[rid])
        rows.append(
            {
                "id": rid,
                "source": "geometry3k_601_behavior_calibration",
                "label": label,
                "prompt": prompt(dataset[rid], gdp[rid], routes[rid]),
                "target": target,
            }
        )

    rng.shuffle(rows)
    val = rows[: args.val_n]
    train_base = rows[args.val_n :]
    train = []
    for row in train_base:
        repeat = 1
        if row["label"] == "route_loss":
            repeat = max(1, args.route_loss_repeat)
        elif row["label"] == "route_win":
            repeat = max(1, args.route_win_repeat)
        for k in range(repeat):
            item = dict(row)
            if repeat > 1:
                item["id"] = f"{row['id']}#aug{k}"
            train.append(item)

    write_jsonl(args.outdir / "train.jsonl", train)
    write_jsonl(args.outdir / "val.jsonl", val)
    (args.outdir / "calibration_ids.json").write_text(json.dumps(sorted(calib_ids), indent=2), encoding="utf-8")
    (args.outdir / "test_ids.json").write_text(json.dumps(sorted(test_ids), indent=2), encoding="utf-8")
    meta = {
        "train": len(train),
        "train_base": len(train_base),
        "val": len(val),
        "calibration_ids": len(calib_ids),
        "test_ids": len(test_ids),
        "labels_train": dict(Counter(r["label"] for r in train)),
        "labels_train_base": dict(Counter(r["label"] for r in train_base)),
        "labels_val": dict(Counter(r["label"] for r in val)),
        "route_loss_repeat": args.route_loss_repeat,
        "route_win_repeat": args.route_win_repeat,
        "note": "Verifier is trained on behavior labels from calibration ids only. Downstream evaluation must use test_ids.json to avoid leakage.",
    }
    (args.outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
