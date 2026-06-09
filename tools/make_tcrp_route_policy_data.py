#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def make_prompt(row: dict[str, Any], gdp: str, route: str) -> str:
    return f"""You are a trust-calibrated route policy model for geometry MLLM solving.
Given a geometry question, answer choices, a GDP-4B automatic diagram parse, and a generated theorem route, decide what auxiliary text should be given to a downstream multimodal solver.

Your job:
- Do not solve the problem numerically.
- Do not choose the final answer.
- Decide whether to use image-only, GDP facts, theorem route, both, or reject generated text.
- Prefer short, verifiable prompts that the MLLM can check against the image.

Question:
{row.get("question", "")}

Choices:
{choices_text(row)}

GDP-4B automatic parse:
{clip(gdp, 1800)}

Generated theorem route:
{clip(route, 1800)}

Output exactly these sections:
[ROUTE]
[RELEVANT_FACTS]
[TRUST_SCORE]
[DECISION]
[MLLM_PROMPT]
"""


def ok(item: dict[str, Any] | None) -> bool:
    return bool(item and item.get("complete_response") and item.get("correct"))


def derive_policy(pair: dict[str, dict[str, Any]], gdp: str, route: str) -> dict[str, str]:
    raw = ok(pair.get("image_only"))
    gdp_ok = any(ok(pair.get(v)) for v in ["image_gdp", "image_gdp_cautious", "image_gdp_selfverify"])
    route_ok = any(ok(pair.get(v)) for v in ["image_route", "image_route_cautious", "image_route_selfverify"])
    combo_ok = any(ok(pair.get(v)) for v in ["image_gdp_route_selective", "image_gdp_route_conflict_aware", "image_gdp_route_cautious"])
    text_ok = any(ok(pair.get(v)) for v in ["gdp_text_only", "route_text_only", "gdp_route_text_only"])

    # Prefer policies that fix raw errors; otherwise avoid breaking raw-correct examples.
    if not raw and combo_ok:
        decision = "use_gdp_route_selective"
        trust = "GDP: medium; Route: medium-high; Image: required"
        reason = "Both structured facts and route may help, but only relevant and visually supported parts should be shown."
        prompt = (
            "Use the following generated theorem route and GDP facts only as hypotheses. "
            "First verify them against the image; ignore irrelevant or conflicting facts.\n\n"
            f"<THEOREM_ROUTE>\n{clip(route, 900)}\n</THEOREM_ROUTE>\n\n"
            f"<SELECTED_GDP_PARSE>\n{clip(gdp, 900)}\n</SELECTED_GDP_PARSE>"
        )
    elif not raw and route_ok:
        decision = "use_route_only"
        trust = "GDP: low-medium; Route: high; Image: required"
        reason = "The generated route is more useful than the raw image baseline; avoid adding GDP facts unless necessary."
        prompt = (
            "Use this theorem route only as a hypothesis. Verify it against the image before solving.\n\n"
            f"<THEOREM_ROUTE>\n{clip(route, 1200)}\n</THEOREM_ROUTE>"
        )
    elif not raw and gdp_ok:
        decision = "use_gdp_only"
        trust = "GDP: high; Route: low; Image: required"
        reason = "GDP facts appear more useful than the route; show only a compact parse hint."
        prompt = (
            "Use these parsed geometry facts only as auxiliary evidence. Verify against the image.\n\n"
            f"<GDP_PARSE>\n{clip(gdp, 1200)}\n</GDP_PARSE>"
        )
    elif raw and not route_ok and not combo_ok:
        decision = "use_image_only"
        trust = "GDP: low; Route: low; Image: high"
        reason = "Generated text is likely distracting; keep the downstream solver close to visual evidence."
        prompt = "No additional structured context is recommended. Solve from the image, question, and choices."
    elif raw and route_ok and not gdp_ok:
        decision = "use_route_cautious"
        trust = "GDP: low; Route: medium; Image: high"
        reason = "Route is not harmful here, but GDP facts are not needed."
        prompt = (
            "Optional low-confidence theorem route. Use only if directly supported by the image.\n\n"
            f"<THEOREM_ROUTE>\n{clip(route, 1000)}\n</THEOREM_ROUTE>"
        )
    elif text_ok and not raw:
        decision = "use_structured_text_cautious"
        trust = "GDP/Route: medium; Image: required"
        reason = "Structured text contains useful information, but it should still be checked visually."
        prompt = (
            "The structured text may contain useful target information. Use cautiously and verify with the image.\n\n"
            f"<THEOREM_ROUTE>\n{clip(route, 700)}\n</THEOREM_ROUTE>\n\n"
            f"<GDP_PARSE>\n{clip(gdp, 700)}\n</GDP_PARSE>"
        )
    else:
        decision = "use_image_first"
        trust = "GDP: low-medium; Route: low-medium; Image: high"
        reason = "No auxiliary source is clearly reliable; image-first solving is safest."
        prompt = "Use the image and problem text as primary evidence. Treat any generated structure as optional and low-confidence."

    relevant = []
    if "route" in decision:
        relevant.append("The theorem route may be relevant if its geometric configuration is visible.")
    if "gdp" in decision:
        relevant.append("Use only GDP facts directly tied to the target quantity.")
    if not relevant:
        relevant.append("No generated structured fact is reliably selected.")

    return {
        "route": clip(route, 1200) if "route" in decision else "No theorem route selected.",
        "facts": "\n".join(f"- {x}" for x in relevant),
        "trust": trust,
        "decision": decision,
        "reason": reason,
        "mllm_prompt": prompt,
    }


def target_text(policy: dict[str, str]) -> str:
    return f"""[ROUTE]
{policy["route"]}

[RELEVANT_FACTS]
{policy["facts"]}

[TRUST_SCORE]
{policy["trust"]}

[DECISION]
{policy["decision"]}

[MLLM_PROMPT]
{policy["mllm_prompt"]}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("multidataset_2x300_eval.jsonl"))
    ap.add_argument("--gdp", type=Path, default=Path("gdp4b_geometry300_parse.jsonl"))
    ap.add_argument("--route-cache", type=Path, required=True)
    ap.add_argument("--trust-results", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("data/tcrp_route_policy"))
    ap.add_argument("--train-n", type=int, default=80)
    ap.add_argument("--val-n", type=int, default=20)
    ap.add_argument("--test-n", type=int, default=200)
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.dataset) if r.get("dataset") == "Geometry3K-300"]
    gdp_by_id = {str(r["id"]): (r.get("gdp_text") or r.get("gdp_response") or "") for r in read_jsonl(args.gdp) if not r.get("error")}
    route_by_id = {str(r["id"]): r.get("generated_route", "") for r in read_jsonl(args.route_cache)}

    results_by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in read_jsonl(args.trust_results):
        results_by_id[str(r["id"])][str(r["variant"])] = r

    usable = [r for r in rows if str(r["id"]) in gdp_by_id and str(r["id"]) in route_by_id and str(r["id"]) in results_by_id]
    if len(usable) < args.train_n + args.val_n + args.test_n:
        raise RuntimeError(f"Need {args.train_n + args.val_n + args.test_n} usable examples, got {len(usable)}")

    train_rows = usable[: args.train_n]
    val_rows = usable[args.train_n : args.train_n + args.val_n]
    test_rows = usable[args.train_n + args.val_n : args.train_n + args.val_n + args.test_n]

    def convert(split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for row in split_rows:
            rid = str(row["id"])
            policy = derive_policy(results_by_id[rid], gdp_by_id[rid], route_by_id[rid])
            out.append(
                {
                    "id": rid,
                    "prompt": make_prompt(row, gdp_by_id[rid], route_by_id[rid]),
                    "target": target_text(policy),
                    "decision": policy["decision"],
                    "answer": row.get("answer"),
                    "image": row.get("image"),
                }
            )
        return out

    train = convert(train_rows)
    val = convert(val_rows)
    write_jsonl(args.outdir / "train.jsonl", train)
    write_jsonl(args.outdir / "val.jsonl", val)
    (args.outdir / "test_ids.json").write_text(json.dumps([str(r["id"]) for r in test_rows], indent=2), encoding="utf-8")
    meta = {
        "train": len(train),
        "val": len(val),
        "test": len(test_rows),
        "decision_counts_train": dict(sorted({d: sum(1 for r in train if r["decision"] == d) for d in {r["decision"] for r in train}}.items())),
        "decision_counts_val": dict(sorted({d: sum(1 for r in val if r["decision"] == d) for d in {r["decision"] for r in val}}.items())),
    }
    (args.outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
