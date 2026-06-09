#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
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


def section_after(prompt: str, marker: str) -> str:
    idx = prompt.find(marker)
    return prompt[idx + len(marker) :].strip() if idx >= 0 else ""


def make_prompt(question: str, parse_text: str, route: str) -> str:
    return f"""You are a trust-calibrated route policy model for geometry MLLM solving.
Given a geometry question, structured diagram facts, and a candidate theorem route, decide what auxiliary text should be given to a downstream multimodal solver.

Your job:
- Do not solve the problem numerically.
- Do not choose the final answer.
- Decide whether the candidate route should be used, used cautiously, or rejected.
- Prefer short, verifiable prompts that the MLLM can check against the image.

Question:
{question}

Structured diagram facts:
{clip(parse_text, 1800)}

Candidate theorem route:
{clip(route, 1800)}

Output exactly these sections:
[ROUTE]
[RELEVANT_FACTS]
[TRUST_SCORE]
[DECISION]
[MLLM_PROMPT]
"""


def target_use_route(route: str) -> str:
    return f"""[ROUTE]
{clip(route, 1300)}

[RELEVANT_FACTS]
- The candidate route is aligned with the symbolic theorem sequence.
- Each theorem should still be verified against the image and problem text.

[TRUST_SCORE]
GDP/CDL facts: medium-high; Route: high; Image: required

[DECISION]
use_route_cautious

[MLLM_PROMPT]
Use the following theorem route as a hypothesis. Verify each theorem against the image before solving.

<THEOREM_ROUTE>
{clip(route, 1200)}
</THEOREM_ROUTE>
"""


def target_reject_route() -> str:
    return """[ROUTE]
No theorem route selected.

[RELEVANT_FACTS]
- The candidate route is not aligned with the structured problem facts.
- Do not expose unsupported theorem hints to the downstream MLLM.

[TRUST_SCORE]
GDP/CDL facts: medium; Route: low; Image: high

[DECISION]
use_image_first

[MLLM_PROMPT]
No additional theorem route is recommended. Solve from the image, question, and choices; treat generated structure as low-confidence.
"""


def route_body(target: str) -> str:
    return target.strip()


def convert(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    rows = rows[:limit]
    targets = [route_body(r.get("target", "")) for r in rows]
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        question = row.get("question") or ""
        prompt_text = row.get("prompt") or ""
        parse_text = section_after(prompt_text, "GDP-4B automatic diagram parse:") or section_after(prompt_text, "FormalGeo CDL structured parse:")
        good_route = route_body(row.get("target", ""))
        if not good_route:
            continue
        rid = str(row.get("id", i))
        out.append(
            {
                "id": f"fg_{rid}_pos",
                "prompt": make_prompt(question, parse_text, good_route),
                "target": target_use_route(good_route),
                "source": "formalgeo_positive",
                "decision": "use_route_cautious",
            }
        )
        # Mismatched route negative: use another problem's route as a plausible but unsupported hint.
        if len(targets) > 1:
            bad_route = targets[(i + rng.randint(1, len(targets) - 1)) % len(targets)]
            if bad_route and bad_route != good_route:
                out.append(
                    {
                        "id": f"fg_{rid}_neg",
                        "prompt": make_prompt(question, parse_text, bad_route),
                        "target": target_reject_route(),
                        "source": "formalgeo_mismatched_route_negative",
                        "decision": "use_image_first",
                    }
                )
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("data/tcrp_formalgeo_pretrain"))
    ap.add_argument("--train-limit", type=int, default=1200)
    ap.add_argument("--val-limit", type=int, default=120)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    train = convert(read_jsonl(args.train), args.train_limit, args.seed)
    val = convert(read_jsonl(args.val), args.val_limit, args.seed + 1)
    write_jsonl(args.outdir / "train.jsonl", train)
    write_jsonl(args.outdir / "val.jsonl", val)
    meta = {
        "train": len(train),
        "val": len(val),
        "train_limit_source_rows": args.train_limit,
        "val_limit_source_rows": args.val_limit,
        "positive_negative": "each source row yields one aligned positive and one mismatched-route negative when possible",
    }
    (args.outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
