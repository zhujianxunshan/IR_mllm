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


def theorem_name(seq: str) -> str:
    return seq.split("(", 1)[0].strip()


def theorem_to_english(seq: str) -> str:
    name = theorem_name(seq)
    for old in ["_property", "_judgment", "_determination", "_definition", "_algebraic", "_with_common_vertex"]:
        name = name.replace(old, "")
    return name.replace("_", " ").strip()


def unique_theorems(seqs: list[str], max_steps: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for seq in seqs:
        text = theorem_to_english(str(seq))
        key = re.sub(r"\s+", " ", text.lower()).strip()
        if key and key not in seen:
            out.append(text)
            seen.add(key)
        if len(out) >= max_steps:
            break
    return out


def infer_target(row: dict[str, Any]) -> str:
    goal = str(row.get("goal_cdl") or row.get("question") or "").lower()
    if any(x in goal for x in ["angle", "measureof(angle", "m \\angle"]):
        return "angle"
    if any(x in goal for x in ["area", "shaded"]):
        return "area"
    if any(x in goal for x in ["arc", "measureof(arc"]):
        return "arc"
    if any(x in goal for x in ["perimeter"]):
        return "perimeter"
    if any(x in goal for x in ["length", "line", "segment", "find x"]):
        return "length_or_variable"
    return "unknown"


def parse_text(row: dict[str, Any]) -> str:
    gdp = row.get("gdp_text") or row.get("structured_parse") or ""
    if gdp:
        return str(gdp)
    parts = []
    for key in ["construction_cdl", "text_cdl", "image_cdl", "goal_cdl"]:
        val = row.get(key)
        if val:
            parts.append(f"[{key.upper()}]\n{val}")
    return "\n".join(parts) or "none"


def base_prompt(row: dict[str, Any], mode: str) -> str:
    return f"""You are a conservative theorem-route generator for geometry MLLM solving.
Given the problem text and a diagram parse, output a short route hint only when it is likely to help.

Rules:
- Never solve numerically and never choose the final answer.
- Never repeat the same theorem.
- Prefer 1-3 target-matched theorem hints.
- If the route is uncertain, mark it LOW_CONFIDENCE.
- If no theorem is clearly supported by the parse and target, output DISCARD.

Training mode: {mode}

Question:
{row.get("question", "")}

Target type:
{infer_target(row)}

Diagram parse:
{parse_text(row)}
"""


def compact_target(row: dict[str, Any], max_steps: int) -> str:
    steps = unique_theorems([str(x) for x in row.get("theorem_seqs", [])], max_steps=max_steps)
    lines = ["[ROUTE_QUALITY]", "TRUSTED" if len(steps) <= 3 else "LOW_CONFIDENCE", "", "[TARGET_BINDING]", f"Target type: {infer_target(row)}", ""]
    if not steps:
        lines += ["[THEOREM ROUTE]", "DISCARD: no supported theorem route."]
    else:
        lines.append("[THEOREM ROUTE]")
        for i, step in enumerate(steps, 1):
            lines.append(f"{i}. Use {step}.")
    lines += ["", "[USE_POLICY]", "Treat this as a weak hint; verify it against the image before using."]
    return "\n".join(lines)


def confidence_target(row: dict[str, Any], max_steps: int) -> str:
    seqs = [str(x) for x in row.get("theorem_seqs", []) if str(x).strip()]
    steps = unique_theorems(seqs, max_steps=max_steps)
    quality = "TRUSTED" if len(seqs) <= 3 and len(steps) <= 3 else "LOW_CONFIDENCE"
    lines = ["[ROUTE_QUALITY]", quality, "", "[TARGET_BINDING]", f"Target type: {infer_target(row)}"]
    if len(seqs) > max_steps:
        lines.append(f"Original proof uses {len(seqs)} theorem calls; only the most compact prefix is shown.")
    lines += ["", "[THEOREM ROUTE]"]
    for i, step in enumerate(steps, 1):
        lines.append(f"{i}. Use {step}.")
    lines += ["", "[USE_POLICY]", "If the theorem conditions or target binding are not visible, ignore this route and solve from the image."]
    return "\n".join(lines)


def discard_negative(row: dict[str, Any], wrong_theorem: str) -> dict[str, str]:
    prompt = base_prompt(row, "confidence_sft_negative") + f"\nDistractor theorem proposed by a weak parser:\nUse {wrong_theorem}.\n"
    target = "\n".join(
        [
            "[ROUTE_QUALITY]",
            "DISCARD",
            "",
            "[TARGET_BINDING]",
            f"Target type: {infer_target(row)}",
            "",
            "[THEOREM ROUTE]",
            "DISCARD: proposed theorem is not target-matched or condition-supported.",
            "",
            "[USE_POLICY]",
            "Do not show this route to the downstream MLLM.",
        ]
    )
    return {"id": f"{row.get('id')}_neg", "prompt": prompt, "target": target, "source_id": row.get("id")}


def convert(rows: list[dict[str, Any]], mode: str, max_steps: int, neg_ratio: float, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    theorem_pool = []
    for row in rows:
        theorem_pool.extend(unique_theorems([str(x) for x in row.get("theorem_seqs", [])], max_steps=10))
    theorem_pool = sorted(set(theorem_pool)) or ["parallel corresponding angle"]

    out: list[dict[str, Any]] = []
    for row in rows:
        if mode == "compact_sft":
            target = compact_target(row, max_steps=max_steps)
        elif mode == "confidence_sft":
            target = confidence_target(row, max_steps=max_steps)
        else:
            raise ValueError(mode)
        out.append({"id": row.get("id"), "prompt": base_prompt(row, mode), "target": target})
        if mode == "confidence_sft" and rng.random() < neg_ratio:
            true_steps = set(unique_theorems([str(x) for x in row.get("theorem_seqs", [])], max_steps=10))
            candidates = [x for x in theorem_pool if x not in true_steps] or theorem_pool
            out.append(discard_negative(row, rng.choice(candidates)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=Path, default=Path("formalgeo_route_data_all"))
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--mode", choices=["compact_sft", "confidence_sft"], required=True)
    ap.add_argument("--train-limit", type=int, default=2500)
    ap.add_argument("--val-limit", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--neg-ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()

    train = read_jsonl(args.src_dir / "train.jsonl")[: args.train_limit]
    val = read_jsonl(args.src_dir / "val.jsonl")[: args.val_limit]
    train_rows = convert(train, args.mode, args.max_steps, args.neg_ratio, args.seed)
    val_rows = convert(val, args.mode, args.max_steps, args.neg_ratio, args.seed + 1)
    write_jsonl(args.outdir / "train.jsonl", train_rows)
    write_jsonl(args.outdir / "val.jsonl", val_rows)
    meta = {
        "mode": args.mode,
        "src_dir": str(args.src_dir),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_source_rows": len(train),
        "val_source_rows": len(val),
        "max_steps": args.max_steps,
        "neg_ratio": args.neg_ratio if args.mode == "confidence_sft" else 0.0,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
