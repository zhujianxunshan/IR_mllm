#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


DEFAULT_WEIGHTS = {
    "route_win": 12,
    "route_loss": 8,
    "hard_negative": 2,
    "both_correct": 1,
    "both_wrong": 1,
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def base_id(row_id: str) -> str:
    return re.sub(r"#rep\d+$", "", row_id)


def strengthen_target(row: dict) -> dict:
    row = dict(row)
    label = row.get("label")
    target = row.get("target", "")
    if label == "route_win":
        target = target.replace("[ROUTE_HELPFULNESS_CONFIDENCE]\n0.90", "[ROUTE_HELPFULNESS_CONFIDENCE]\n0.95")
        target = target.replace(
            "Use as a weak target-bound route:",
            "Use this route because it counterfactually fixes the raw image-only failure. Still verify conditions:",
        )
    elif label == "route_loss":
        target = target.replace("[ROUTE_HELPFULNESS_CONFIDENCE]\n0.05", "[ROUTE_HELPFULNESS_CONFIDENCE]\n0.02")
    row["target"] = target
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, default=Path("data/target_binding_verifier_601"))
    ap.add_argument("--outdir", type=Path, default=Path("data/target_binding_verifier_v2_balanced"))
    ap.add_argument("--seed", type=int, default=20260611)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    source_train = read_jsonl(args.source_dir / "train.jsonl")
    source_val = read_jsonl(args.source_dir / "val.jsonl")

    dedup: dict[str, dict] = {}
    for row in source_train:
        dedup.setdefault(base_id(str(row["id"])), row)
    base_rows = [strengthen_target(row) for row in dedup.values()]

    train: list[dict] = []
    for row in base_rows:
        weight = DEFAULT_WEIGHTS.get(row.get("label"), 1)
        for i in range(weight):
            item = dict(row)
            item["id"] = f"{base_id(str(row['id']))}#v2rep{i}" if i else base_id(str(row["id"]))
            train.append(item)
    rng.shuffle(train)

    val = [strengthen_target(row) for row in source_val]
    write_jsonl(args.outdir / "train.jsonl", train)
    write_jsonl(args.outdir / "val.jsonl", val)

    for name in ["test_ids.json", "calibration_ids.json"]:
        src = args.source_dir / name
        if src.exists():
            (args.outdir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    meta = {
        "source_dir": str(args.source_dir),
        "train": len(train),
        "train_base": len(base_rows),
        "val": len(val),
        "weights": DEFAULT_WEIGHTS,
        "labels_train": dict(Counter(row["label"] for row in train)),
        "labels_base": dict(Counter(row["label"] for row in base_rows)),
        "labels_val": dict(Counter(row["label"] for row in val)),
        "purpose": "V2 balanced verifier: increase route_win/route_loss supervision to reduce over-conservative route rejection while retaining hard-negative safety.",
    }
    (args.outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
