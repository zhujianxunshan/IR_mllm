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


def used_ids(paths: list[Path]) -> set[str]:
    out: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        if path.suffix == ".json":
            out.update(map(str, json.loads(path.read_text(encoding="utf-8"))))
        else:
            for row in read_jsonl(path):
                if "id" in row:
                    out.add(str(row["id"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry3k-root", type=Path, default=Path("data/geometry3k/test"))
    ap.add_argument("--exclude-jsonl", type=Path, action="append", default=[])
    ap.add_argument("--exclude-json", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, default=Path("data/geometry3k_extra100_eval.jsonl"))
    ap.add_argument("--ids-out", type=Path, default=Path("data/geometry3k_extra100_ids.json"))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=73)
    args = ap.parse_args()

    exclude = used_ids(list(args.exclude_jsonl) + list(args.exclude_json))
    candidates = []
    for d in sorted(args.geometry3k_root.iterdir()):
        if not d.is_dir() or not (d / "data.json").exists():
            continue
        raw = json.loads((d / "data.json").read_text(encoding="utf-8"))
        rid = f"geometry3k_test_{raw.get('id', d.name)}"
        if rid in exclude or str(raw.get("id")) in exclude:
            continue
        choices = raw.get("choices") or raw.get("compact_choices") or []
        answer = str(raw.get("answer", "")).strip().upper()
        if not choices or answer not in "ABCDE"[: len(choices)]:
            continue
        img = d / "img_diagram.png"
        if not img.exists():
            continue
        candidates.append(
            {
                "id": rid,
                "dataset": "Geometry3K-extra",
                "source": "Geometry3K/InterGPS-extra-heldout",
                "image": str(img),
                "question": raw.get("problem_text") or raw.get("compact_text") or raw.get("annotat_text") or "",
                "choices": choices,
                "answer": answer,
                "problem_type_graph": raw.get("problem_type_graph", []),
                "problem_type_goal": raw.get("problem_type_goal", []),
            }
        )
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    rows = candidates[: args.limit]
    if len(rows) < args.limit:
        raise RuntimeError(f"Need {args.limit} rows, got {len(rows)} after excluding {len(exclude)} ids")
    write_jsonl(args.out, rows)
    args.ids_out.parent.mkdir(parents=True, exist_ok=True)
    args.ids_out.write_text(json.dumps([r["id"] for r in rows], indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "ids_out": str(args.ids_out), "rows": len(rows), "excluded": len(exclude)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
