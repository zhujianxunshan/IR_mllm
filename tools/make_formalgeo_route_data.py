#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_simple_answer(answer: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?|-?\d+/\d+", answer.strip()))


def theorem_name(seq: str) -> str:
    return seq.split("(", 1)[0].strip()


def theorem_to_english(seq: str) -> str:
    name = theorem_name(seq)
    for old in [
        "_property",
        "_judgment",
        "_determination",
        "_definition",
        "_algebraic",
        "_with_common_vertex",
    ]:
        name = name.replace(old, "")
    return name.replace("_", " ").strip()


def theorem_route_text(theorem_seqs: list[str], include_raw: bool = True) -> str:
    lines = [
        "The following theorem route is a hypothesis for solving the geometry problem.",
        "",
        "[THEOREM ROUTE]",
    ]
    for i, seq in enumerate(theorem_seqs, 1):
        lines.append(f"{i}. Use {theorem_to_english(seq)}.")
    if include_raw:
        lines.extend(["", "[RAW THEOREM SEQUENCE]"])
        for i, seq in enumerate(theorem_seqs, 1):
            lines.append(f"{i}. {seq}")
    lines.extend(
        [
            "",
            "[USE POLICY]",
            "- Verify each theorem against the diagram and problem text.",
            "- Do not output the final numeric answer.",
        ]
    )
    return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def structured_parse(problem: dict[str, Any]) -> str:
    sections = [
        ("[CONSTRUCTION_CDL]", problem.get("construction_cdl") or []),
        ("[TEXT_CDL]", problem.get("text_cdl") or []),
        ("[IMAGE_CDL]", problem.get("image_cdl") or []),
    ]
    out: list[str] = []
    for title, items in sections:
        out.append(title)
        if items:
            out.extend(f"- {x}" for x in items)
        else:
            out.append("- none")
    out.append("[GOAL_CDL]")
    out.append(str(problem.get("goal_cdl", "")) or "none")
    return "\n".join(out)


def prompt_for(problem: dict[str, Any], parse_text: str, parse_name: str) -> str:
    return f"""You are a theorem-route generator for geometry problems.
Given the problem text and an automatic diagram parse, predict a compact theorem route that may help a multimodal model solve the problem.

Important:
- Do not solve the problem numerically.
- Do not output the final answer.
- Output only [THEOREM ROUTE], optional [RAW THEOREM SEQUENCE], and [USE POLICY].

Question:
{problem.get("problem_text_en", "")}

{parse_name}:
{parse_text}
"""


def bucket(row: dict[str, Any]) -> str:
    k = len(row["theorem_seqs"])
    if k <= 1:
        return "len1"
    if k <= 3:
        return "len2_3"
    return "len4plus"


def stratified_take(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[bucket(row)].append(row)
    names = ["len1", "len2_3", "len4plus"]
    for name in names:
        rng.shuffle(groups[name])
    quota = {name: limit // len(names) for name in names}
    for name in names[: limit % len(names)]:
        quota[name] += 1
    selected: list[dict[str, Any]] = []
    for name in names:
        selected.extend(groups[name][: quota[name]])
    chosen = {row["id"] for row in selected}
    rest = [row for row in rows if row["id"] not in chosen]
    rng.shuffle(rest)
    selected.extend(rest[: max(0, limit - len(selected))])
    return sorted(selected, key=lambda x: int(x["id"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", default="formalgeo7k_v2")
    ap.add_argument("--datasets-path", default="datasets/formalgeo")
    ap.add_argument("--outdir", type=Path, default=Path("formalgeo_route_data"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--train-limit", type=int, default=5000)
    ap.add_argument("--val-limit", type=int, default=500)
    ap.add_argument("--test-limit", type=int, default=500)
    ap.add_argument("--gdp", type=Path, default=None)
    ap.add_argument("--require-gdp", action="store_true")
    args = ap.parse_args()

    from formalgeo.data import DatasetLoader

    loader = DatasetLoader(args.dataset_name, args.datasets_path)
    gdp_by_id: dict[str, str] = {}
    if args.gdp and args.gdp.exists():
        for item in read_jsonl(args.gdp):
            if item.get("error"):
                continue
            text = item.get("gdp_text") or item.get("gdp_response") or item.get("response") or ""
            if text:
                gdp_by_id[str(item["id"])] = str(text)
    root = Path(args.datasets_path) / args.dataset_name
    diagrams = root / "diagrams"
    n = int(loader.info.get("problem_number", 7000))
    rows: list[dict[str, Any]] = []
    for pid in range(1, n + 1):
        p = loader.get_problem(pid)
        theorem_seqs = [str(x) for x in p.get("theorem_seqs", []) if str(x).strip()]
        answer = str(p.get("problem_answer", "")).strip()
        image_path = diagrams / str(p.get("problem_img", ""))
        if not theorem_seqs or not p.get("problem_text_en") or not image_path.exists():
            continue
        gdp_text = gdp_by_id.get(str(pid), "")
        if args.require_gdp and not gdp_text:
            continue
        parse_text = gdp_text or structured_parse(p)
        parse_name = "GDP-4B automatic diagram parse" if gdp_text else "FormalGeo CDL structured parse"
        row = {
            "id": str(pid),
            "source": p.get("source", ""),
            "question": p.get("problem_text_en", ""),
            "image": str(image_path),
            "answer": answer,
            "simple_answer": is_simple_answer(answer),
            "construction_cdl": p.get("construction_cdl") or [],
            "text_cdl": p.get("text_cdl") or [],
            "image_cdl": p.get("image_cdl") or [],
            "goal_cdl": p.get("goal_cdl", ""),
            "structured_parse": structured_parse(p),
            "gdp_text": gdp_text,
            "theorem_seqs": theorem_seqs,
            "prompt": prompt_for(p, parse_text, parse_name),
            "target": theorem_route_text(theorem_seqs),
        }
        rows.append(row)

    rows_by_id = {row["id"]: row for row in rows}
    existing_train = args.outdir / "train.jsonl"
    existing_val = args.outdir / "val.jsonl"
    existing_test = args.outdir / "test.jsonl"
    if args.require_gdp and existing_train.exists() and existing_val.exists() and existing_test.exists():
        def reuse(path: Path) -> list[dict[str, Any]]:
            out = []
            for old in read_jsonl(path):
                row = rows_by_id.get(str(old["id"]))
                if row is not None:
                    out.append(row)
            return out

        train = reuse(existing_train)
        val = reuse(existing_val)
        test = reuse(existing_test)
    else:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        val_pool = rows[: args.val_limit * 2]
        test_pool = rows[args.val_limit * 2 : args.val_limit * 2 + args.test_limit * 3]
        train_pool = rows[args.val_limit * 2 + args.test_limit * 3 :]
        train = stratified_take(train_pool, args.train_limit, args.seed)
        val = stratified_take(val_pool, args.val_limit, args.seed + 1)
        # Downstream numeric accuracy is only meaningful for simple numeric answers.
        test = stratified_take([r for r in test_pool if r["simple_answer"]], args.test_limit, args.seed + 2)

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.outdir / "train.jsonl", train)
    write_jsonl(args.outdir / "val.jsonl", val)
    write_jsonl(args.outdir / "test.jsonl", test)
    parse_rows = []
    for row in train + val + test:
        parse_rows.append(
            {
                "id": row["id"],
                "image": row["image"],
                "question": row["question"],
                "answer": row["answer"],
            }
        )
    write_jsonl(args.outdir / "gdp_parse_input.jsonl", parse_rows)
    meta = {
        "dataset_name": args.dataset_name,
        "total_rows": len(rows),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "gdp_rows": len(gdp_by_id),
        "require_gdp": args.require_gdp,
        "train_buckets": dict(defaultdict(int, {b: sum(1 for r in train if bucket(r) == b) for b in ["len1", "len2_3", "len4plus"]})),
        "val_buckets": dict(defaultdict(int, {b: sum(1 for r in val if bucket(r) == b) for b in ["len1", "len2_3", "len4plus"]})),
        "test_buckets": dict(defaultdict(int, {b: sum(1 for r in test if bucket(r) == b) for b in ["len1", "len2_3", "len4plus"]})),
    }
    (args.outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
