#!/usr/bin/env python3
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def stable_score(pid):
    return hashlib.md5(str(pid).encode()).hexdigest()


def meta_value(row, key):
    metadata = row["metadata"]
    return metadata.get(key) if isinstance(metadata, dict) else None


def jsonable_metadata(metadata):
    if not isinstance(metadata, dict):
        return {}
    return {
        key: value.tolist() if hasattr(value, "tolist") else value
        for key, value in metadata.items()
    }


def main():
    parquet_path = "datasets/mathvista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet"
    df = pd.read_parquet(parquet_path)
    mc = df[df["question_type"].astype(str).eq("multi_choice")].copy()

    selected = []
    source_series = mc.apply(lambda row: meta_value(row, "source"), axis=1)
    for _, group in mc.groupby(source_series):
        group = group.copy()
        group["_score"] = group["pid"].map(stable_score)
        group = group.sort_values("_score")
        selected.append(group.head(max(1, round(len(group) / 3))))

    sel = pd.concat(selected).copy()
    sel["_pid_int"] = sel["pid"].astype(str).astype(int)
    sel = sel.sort_values("_pid_int").reset_index(drop=True)

    rows = []
    for _, row in sel.iterrows():
        choices = list(row["choices"]) if row["choices"] is not None else []
        answer_text = str(row["answer"]).strip()
        gold = ""
        for index, choice in enumerate(choices):
            if str(choice).strip() == answer_text:
                gold = "ABCD"[index] if index < 4 else ""
                break
        if not gold:
            continue

        metadata = jsonable_metadata(row["metadata"])
        rows.append(
            {
                "id": f"mathvista_testmini_{row['pid']}",
                "pid": str(row["pid"]),
                "dataset": "MathVista-testmini-mc-1of3",
                "question": str(row["question"]),
                "choices": choices,
                "answer": gold,
                "answer_text": answer_text,
                "image": "datasets/mathvista/" + str(row["image"]),
                "source": metadata.get("source"),
                "task": metadata.get("task"),
                "context": metadata.get("context"),
                "category": metadata.get("category"),
                "grade": metadata.get("grade"),
                "metadata": metadata,
            }
        )

    out = Path("datasets/mathvista/mathvista_testmini_mc_1of3.jsonl")
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    print("wrote", out, "rows", len(rows), "from multi_choice", len(mc))
    print("source", Counter(row["source"] for row in rows).most_common())
    print("task", Counter(row["task"] for row in rows).most_common())
    print("context", Counter(row["context"] for row in rows).most_common())
    print("first")
    print(json.dumps(rows[0], ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    main()
