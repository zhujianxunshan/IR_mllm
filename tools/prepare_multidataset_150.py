#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

import pandas as pd


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_score(value):
    return hashlib.md5(str(value).encode()).hexdigest()


def write_jsonl(path, rows):
    Path(path).write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def clean_metadata(metadata):
    if not isinstance(metadata, dict):
        return {}
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in metadata.items()}


def mathvista_row(row, dataset_name):
    choices = list(row["choices"]) if row["choices"] is not None else []
    if not 2 <= len(choices) <= 5:
        return None
    answer_text = str(row["answer"]).strip()
    gold = ""
    for index, choice in enumerate(choices):
        if str(choice).strip() == answer_text:
            if index >= 5:
                return None
            gold = chr(65 + index)
            break
    if not gold:
        return None
    metadata = clean_metadata(row["metadata"])
    return {
        "id": f"{dataset_name}_{row['pid']}",
        "pid": str(row["pid"]),
        "dataset": dataset_name,
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


def main():
    rows = []

    geometry = read_jsonl("geometry3k_test_full.jsonl")
    geometry = sorted(geometry, key=lambda row: stable_score(row["id"]))[:300]
    for row in geometry:
        row = dict(row)
        row["dataset"] = "Geometry3K-300"
        rows.append(row)

    df = pd.read_parquet("datasets/mathvista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet")
    mc = df[df["question_type"].astype(str).eq("multi_choice")].copy()
    mc["_score"] = mc["pid"].map(stable_score)
    mc = mc.sort_values("_score")
    mathvista_count = 0
    for _, row in mc.iterrows():
        item = mathvista_row(row, "MathVista-MC-300")
        if item:
            rows.append(item)
            mathvista_count += 1
            if mathvista_count >= 300:
                break

    counts = {}
    for row in rows:
        counts[row["dataset"]] = counts.get(row["dataset"], 0) + 1
    out = "multidataset_2x300_eval.jsonl"
    write_jsonl(out, rows)
    print("wrote", out, counts, "total", len(rows))
    for dataset in sorted(counts):
        first = next(row for row in rows if row["dataset"] == dataset)
        print(dataset, json.dumps({k: first.get(k) for k in ["id", "question", "answer", "image", "source", "context"]}, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
