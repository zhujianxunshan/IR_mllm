#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def esc(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def block(text: Any) -> str:
    safe = str(text or "").strip().replace("```", "'''")
    return "```text\n" + safe + "\n```"


def choices_text(choices: list[Any]) -> str:
    return "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices or []))


def route_family(route: str) -> str:
    low = route.lower()
    keys = [
        "area",
        "sine",
        "cosine",
        "tangent",
        "circle",
        "arc",
        "parallel",
        "similar",
        "midsegment",
        "trapezoid",
        "parallelogram",
        "pythagorean",
        "right triangle",
        "line addition",
        "perimeter",
        "angle",
    ]
    found = [key for key in keys if key in low]
    return ", ".join(found) if found else "other"


def collect_items(
    set_name: str,
    result_path: Path,
    route_path: Path,
    data_path: Path,
) -> list[dict[str, Any]]:
    rows = read_jsonl(result_path)
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("variant") in {"image_only", "image_route"}:
            by_id[str(row["id"])][row["variant"]] = row

    routes = {str(row["id"]): row.get("generated_route", "") for row in read_jsonl(route_path)}
    data = {str(row["id"]): row for row in read_jsonl(data_path)}

    items: list[dict[str, Any]] = []
    for qid, pair in by_id.items():
        if "image_only" not in pair or "image_route" not in pair:
            continue
        raw = pair["image_only"]
        route = pair["image_route"]
        if (not raw.get("correct")) and route.get("correct"):
            category = "route_win"
        elif raw.get("correct") and (not route.get("correct")):
            category = "route_loss"
        elif raw.get("correct") and route.get("correct"):
            category = "both_correct"
        else:
            category = "both_wrong"
        route_text = routes.get(qid, "")
        items.append(
            {
                "set": set_name,
                "id": qid,
                "category": category,
                "row": data.get(qid, {}),
                "raw": raw,
                "route": route,
                "route_text": route_text,
                "family": route_family(route_text),
            }
        )
    return items


def main() -> None:
    result_sets = [
        (
            "main200",
            Path("results/qwen3vl8b_route_verifier_downstream200_4var.jsonl"),
            Path("results/geometry3k_trust_selection_route_cache_300.jsonl"),
            Path("multidataset_2x300_eval.jsonl"),
        ),
        (
            "extra100",
            Path("results/qwen3vl8b_route_verifier_extra100_4var.jsonl"),
            Path("results/geometry3k_extra100_route_cache.jsonl"),
            Path("data/geometry3k_extra100_eval.jsonl"),
        ),
    ]
    all_items: list[dict[str, Any]] = []
    for args in result_sets:
        all_items.extend(collect_items(*args))

    counter = Counter((item["set"], item["category"]) for item in all_items)
    lines: list[str] = []
    lines.append("# Route vs Raw 详细对比报告\n")
    lines.append(
        "本报告自动整理 `image_route` 相比 `image_only/raw` 的结果。"
        "重点列出所有 route 修正 raw 错误的题目，以及所有 route 把 raw 正确带错的题目；"
        "附录包含 300 题的完整分类索引。\n"
    )
    lines.append("## 总览\n")
    lines.append("| 测试集 | route win: raw错/route对 | route loss: raw对/route错 | both correct | both wrong | 总题数 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for set_name in ["main200", "extra100"]:
        total = sum(counter[(set_name, cat)] for cat in ["route_win", "route_loss", "both_correct", "both_wrong"])
        lines.append(
            f"| {set_name} | {counter[(set_name, 'route_win')]} | {counter[(set_name, 'route_loss')]} | "
            f"{counter[(set_name, 'both_correct')]} | {counter[(set_name, 'both_wrong')]} | {total} |"
        )
    lines.append("")
    lines.append("## 解释口径\n")
    lines.append("- `route_win`: raw/image-only 答错，但加入 generated theorem route 后答对。")
    lines.append("- `route_loss`: raw/image-only 答对，但加入 generated theorem route 后答错，即 route 带偏。")
    lines.append("- `route输入` 是下游 MLLM 收到的 theorem-route hypothesis 内容。")
    lines.append("- `MLLM回答` 保留模型原始输出，通常是 `ANSWER: X` 或单个选项字母。\n")

    section_titles = {
        "main200": "主测试 main200: Geometry3K held-out 200",
        "extra100": "扩展测试 extra100: Geometry3K extra clean 100",
    }
    category_titles = {
        "route_win": "Route Win：raw 错，route 对",
        "route_loss": "Route Loss：raw 对，route 错",
    }
    for set_name in ["main200", "extra100"]:
        for category in ["route_win", "route_loss"]:
            subset = [item for item in all_items if item["set"] == set_name and item["category"] == category]
            lines.append(f"## {section_titles[set_name]} / {category_titles[category]}（{len(subset)}题）\n")
            if not subset:
                lines.append("无。\n")
                continue
            for index, item in enumerate(subset, 1):
                row = item["row"]
                raw = item["raw"]
                route = item["route"]
                lines.append(f"### {index}. {item['id']}\n")
                lines.append(f"- 分类：`{category}`")
                lines.append(f"- Route family：`{item['family']}`")
                lines.append(f"- 图片：`{row.get('image', '')}`")
                lines.append(f"- 正确答案：`{raw.get('answer')}`")
                lines.append(f"- Raw 预测：`{raw.get('prediction')}`，correct=`{raw.get('correct')}`")
                lines.append(f"- Route 预测：`{route.get('prediction')}`，correct=`{route.get('correct')}`")
                lines.append("\n**原题**\n")
                lines.append(block(row.get("question", "")))
                lines.append("\n**选项**\n")
                lines.append(block(choices_text(row.get("choices") or [])))
                lines.append("\n**Route 输入给 MLLM 的内容**\n")
                lines.append(block(item["route_text"]))
                lines.append("\n**Raw MLLM 回答**\n")
                lines.append(block(raw.get("raw_response", "")))
                lines.append("\n**Route MLLM 回答**\n")
                lines.append(block(route.get("raw_response", "")))
                lines.append("")

    lines.append("## 附录 A：全部 300 题分类索引\n")
    lines.append("| 测试集 | ID | 分类 | Gold | Raw pred | Route pred | Route family | 题干 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    order = {"route_loss": 0, "route_win": 1, "both_correct": 2, "both_wrong": 3}
    for item in sorted(all_items, key=lambda x: (x["set"], order[x["category"]], x["id"])):
        row = item["row"]
        raw = item["raw"]
        route = item["route"]
        question = str(row.get("question", "") or "")
        if len(question) > 160:
            question = question[:157] + "..."
        lines.append(
            f"| {item['set']} | `{item['id']}` | `{item['category']}` | `{raw.get('answer')}` | "
            f"`{raw.get('prediction')}` | `{route.get('prediction')}` | {esc(item['family'])} | {esc(question)} |"
        )

    lines.append("\n## 附录 B：按 Route Family 统计 win/loss\n")
    lines.append("| 测试集 | Route family | route_win | route_loss |")
    lines.append("|---|---|---:|---:|")
    for set_name in ["main200", "extra100"]:
        families = sorted(
            {
                item["family"]
                for item in all_items
                if item["set"] == set_name and item["category"] in {"route_win", "route_loss"}
            }
        )
        for family in families:
            wins = sum(
                1
                for item in all_items
                if item["set"] == set_name and item["family"] == family and item["category"] == "route_win"
            )
            losses = sum(
                1
                for item in all_items
                if item["set"] == set_name and item["family"] == family and item["category"] == "route_loss"
            )
            lines.append(f"| {set_name} | {esc(family)} | {wins} | {losses} |")

    out = Path("results/route_vs_raw_detailed_report.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)
    print(
        json.dumps(
            {
                "items": len(all_items),
                "route_wins": sum(1 for item in all_items if item["category"] == "route_win"),
                "route_losses": sum(1 for item in all_items if item["category"] == "route_loss"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
