#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REQUIRED_SECTIONS = ["[ROUTE]", "[RELEVANT_FACTS]", "[TRUST_SCORE]", "[DECISION]", "[MLLM_PROMPT]"]


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


def route_body(target: str) -> str:
    return (target or "").strip()


def student_prompt(question: str, parse_text: str, route: str) -> str:
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


def fallback_use_route(route: str) -> str:
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


def fallback_reject_route() -> str:
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


def normalize_target(text: str, fallback: str) -> str:
    text = (text or "").strip()
    match = re.search(r"(?m)^\[ROUTE\]\s*$", text)
    if match:
        text = text[match.start() :].strip()
        if all(section in text for section in REQUIRED_SECTIONS):
            return text
    return fallback


def teacher_user_prompt(question: str, parse_text: str, reference_route: str, candidate_route: str, candidate_kind: str) -> str:
    return f"""You are labeling data for a small trust-calibrated route policy model.

The downstream task is geometry multiple-choice solving by a multimodal LLM. The small policy model will not answer the problem. It will only decide whether a generated theorem route and structured diagram facts should be shown to the multimodal solver.

Use the reference theorem sequence only as supervision for this labeling task. The student model will not see the reference sequence at inference time.

Problem:
{question}

Structured diagram facts:
{clip(parse_text, 2200)}

Reference theorem route from the dataset:
{clip(reference_route, 1800)}

Candidate theorem route to evaluate:
{clip(candidate_route, 1800)}

Candidate construction note:
{candidate_kind}

Return exactly these sections:
[ROUTE]
Either keep a compact route, revise it into a safer partial route, or say "No theorem route selected."

[RELEVANT_FACTS]
List only facts/theorems that are relevant and visually/verifiably supported. Mention likely missing or risky relations if important.

[TRUST_SCORE]
Use this style: GDP/CDL facts: low|medium|high; Route: low|medium|high; Image: required

[DECISION]
Choose one: use_route_cautious, use_partial_route_cautious, use_image_first, reject_route

[MLLM_PROMPT]
Write the exact concise auxiliary prompt that should be given to the downstream multimodal solver. Do not include final answer or numerical solution.
"""


def openai_chat(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: int,
    retries: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            message = obj["choices"][0]["message"]
            content = (message.get("content") or "").strip()
            return content
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"teacher request failed after {retries + 1} attempts: {last_err}")


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        out[str(row["id"])] = row
    return out


def append_cache(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_candidates(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    rows = rows[:limit]
    routes = [route_body(r.get("target", "")) for r in rows]
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        question = row.get("question") or ""
        prompt_text = row.get("prompt") or ""
        parse_text = (
            row.get("gdp_text")
            or row.get("structured_parse")
            or section_after(prompt_text, "GDP-4B automatic diagram parse:")
            or section_after(prompt_text, "FormalGeo CDL structured parse:")
        )
        good_route = route_body(row.get("target", ""))
        if not good_route:
            continue
        rid = str(row.get("id", i))
        out.append(
            {
                "id": f"fg_{rid}_pos",
                "question": question,
                "parse_text": parse_text,
                "reference_route": good_route,
                "candidate_route": good_route,
                "candidate_kind": "aligned_gold_route",
                "fallback": fallback_use_route(good_route),
                "decision_hint": "use_route_cautious",
            }
        )
        if len(routes) > 1:
            bad_route = routes[(i + rng.randint(1, len(routes) - 1)) % len(routes)]
            if bad_route and bad_route != good_route:
                out.append(
                    {
                        "id": f"fg_{rid}_neg",
                        "question": question,
                        "parse_text": parse_text,
                        "reference_route": good_route,
                        "candidate_route": bad_route,
                        "candidate_kind": "mismatched_route_from_another_problem",
                        "fallback": fallback_reject_route(),
                        "decision_hint": "reject_route",
                    }
                )
    rng.shuffle(out)
    return out


def convert(
    source_rows: list[dict[str, Any]],
    limit: int,
    seed: int,
    cache_path: Path,
    use_teacher: bool,
    fallback_on_error: bool,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    retries: int,
) -> list[dict[str, Any]]:
    cache = load_cache(cache_path)
    out: list[dict[str, Any]] = []
    candidates = make_candidates(source_rows, limit, seed)
    for idx, cand in enumerate(candidates, 1):
        cid = cand["id"]
        cached = cache.get(cid)
        if cached and cached.get("target"):
            target = cached["target"]
            teacher_raw = cached.get("teacher_raw", "")
            teacher_error = cached.get("teacher_error", "")
        elif use_teacher:
            messages = [
                {
                    "role": "system",
                    "content": "You are a careful geometry route-policy labeling teacher. Return only the requested labeled sections.",
                },
                {
                    "role": "user",
                    "content": teacher_user_prompt(
                        cand["question"],
                        cand["parse_text"],
                        cand["reference_route"],
                        cand["candidate_route"],
                        cand["candidate_kind"],
                    ),
                },
            ]
            teacher_error = ""
            try:
                teacher_raw = openai_chat(base_url, api_key, model, messages, max_tokens, temperature, timeout, retries)
                target = normalize_target(teacher_raw, cand["fallback"])
            except Exception as exc:
                if not fallback_on_error:
                    raise
                teacher_raw = ""
                teacher_error = str(exc)
                target = cand["fallback"]
            cache_row = {
                "id": cid,
                "target": target,
                "teacher_raw": teacher_raw,
                "teacher_error": teacher_error,
                "candidate_kind": cand["candidate_kind"],
            }
            append_cache(cache_path, cache_row)
        else:
            target = cand["fallback"]
            teacher_raw = ""
            teacher_error = "teacher_disabled"

        out.append(
            {
                "id": cid,
                "prompt": student_prompt(cand["question"], cand["parse_text"], cand["candidate_route"]),
                "target": target,
                "source": "formalgeo_mimo_teacher" if use_teacher else "formalgeo_rule_fallback",
                "candidate_kind": cand["candidate_kind"],
                "decision_hint": cand["decision_hint"],
                "teacher_error": teacher_error,
            }
        )
        if idx % 20 == 0:
            print(f"converted {idx}/{len(candidates)} candidates", flush=True)
    random.Random(seed + 99).shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("data/tcrp_formalgeo_mimo_pretrain"))
    ap.add_argument("--cache", type=Path, default=Path("results/tcrp_mimo_teacher_cache.jsonl"))
    ap.add_argument("--train-limit", type=int, default=300)
    ap.add_argument("--val-limit", type=int, default=40)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--base-url", default=os.environ.get("MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1"))
    ap.add_argument("--model", default=os.environ.get("MIMO_MODEL", "mimo-v2.5-pro"))
    ap.add_argument("--api-key-env", default="MIMO_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--fallback-on-error", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    use_teacher = bool(api_key)
    if not use_teacher:
        print(f"{args.api_key_env} is not set; writing rule-fallback pretraining data.", flush=True)

    train = convert(
        read_jsonl(args.train),
        args.train_limit,
        args.seed,
        args.cache,
        use_teacher,
        args.fallback_on_error,
        args.base_url,
        api_key,
        args.model,
        args.max_tokens,
        args.temperature,
        args.timeout,
        args.retries,
    )
    val = convert(
        read_jsonl(args.val),
        args.val_limit,
        args.seed + 1,
        args.cache.with_name(args.cache.stem + "_val.jsonl"),
        use_teacher,
        args.fallback_on_error,
        args.base_url,
        api_key,
        args.model,
        args.max_tokens,
        args.temperature,
        args.timeout,
        args.retries,
    )
    write_jsonl(args.outdir / "train.jsonl", train)
    write_jsonl(args.outdir / "val.jsonl", val)
    meta = {
        "teacher": "mimo" if use_teacher else "rule_fallback",
        "model": args.model if use_teacher else "",
        "train": len(train),
        "val": len(val),
        "train_source_rows_limit": args.train_limit,
        "val_source_rows_limit": args.val_limit,
        "cache": str(args.cache),
        "fallback_on_error": args.fallback_on_error,
    }
    (args.outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
