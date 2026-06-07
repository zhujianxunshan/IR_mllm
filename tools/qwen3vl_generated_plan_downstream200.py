#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


VARIANTS = ["raw", "image_gdp", "generated_construction_plan", "generated_plan_with_verifier", "oracle_construction_plan"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def normalize_answer(text: str, valid_letters: str = "ABCDE") -> str:
    letter_class = re.escape(valid_letters)
    matches = re.findall(rf"ANSWER\s*[:：]\s*([{letter_class}])\b", text or "", flags=re.I)
    if matches:
        return matches[-1].upper()
    matches = re.findall(rf"(?:option|choice|answer)\s*(?:is|:|：)?\s*([{letter_class}])\b", text or "", flags=re.I)
    if matches:
        return matches[-1].upper()
    stripped = (text or "").strip()
    if re.fullmatch(rf"[{letter_class}]", stripped, flags=re.I):
        return stripped.upper()
    prefix = re.match(rf"^\s*([{letter_class}])\s*[.)．。:：]", stripped, flags=re.I)
    return prefix.group(1).upper() if prefix else ""


def formal(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("formal_logic") or {}


def symbols(text: str) -> set[str]:
    return set(re.findall(r"\b[A-Z]\b|[a-z]\b", text or ""))


def all_predicates(row: dict[str, Any]) -> list[str]:
    preds: list[str] = []
    for key in ("dissolved_text_logic_form", "text_logic_form", "diagram_logic_form"):
        value = formal(row).get(key) or []
        if isinstance(value, list):
            preds.extend(str(x) for x in value if str(x).strip())
    seen: set[str] = set()
    out: list[str] = []
    for pred in preds:
        if pred not in seen:
            seen.add(pred)
            out.append(pred)
    return out


def pred_score(pred: str, row: dict[str, Any]) -> tuple[int, int, int, int]:
    q_syms = symbols(row.get("question", ""))
    p_syms = symbols(pred)
    return (
        int("Find(" in pred),
        len(q_syms & p_syms),
        int(bool(re.search(r"(?<![A-Za-z])-?\d+(?:\.\d+)?", pred))),
        int(any(k in pred for k in ("Equals", "Parallel", "Perpendicular", "Similar", "Congruent", "Tangent", "Bisects", "IsMidpoint"))),
    )


def ranked_predicates(row: dict[str, Any], k: int = 12) -> list[str]:
    return sorted(all_predicates(row), key=lambda p: pred_score(p, row), reverse=True)[:k]


def simplify_predicate(pred: str) -> str:
    text = pred
    for old, new in [
        ("LengthOf(Line(", "length("),
        ("MeasureOf(Angle(", "angle("),
        ("RadiusOf(Circle(", "radius("),
        ("Line(", ""),
        ("Angle(", "angle("),
        ("Circle(", "circle("),
    ]:
        text = text.replace(old, new)
    return text.replace("))", ")")


def structured_facts(row: dict[str, Any]) -> str:
    preds = ranked_predicates(row, k=12)
    body = "\n".join(f"- {simplify_predicate(p)}" for p in preds) or "- none"
    return "[STRUCTURED FACTS]\nGeometry3K oracle formal predicates, shortened and answer-neutral:\n" + body


def theorem_cues(row: dict[str, Any], preds: list[str]) -> list[str]:
    text = " ".join([row.get("question", "")] + preds)
    low = text.lower()
    cues: list[str] = []
    if "right" in low or "perpendicular" in low or "90" in text:
        cues.append("Check right-triangle, perpendicular-angle, Pythagorean, or trig relations.")
    if "similar" in low or "proportion" in low:
        cues.append("Match corresponding parts and set a proportion.")
    if "congruent" in low:
        cues.append("Transfer equal sides or angles between corresponding parts.")
    if any(x in low for x in ["circle", "chord", "arc", "tangent", "radius", "diameter"]):
        cues.append("Use circle angle, radius, chord, tangent, or arc rules.")
    if any(x in low for x in ["parallel", "parallelogram", "rectangle", "rhombus", "square"]):
        cues.append("Use parallel-line angle rules or quadrilateral side/diagonal properties.")
    if "area" in low:
        cues.append("Identify base-height, decomposition, or subtraction of regions.")
    if "perimeter" in low:
        cues.append("Solve missing side lengths before summing the boundary.")
    if not cues:
        cues.append("Translate the visible relations into the shortest equation for the target.")
    return cues[:4]


def auxiliary_candidates(row: dict[str, Any]) -> list[str]:
    text = " ".join([row.get("question", "")] + ranked_predicates(row, k=12))
    low = text.lower()
    out: list[str] = []
    if any(key in low for key in ("circle", "radius", "diameter", "tangent", "chord", "arc")):
        out.append("Circle construction: connect center to tangent point/chord endpoint; mark equal radii; use perpendicular radius-to-tangent/chord if applicable.")
    if any(key in low for key in ("similar", "proportion", "scale")):
        out.append("Similarity construction: identify corresponding vertices and align matching sides before setting a proportion.")
    if any(key in low for key in ("right", "perpendicular", "height", "altitude", "distance")):
        out.append("Right-triangle construction: drop or identify a perpendicular/altitude and use Pythagorean or trig relations.")
    if any(key in low for key in ("parallel", "angle", "transversal")):
        out.append("Angle construction: trace parallel/transversal angle pairs, vertical angles, supplements, and triangle angle sums.")
    if any(key in low for key in ("midpoint", "bisect", "median")):
        out.append("Bisection construction: split the relevant segment/angle into equal parts and transfer equality.")
    if "area" in low:
        out.append("Area construction: choose base-height or decompose the region into standard shapes.")
    if not out:
        out.append("Minimal construction: avoid adding extra lines unless a visible relation cannot be used directly.")
    return out[:4]


def oracle_construction_plan(row: dict[str, Any]) -> str:
    facts = structured_facts(row)
    candidates = "\n".join(f"- {item}" for item in auxiliary_candidates(row))
    cues = "\n".join(f"- {cue}" for cue in theorem_cues(row, ranked_predicates(row, k=12)))
    return (
        "Construction-aware intermediate language. It proposes answer-neutral helper constructions and a theorem route.\n\n"
        "[KNOWN FACTS]\n"
        + facts
        + "\n\n[AUXILIARY CONSTRUCTION CANDIDATES]\n"
        + candidates
        + "\n\n[THEOREM ROUTE]\n"
        + cues
        + "\n\n[EXECUTION]\n"
        "- Pick at most one useful construction; do not invent unsupported geometry.\n"
        "- After each visual/theorem step, re-check the relevant image mark or label.\n"
        "- Form the equation or comparison, compute the target, then choose one of the available options."
    )


def generator_prompt(row: dict[str, Any], gdp_text: str) -> str:
    choices = row.get("choices") or []
    choice_text = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    return f"""You are a geometry intermediate-language generator.
Given a geometry problem, produce a construction-aware intermediate language.
The language must be answer-neutral: do not choose an option and do not solve numerically unless it is a stated fact.

Question: {row.get('question','')}
Choices:
{choice_text}

GDP automatic diagram parse:
{gdp_text or 'No GDP parse available.'}

Output only the construction-aware intermediate language."""


def generate_plan(row: dict[str, Any], gdp_text: str, model_bundle: dict[str, Any], max_new_tokens: int = 420) -> str:
    import torch

    tok = model_bundle["tokenizer"]
    model = model_bundle["model"]
    prompt = f"<|im_start|>user\n{generator_prompt(row, gdp_text).strip()}<|im_end|>\n<|im_start|>assistant\n"
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)


def load_generator(base_model: str, adapter: str) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quant,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return {"tokenizer": tok, "model": model}


def token_set(text: str) -> set[str]:
    toks = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text or "")
    return {t.lower() for t in toks if len(t) > 1 or t.isdigit()}


def verify_generated_plan(plan: str, row: dict[str, Any], gdp_text: str) -> str:
    support = token_set(row.get("question", "")) | token_set(gdp_text)
    out_lines: list[str] = []
    dropped: list[str] = []
    for line in plan.splitlines():
        stripped = line.strip()
        if stripped.startswith("-") and any(key in stripped for key in ("Equals", "length(", "angle(", "Parallel", "Perpendicular", "PointLiesOn")):
            toks = token_set(stripped)
            nums = {t for t in toks if re.fullmatch(r"\d+(?:\.\d+)?", t)}
            letters = {t for t in toks if len(t) == 1 and t.isalpha()}
            relation_words = toks & {"parallel", "perpendicular", "equals", "length", "angle", "pointlieson"}
            # Keep target-like and broadly supported facts; drop suspicious facts with unsupported numbers/labels.
            unsupported_nums = nums - support
            unsupported_letters = letters - support
            if (unsupported_nums or len(unsupported_letters) >= 2) and relation_words:
                dropped.append(stripped)
                continue
        out_lines.append(line)
    note = (
        "\n\n[VERIFIER]\n"
        "- Generated facts were treated as parser hypotheses, not guaranteed facts.\n"
        "- Facts with unsupported labels or measurements were removed before solving.\n"
        "- Always trust the image and question over the generated plan."
    )
    if dropped:
        note += "\n- Removed suspicious generated facts: " + "; ".join(dropped[:6])
    return "\n".join(out_lines).strip() + note


def build_context(row: dict[str, Any], variant: str, gdp_text: str, generated_plan: str, verified_plan: str) -> str:
    if variant == "raw":
        return "No extra structured representation is provided."
    if variant == "image_gdp":
        return (
            "GDP-4B automatic geometric parsing output. Use it only as auxiliary evidence; trust the image and question if there is conflict.\n"
            f"<GDP_PARSE>\n{gdp_text}\n</GDP_PARSE>"
        )
    if variant == "generated_construction_plan":
        return (
            "Automatically generated construction-aware intermediate language. It may contain parser errors; trust the image and question if there is conflict.\n"
            f"<GENERATED_CONSTRUCTION_PLAN>\n{generated_plan}\n</GENERATED_CONSTRUCTION_PLAN>"
        )
    if variant == "generated_plan_with_verifier":
        return (
            "Verified generated construction-aware intermediate language. Unsupported generated facts were filtered by a lightweight verifier; still trust the image and question if there is conflict.\n"
            f"<VERIFIED_GENERATED_PLAN>\n{verified_plan}\n</VERIFIED_GENERATED_PLAN>"
        )
    if variant == "oracle_construction_plan":
        return (
            "Oracle-guided construction-aware intermediate language from Geometry3K annotations. Use it as auxiliary theorem guidance while checking the image.\n"
            f"<ORACLE_CONSTRUCTION_PLAN>\n{oracle_construction_plan(row)}\n</ORACLE_CONSTRUCTION_PLAN>"
        )
    raise ValueError(variant)


def build_messages(row: dict[str, Any], variant: str, gdp_text: str, generated_plan: str, verified_plan: str, image_path: Path) -> list[dict[str, Any]]:
    choices = row.get("choices", [])
    choice_text = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
    valid_letters = "/".join(chr(65 + i) for i in range(len(choices)))
    prompt = f"""Solve this geometry multiple-choice problem.

Question:
{row["question"]}

Choices:
{choice_text}

Additional context:
{build_context(row, variant, gdp_text, generated_plan, verified_plan)}

Return exactly one line and nothing else:
ANSWER: X

X must be one of {valid_letters}."""
    return [{"role": "user", "content": [{"type": "image", "image": str(image_path.resolve())}, {"type": "text", "text": prompt}]}]


def summarize(rows: list[dict[str, Any]], variants: list[str]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_variant[row["variant"]].append(row)
        by_id[row["id"]][row["variant"]] = row
    summary: dict[str, Any] = {"variants": {}, "pairwise_vs_raw": {}, "n_unique_ids": len(by_id)}
    for variant in variants:
        items = by_variant.get(variant, [])
        complete = [r for r in items if r.get("complete_response") and not r.get("error")]
        correct = [r for r in complete if r.get("correct")]
        summary["variants"][variant] = {
            "n": len(items),
            "complete": len(complete),
            "correct": len(correct),
            "accuracy": len(correct) / len(items) if items else 0.0,
            "accuracy_on_complete": len(correct) / len(complete) if complete else 0.0,
            "avg_duration_s": sum(float(r.get("duration_s", 0.0)) for r in items) / len(items) if items else 0.0,
            "errors": sum(1 for r in items if r.get("error")),
            "missing_answer_line": sum(1 for r in items if r.get("missing_answer_line")),
        }
    for variant in variants:
        if variant == "raw":
            continue
        win = loss = both_correct = both_wrong = incomplete = 0
        examples = {"win": [], "loss": []}
        for rid, item in by_id.items():
            raw = item.get("raw")
            other = item.get(variant)
            if not raw or not other or not raw.get("complete_response") or not other.get("complete_response"):
                incomplete += 1
            elif other.get("correct") and not raw.get("correct"):
                win += 1
                if len(examples["win"]) < 8:
                    examples["win"].append(rid)
            elif raw.get("correct") and not other.get("correct"):
                loss += 1
                if len(examples["loss"]) < 8:
                    examples["loss"].append(rid)
            elif raw.get("correct") and other.get("correct"):
                both_correct += 1
            else:
                both_wrong += 1
        summary["pairwise_vs_raw"][variant] = {
            "win": win,
            "loss": loss,
            "net": win - loss,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "incomplete": incomplete,
            "examples": examples,
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="multidataset_2x300_eval.jsonl")
    ap.add_argument("--gdp", default="gdp4b_geometry300_parse.jsonl")
    ap.add_argument("--qwen-vl-model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--generator-base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--generator-adapter", default="plan_generator_runs/gdp_to_plan_qwen17b_lora_8ep/final")
    ap.add_argument("--plan-cache", default="generated_plan_cache_200.jsonl")
    ap.add_argument("--out", default="qwen3vl8b_generated_plan_downstream200.jsonl")
    ap.add_argument("--summary", default="qwen3vl8b_generated_plan_downstream200_summary.json")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--plan-max-new-tokens", type=int, default=420)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--skip-plan-generation", action="store_true")
    args = ap.parse_args()

    rows = [r for r in read_jsonl(Path(args.dataset)) if r.get("dataset") == "Geometry3K-300"]
    gdp_by_id = {r["id"]: r for r in read_jsonl(Path(args.gdp)) if r.get("error") is None}
    rows = [r for r in rows if r["id"] in gdp_by_id][: args.limit]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    plan_cache_path = Path(args.plan_cache)
    plan_by_id: dict[str, dict[str, Any]] = {}
    if plan_cache_path.exists():
        for item in read_jsonl(plan_cache_path):
            plan_by_id[item["id"]] = item

    if not args.skip_plan_generation:
        bundle = None
        for row in rows:
            if row["id"] in plan_by_id:
                continue
            if bundle is None:
                bundle = load_generator(args.generator_base, args.generator_adapter)
            gdp_text = gdp_by_id[row["id"]].get("gdp_text", "")
            started = time.time()
            plan = generate_plan(row, gdp_text, bundle, max_new_tokens=args.plan_max_new_tokens)
            verified = verify_generated_plan(plan, row, gdp_text)
            rec = {
                "id": row["id"],
                "generated_plan": plan,
                "verified_plan": verified,
                "gdp_text_chars": len(gdp_text),
                "plan_chars": len(plan),
                "verified_plan_chars": len(verified),
                "duration_s": time.time() - started,
            }
            append_jsonl(plan_cache_path, rec)
            plan_by_id[row["id"]] = rec
            print(f"PLAN\t{row['id']}\tchars={len(plan)}\tduration_s={rec['duration_s']:.2f}", flush=True)
        if bundle is not None:
            del bundle

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True, "torch_dtype": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            compute_dtype = torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    processor = AutoProcessor.from_pretrained(args.qwen_vl_model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.qwen_vl_model, **kwargs)
    model.eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else ({int(eos)} if eos is not None else set())

    out_path = Path(args.out)
    done: set[tuple[str, str]] = set()
    results: list[dict[str, Any]] = []
    if out_path.exists():
        for item in read_jsonl(out_path):
            results.append(item)
            done.add((item["id"], item["variant"]))

    for row in rows:
        gdp_text = gdp_by_id[row["id"]].get("gdp_text", "")
        plan_item = plan_by_id.get(row["id"], {})
        generated_plan = plan_item.get("generated_plan", "")
        verified_plan = plan_item.get("verified_plan", "")
        image_path = Path(str(row["image"]))
        for variant in variants:
            if (row["id"], variant) in done:
                print("SKIP", row["id"], variant, flush=True)
                continue
            started = time.time()
            text = ""
            error = None
            new_len = 0
            truncated = False
            try:
                messages = build_messages(row, variant, gdp_text, generated_plan, verified_plan, image_path)
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
                new_tokens = trimmed[0].tolist()
                new_len = len(new_tokens)
                truncated = new_len >= args.max_new_tokens and (not eos_ids or new_tokens[-1] not in eos_ids)
                text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            valid_letters = "".join(chr(65 + i) for i in range(len(row.get("choices", []))))
            pred = normalize_answer(text, valid_letters=valid_letters or "ABCDE")
            complete = error is None and not truncated and pred != ""
            correct = complete and pred == str(row["answer"]).strip().upper()
            rec = {
                "dataset": "Geometry3K-200-GDP",
                "id": row["id"],
                "variant": variant,
                "answer": row["answer"],
                "prediction": pred,
                "correct": correct,
                "complete_response": complete,
                "truncated": truncated,
                "missing_answer_line": pred == "",
                "new_tokens": new_len,
                "duration_s": time.time() - started,
                "raw_response": text,
                "error": error,
                "gdp_text_chars": len(gdp_text),
                "generated_plan_chars": len(generated_plan),
                "verified_plan_chars": len(verified_plan),
            }
            append_jsonl(out_path, rec)
            results.append(rec)
            done.add((row["id"], variant))
            print(f"{'OK' if correct else 'FAIL'}\t{row['id']}\t{variant}\tpred={pred!r}\tgold={row['answer']!r}\tcomplete={complete}\tduration_s={rec['duration_s']:.2f}", flush=True)

    summary = summarize(results, variants)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
