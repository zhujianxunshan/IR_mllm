#!/usr/bin/env bash
set -euo pipefail

cd ~/topic_results
source ~/miniforge3/bin/activate qwen
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=warning

DATASET="geometry3k_test_full.jsonl"
GDP="results/gdp4b_geometry3k_test601_parse.jsonl"
IDS="data/tcrp_route_policy/geometry3k_test601_ids.json"
OLD_ROUTE="results/geometry3k_old_route_test601.jsonl"
COMPACT_ROUTE="results/safe_route_compact_geometry3k_test601.jsonl"
OUT="results/qwen3vl8b_safe_route_test601_raw_old_compact.jsonl"
SUMMARY="results/qwen3vl8b_safe_route_test601_raw_old_compact_summary.json"

echo "[$(date)] Prepare 601 ids and seed caches"
python - <<'PY'
import json
from pathlib import Path

ids = [json.loads(line)["id"] for line in open("geometry3k_test_full.jsonl", encoding="utf-8") if line.strip()]
Path("data/tcrp_route_policy").mkdir(parents=True, exist_ok=True)
Path("data/tcrp_route_policy/geometry3k_test601_ids.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")

def merge_unique(out, inputs):
    seen = set()
    rows = []
    for p in inputs:
        path = Path(p)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rid = str(row.get("id"))
            if rid in seen:
                continue
            seen.add(rid)
            rows.append(row)
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(out, len(rows))

merge_unique("results/geometry3k_old_route_test601.jsonl", [
    "results/geometry3k_trust_selection_route_cache_300.jsonl",
])
merge_unique("results/safe_route_compact_geometry3k_test601.jsonl", [
    "results/safe_route_compact_geometry3k_heldout200.jsonl",
    "results/safe_route_compact_geometry3k_heldout300.jsonl",
])
PY

echo "[$(date)] Wait for GDP parse to cover all 601 test rows"
while true; do
  count=$(python - <<'PY'
import json
from pathlib import Path
p=Path("results/gdp4b_geometry3k_test601_parse.jsonl")
print(sum(1 for line in p.open(encoding="utf-8") if line.strip()) if p.exists() else 0)
PY
)
  echo "[$(date)] GDP rows: ${count}/601"
  if [ "$count" -ge 601 ]; then
    break
  fi
  sleep 120
done

echo "[$(date)] Generate missing old routes for 601"
python tools/generate_route_cache_from_gdp.py \
  --dataset "$DATASET" \
  --gdp "$GDP" \
  --base-model Qwen/Qwen3-1.7B \
  --adapter plan_generator_runs/formalgeo_route_qwen17b_lora/final \
  --out "$OLD_ROUTE" \
  --limit 601 \
  --max-new-tokens 220

echo "[$(date)] Run downstream raw / old_route / compact_route on Geometry3K test601"
python tools/qwen3vl_safe_route_eval.py \
  --dataset "$DATASET" \
  --dataset-label ALL \
  --gdp "$GDP" \
  --test-ids "$IDS" \
  --old-route-cache "$OLD_ROUTE" \
  --compact-adapter plan_generator_runs/safe_route_compact_qwen17b_lora/final \
  --confidence-adapter plan_generator_runs/safe_route_confidence_qwen17b_lora/final \
  --compact-cache "$COMPACT_ROUTE" \
  --confidence-cache results/unused_confidence_test601.jsonl \
  --out "$OUT" \
  --summary "$SUMMARY" \
  --limit 601 \
  --variants image_only,old_route,compact_route \
  --load-in-4bit \
  --route-max-new-tokens 160 \
  --answer-max-new-tokens 32

echo "[$(date)] DONE"
cat "$SUMMARY"
