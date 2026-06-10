#!/usr/bin/env bash
set -euo pipefail

cd ~/topic_results
source ~/miniforge3/etc/profile.d/conda.sh
conda activate qwen

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=warning

DATA_DIR="data/route_verifier_601"
OUT_DIR="plan_generator_runs/route_verifier_601_qwen17b_lora"
VERIFIER_CACHE="results/route_verifier_601_cache.jsonl"
DOWNSTREAM="results/qwen3vl8b_route_verifier_601_gate.jsonl"
SUMMARY="results/qwen3vl8b_route_verifier_601_gate_summary.json"

echo "[$(date)] Stage 1/4: build leakage-controlled verifier data"
python tools/make_route_verifier_601_data.py \
  --dataset geometry3k_test_full.jsonl \
  --gdp results/gdp4b_geometry3k_test601_parse.jsonl \
  --route-cache results/safe_route_compact_geometry3k_test601.jsonl \
  --downstream-results results/qwen3vl8b_safe_route_test601_raw_old_compact.jsonl \
  --ids data/tcrp_route_policy/geometry3k_test601_ids.json \
  --outdir "$DATA_DIR" \
  --calibration-n 401 \
  --val-n 80 \
  --seed 61

echo "[$(date)] Stage 2/4: train Qwen3-1.7B LoRA verifier"
python tools/train_qwen_plan_lora.py \
  --model Qwen/Qwen3-1.7B \
  --train "$DATA_DIR/train.jsonl" \
  --val "$DATA_DIR/val.jsonl" \
  --outdir "$OUT_DIR" \
  --max-length 2048 \
  --batch-size 1 \
  --grad-accum 8 \
  --epochs 5 \
  --lr 1.5e-4 \
  --warmup-ratio 0.05 \
  --lora-r 16 \
  --lora-alpha 32 \
  --save-every 40 \
  --eval-every 20

echo "[$(date)] Stage 3/4: downstream held-out verifier gate evaluation"
python tools/qwen3vl_route_verifier_601_eval.py \
  --dataset geometry3k_test_full.jsonl \
  --test-ids "$DATA_DIR/test_ids.json" \
  --gdp results/gdp4b_geometry3k_test601_parse.jsonl \
  --route-cache results/safe_route_compact_geometry3k_test601.jsonl \
  --adapter "$OUT_DIR/final" \
  --verifier-cache "$VERIFIER_CACHE" \
  --out "$DOWNSTREAM" \
  --summary "$SUMMARY" \
  --limit 200 \
  --variants image_only,compact_route,verifier_gate,verifier_rewrite \
  --load-in-4bit \
  --verifier-max-new-tokens 360 \
  --answer-max-new-tokens 32

echo "[$(date)] Stage 4/4: summary"
cat "$SUMMARY"
