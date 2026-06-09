#!/usr/bin/env bash
set -euo pipefail

cd ~/topic_results
source ~/miniforge3/etc/profile.d/conda.sh
conda activate qwen

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=warning

COMPACT_DATA="data/safe_route_compact"
CONF_DATA="data/safe_route_confidence"
COMPACT_OUT="plan_generator_runs/safe_route_compact_qwen17b_lora"
CONF_OUT="plan_generator_runs/safe_route_confidence_qwen17b_lora"
COMPACT_CACHE="results/safe_route_compact_geometry3k_heldout200.jsonl"
CONF_CACHE="results/safe_route_confidence_geometry3k_heldout200.jsonl"
DOWNSTREAM="results/qwen3vl8b_safe_route_heldout200_6var.jsonl"
SUMMARY="results/qwen3vl8b_safe_route_heldout200_6var_summary.json"

echo "[$(date)] Stage 1/5: build compact route SFT data"
python tools/make_safe_route_training_data.py \
  --src-dir formalgeo_route_data_all \
  --outdir "$COMPACT_DATA" \
  --mode compact_sft \
  --train-limit 2600 \
  --val-limit 300 \
  --max-steps 3 \
  --seed 43

echo "[$(date)] Stage 2/5: build confidence-aware route SFT data"
python tools/make_safe_route_training_data.py \
  --src-dir formalgeo_route_data_all \
  --outdir "$CONF_DATA" \
  --mode confidence_sft \
  --train-limit 2600 \
  --val-limit 300 \
  --max-steps 3 \
  --neg-ratio 0.35 \
  --seed 47

echo "[$(date)] Stage 3/5: train compact route generator"
python tools/train_qwen_plan_lora.py \
  --model Qwen/Qwen3-1.7B \
  --train "$COMPACT_DATA/train.jsonl" \
  --val "$COMPACT_DATA/val.jsonl" \
  --outdir "$COMPACT_OUT" \
  --max-length 1536 \
  --batch-size 1 \
  --grad-accum 8 \
  --epochs 1.5 \
  --lr 2e-4 \
  --warmup-ratio 0.05 \
  --lora-r 16 \
  --lora-alpha 32 \
  --save-every 80 \
  --eval-every 40

echo "[$(date)] Stage 4/5: train confidence-aware route generator"
python tools/train_qwen_plan_lora.py \
  --model Qwen/Qwen3-1.7B \
  --train "$CONF_DATA/train.jsonl" \
  --val "$CONF_DATA/val.jsonl" \
  --outdir "$CONF_OUT" \
  --max-length 1536 \
  --batch-size 1 \
  --grad-accum 8 \
  --epochs 1.5 \
  --lr 2e-4 \
  --warmup-ratio 0.05 \
  --lora-r 16 \
  --lora-alpha 32 \
  --save-every 80 \
  --eval-every 40

echo "[$(date)] Stage 5/5: downstream Geometry3K held-out 200 evaluation"
python tools/qwen3vl_safe_route_eval.py \
  --dataset multidataset_2x300_eval.jsonl \
  --gdp gdp4b_geometry300_parse.jsonl \
  --test-ids data/tcrp_route_policy/test_ids.json \
  --old-route-cache results/geometry3k_trust_selection_route_cache_300.jsonl \
  --compact-adapter "$COMPACT_OUT/final" \
  --confidence-adapter "$CONF_OUT/final" \
  --compact-cache "$COMPACT_CACHE" \
  --confidence-cache "$CONF_CACHE" \
  --out "$DOWNSTREAM" \
  --summary "$SUMMARY" \
  --limit 200 \
  --load-in-4bit \
  --route-max-new-tokens 160 \
  --answer-max-new-tokens 32

echo "[$(date)] Done. Summary:"
cat "$SUMMARY"
