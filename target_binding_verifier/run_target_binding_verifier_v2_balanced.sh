#!/usr/bin/env bash
set -euo pipefail
cd ~/topic_results

mkdir -p logs results data/target_binding_verifier_v2_balanced plan_generator_runs
LOG=logs/target_binding_verifier_v2_balanced.log
exec > >(tee -a "$LOG") 2>&1

PY=~/miniforge3/envs/qwen/bin/python

if [[ "${WAIT_FOR_PREV:-0}" == "1" ]]; then
  echo "[$(date)] waiting for previous target_binding_verifier_601 summary"
  until [[ -f results/target_binding_verifier_601_summary.json ]]; do
    date
    if [[ -f results/target_binding_verifier_601_cache.jsonl ]]; then
      wc -l results/target_binding_verifier_601_cache.jsonl
    fi
    sleep 120
  done
fi

echo "[$(date)] target-binding verifier v2 balanced experiment start"
echo "Using Python: $PY"
$PY - <<'PYCHK'
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
PYCHK

$PY tools/make_target_binding_verifier_v2_balanced.py \
  --source-dir data/target_binding_verifier_601 \
  --outdir data/target_binding_verifier_v2_balanced

$PY tools/train_qwen_plan_lora.py \
  --model Qwen/Qwen3-1.7B \
  --train data/target_binding_verifier_v2_balanced/train.jsonl \
  --val data/target_binding_verifier_v2_balanced/val.jsonl \
  --outdir plan_generator_runs/target_binding_verifier_v2_balanced_qwen17b_lora \
  --max-length 2048 \
  --batch-size 1 \
  --grad-accum 8 \
  --epochs 2.0 \
  --lr 1.2e-4 \
  --warmup-ratio 0.05 \
  --lora-r 16 \
  --lora-alpha 32 \
  --save-every 80 \
  --eval-every 40

$PY tools/eval_target_binding_verifier.py \
  --adapter plan_generator_runs/target_binding_verifier_v2_balanced_qwen17b_lora/final \
  --cache results/target_binding_verifier_v2_balanced_cache.jsonl \
  --summary results/target_binding_verifier_v2_balanced_summary.json \
  --limit 200 \
  --max-new-tokens 260

echo "[$(date)] target-binding verifier v2 balanced experiment done"
