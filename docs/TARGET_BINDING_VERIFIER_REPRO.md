# Target-Binding Route Verifier Experiment Reproduction Notes

Updated: 2026-06-10 23:05 CST

## 1. Current Experiment Status

Two copies of the same experiment are running.

### Old server

- SSH: `ubuntu@43.128.112.148`
- Hostname: `VM-0-8-ubuntu`
- Project directory: `/home/ubuntu/topic_results`
- Screen session: `target_binding_verifier`
- Log: `/home/ubuntu/topic_results/logs/target_binding_verifier_601.log`
- Current status at last check: training still running, no final summary yet.
- Latest observed progress: `step 775`, `epoch 3`, `elapsed_s = 14973.9`.
- Recent validation: `step 760 val_loss = 0.008185`, `val_ppl = 1.008219`.
- GPU at last check: Tesla T4, `7166 MiB / 15360 MiB`, utilization `100%`.

### New T4 server

- SSH: `ubuntu@43.166.7.63`
- Hostname: `VM-0-11-ubuntu`
- Project directory: `/home/ubuntu/topic_results`
- Screen session: `target_binding_verifier`
- Log: `/home/ubuntu/topic_results/logs/target_binding_verifier_601.log`
- Current status at last check: training still running, no final summary yet.
- Latest observed progress: `step 650`, `epoch 2`, `elapsed_s = 12369.0`.
- Recent validation: `step 640 val_loss = 0.008965`, `val_ppl = 1.009005`.
- GPU at last check: Tesla T4, `9275 MiB / 16384 MiB`, utilization `97%`.

Expected training length is about 1017 optimizer steps, followed by a 200-example verifier evaluation. The old server should finish first.

## 2. Research Purpose

This experiment trains a target-binding route verifier. The goal is to decide whether a generated theorem route should be trusted for a specific geometry problem.

The key supervision idea is counterfactual:

- `raw wrong, route correct -> use_route`
- `raw correct, route wrong -> reject_route`

Hard negatives are not random wrong routes. They are locally plausible but target-mismatched routes, for example a true parallelogram theorem that does not serve the requested adjacent-angle equation.

Evaluation should report more than final accuracy:

- downstream accuracy after gating
- route win count
- route loss count
- route acceptance rate
- route-helpfulness confidence calibration

## 3. Environment

On both servers, the active Python environment is:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate qwen
```

Observed package versions:

```text
torch 2.5.1+cu121, cuda True
transformers 4.57.1
peft 0.19.1
accelerate 1.13.0
bitsandbytes 0.49.2
```

The old server disk state at last check:

```text
/home/ubuntu/topic_results
/dev/vda2 119G total, 74G used, 40G available, 66% used
```

## 4. Model Assets

Required base model:

```text
~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B
```

Observed size on old server:

```text
3.8G ~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B
```

Other relevant cached model:

```text
~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct
```

GDP metadata directory present in project:

```text
/home/ubuntu/topic_results/models_meta/GDP-4B
```

Note: this specific verifier training uses Qwen3-1.7B LoRA. It consumes previously generated downstream raw/route/parse result files; it does not run Qwen3-VL inference during training.

## 5. Main Run Script

Remote script:

```text
/home/ubuntu/topic_results/tools/run_target_binding_verifier.sh
```

Full command to launch:

```bash
cd ~/topic_results
screen -dmS target_binding_verifier bash -lc 'CUDA_VISIBLE_DEVICES=0 tools/run_target_binding_verifier.sh'
```

The script does three stages:

1. Build target-binding verifier data.
2. Train Qwen3-1.7B LoRA verifier/generator.
3. Evaluate on held-out 200 examples and write summary JSON.

Training command inside the script:

```bash
~/miniforge3/envs/qwen/bin/python tools/train_qwen_plan_lora.py \
  --model Qwen/Qwen3-1.7B \
  --train data/target_binding_verifier_601/train.jsonl \
  --val data/target_binding_verifier_601/val.jsonl \
  --outdir plan_generator_runs/target_binding_verifier_qwen17b_lora \
  --max-length 2048 \
  --batch-size 1 \
  --grad-accum 8 \
  --epochs 3 \
  --lr 1.2e-4 \
  --warmup-ratio 0.05 \
  --lora-r 16 \
  --lora-alpha 32 \
  --save-every 80 \
  --eval-every 40
```

Evaluation command inside the script:

```bash
~/miniforge3/envs/qwen/bin/python tools/eval_target_binding_verifier.py \
  --adapter plan_generator_runs/target_binding_verifier_qwen17b_lora/final \
  --cache results/target_binding_verifier_601_cache.jsonl \
  --summary results/target_binding_verifier_601_summary.json \
  --limit 200 \
  --max-new-tokens 260
```

## 6. Required Code Files

All are under `/home/ubuntu/topic_results`:

```text
tools/run_target_binding_verifier.sh
tools/make_target_binding_verifier_data.py
tools/eval_target_binding_verifier.py
tools/train_qwen_plan_lora.py
```

Observed sizes on old server:

```text
tools/eval_target_binding_verifier.py        8.3K
tools/make_target_binding_verifier_data.py   9.9K
tools/run_target_binding_verifier.sh         1.4K
tools/train_qwen_plan_lora.py                7.7K
```

## 7. Required Data and Intermediate Result Files

All are under `/home/ubuntu/topic_results`.

Data files generated by this experiment:

```text
data/target_binding_verifier_601/train.jsonl      2711 lines, 5.3M
data/target_binding_verifier_601/val.jsonl         100 lines, 194K
data/target_binding_verifier_601/test_ids.json       5.1K
data/target_binding_verifier_601/meta.json
```

Required upstream inputs:

```text
geometry3k_test_full.jsonl                                      601 lines, 4.7M
results/gdp4b_geometry3k_test601_parse.jsonl                    601 lines, 268K
results/qwen3vl8b_safe_route_test601_raw_old_compact.jsonl     1803 lines, 1.1M
results/geometry3k_old_route_test601.jsonl                      601 lines, 279K
results/safe_route_compact_geometry3k_test601.jsonl             601 lines, 267K
```

The `qwen3vl8b_safe_route_test601_raw_old_compact.jsonl` file has 1803 lines because it contains multiple variants per item.

## 8. Data Construction Metadata

Generated by:

```bash
python tools/make_target_binding_verifier_data.py \
  --outdir data/target_binding_verifier_601 \
  --calibration-n 401 \
  --val-n 100 \
  --hard-negative-per-id 1
```

Observed `meta.json`:

```json
{
  "train": 2711,
  "train_base": 1103,
  "val": 100,
  "test_ids": 200,
  "labels_train": {
    "both_correct": 364,
    "route_win": 135,
    "hard_negative": 1468,
    "both_wrong": 594,
    "route_loss": 150
  },
  "labels_base": {
    "both_correct": 364,
    "route_win": 45,
    "hard_negative": 367,
    "both_wrong": 297,
    "route_loss": 30
  },
  "labels_val": {
    "both_wrong": 25,
    "route_win": 7,
    "both_correct": 30,
    "hard_negative": 34,
    "route_loss": 4
  },
  "note": "Target-binding verifier trained on calibration split only; test_ids held out for counterfactual gate evaluation."
}
```

## 9. Output Files

Training output directory:

```text
plan_generator_runs/target_binding_verifier_qwen17b_lora
```

Observed size on old server at step 775:

```text
736M plan_generator_runs/target_binding_verifier_qwen17b_lora
```

Observed checkpoints include:

```text
checkpoint-80
checkpoint-160
checkpoint-240
checkpoint-320
checkpoint-400
checkpoint-480
checkpoint-560
checkpoint-640
checkpoint-720
```

Expected final adapter after training:

```text
plan_generator_runs/target_binding_verifier_qwen17b_lora/final
```

Expected evaluation outputs:

```text
results/target_binding_verifier_601_cache.jsonl
results/target_binding_verifier_601_summary.json
results/target_binding_verifier_601_summary.json.decisions.jsonl
```

At last check, these final result files had not yet appeared.

## 10. Monitoring Commands

Old server:

```bash
ssh ubuntu@43.128.112.148
cd ~/topic_results
screen -ls
tail -f logs/target_binding_verifier_601.log
nvidia-smi
```

New T4:

```bash
ssh ubuntu@43.166.7.63
cd ~/topic_results
screen -ls
tail -f logs/target_binding_verifier_601.log
nvidia-smi
```

Check final result:

```bash
cd ~/topic_results
ls -lh results/target_binding_verifier_601*
cat results/target_binding_verifier_601_summary.json
```

## 11. Notes for Migration to a Fresh Server

Minimum practical migration set:

```text
~/topic_results/tools/run_target_binding_verifier.sh
~/topic_results/tools/make_target_binding_verifier_data.py
~/topic_results/tools/eval_target_binding_verifier.py
~/topic_results/tools/train_qwen_plan_lora.py
~/topic_results/data/target_binding_verifier_601/
~/topic_results/geometry3k_test_full.jsonl
~/topic_results/results/gdp4b_geometry3k_test601_parse.jsonl
~/topic_results/results/qwen3vl8b_safe_route_test601_raw_old_compact.jsonl
~/topic_results/results/geometry3k_old_route_test601.jsonl
~/topic_results/results/safe_route_compact_geometry3k_test601.jsonl
~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/
```

If copying while a training job is still running, do not copy partially written checkpoint directories as a final model. Wait for `final/` or copy a complete checkpoint directory only after it stops changing.

Recommended environment install sketch:

```bash
conda create -n qwen python=3.11 -y
conda activate qwen
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.57.1 peft==0.19.1 accelerate==1.13.0 bitsandbytes==0.49.2 safetensors sentencepiece protobuf
```

If `transformers` is accidentally upgraded to 5.x, downgrade back to 4.57.1. The experiment was verified with 4.57.1.

## 12. Relation to Paper Demo

The paper demo has already been updated separately under:

```text
~/topic_results/paper_orchestra_ir_mllm/workspace/final/
~/topic_results/paper_orchestra_ir_mllm/workspace/final_zh/
```

The current target-binding verifier experiment should be added to the paper only after `results/target_binding_verifier_601_summary.json` is generated and inspected.
