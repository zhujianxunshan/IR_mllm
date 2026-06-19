# Artifact Manifest

This repository is the cleaned GitHub package for **Selective Exposure of Intermediate Geometry Language**. It contains paper sources, scripts, derived data, adapters, and result snapshots needed to inspect and reproduce the main claims.

## Paper

- `paper_arxiv_demo/main.tex`: English paper draft.
- `paper_arxiv_demo/main_zh.tex`: Chinese paper draft.
- `paper_arxiv_demo/main.pdf`: compiled English PDF.
- `paper_arxiv_demo/main_zh.pdf`: compiled Chinese PDF.
- `paper_arxiv_demo/figures/`: paper figures, including selective-exposure overview, win/loss decomposition, compact-route cross-model results, route help/harm examples, and candidate comparisons.
- `paper_arxiv_demo/tables/`: LaTeX tables for earlier Geometry3K trust experiments.
- `paper_arxiv_demo/results_snapshot/`: compact result snapshots used by the paper.

## Code

- `tools/make_formalgeo_route_data.py`: build route-generator SFT examples from FormalGeo theorem sequences and GDP parses.
- `tools/train_qwen_plan_lora.py`: Qwen3-1.7B LoRA SFT trainer.
- `tools/qwen3vl_geometry3k_generated_route_eval.py`: Geometry3K generated-route downstream evaluator.
- `tools/qwen3vl_formalgeo_generated_route_eval.py`: FormalGeo downstream evaluator.
- `tools/qwen3vl_route_verifier_601_eval.py`: route-verifier downstream evaluator.
- `tools/qwen3vl_safe_route_eval.py`: compact/safe-route downstream evaluator.
- `tools/make_route_verifier_data.py`, `tools/make_route_verifier_601_data.py`: route usefulness/verifier data builders.
- `target_binding_verifier/`: balanced target-binding verifier data/training launcher.

## Included Derived Data

- `data_artifacts/formalgeo_route_data.zip`: FormalGeo route-generator SFT data, GDP parses, parser rows, and resized images.
- `data_artifacts/construction_plan_data.zip`: earlier construction-plan pseudo-label data.
- `data_artifacts/geometry3k_data.zip`: Geometry3K downstream data artifact.
- `data/tcrp_route_policy/`: small split/id metadata used by route-policy experiments.

## Included Models

- `models/formalgeo_route_qwen17b_lora/final/`: LoRA adapter for GDP-parse-to-theorem-route generation.
- `models/formalgeo_route_qwen17b_lora/config.json`, `summary.json`: training metadata.
- `models/gdp_to_plan_qwen17b_lora_8ep/final/`: earlier LoRA adapter for GDP-to-construction-plan generation.

All adapter files are below GitHub's 100 MB single-file limit and are committed directly.

## Included Result Snapshots

Representative files:

- `results/generated_route_policy_geoqa271_summary.json`: GeoQA271 raw/route/heuristic-policy route result.
- `results/generated_route_trust_ablation_geoqa120_summary.json`: GeoQA120 presentation-interface ablation.
- `results/generated_route_equation_candidates_geoqa220_summary.json`: GeoQA220 theorem/equation candidate comparison.
- `results/generated_route_cascade_gate_geoqa271_summary.json`: strict-gate and cascade ablation.
- `results/safe_route_601/qwen3vl8b_safe_route_test601_raw_old_compact_summary.json`: Geometry3K-601 compact-route result.
- `results/qwen3vl8b_route_verifier_downstream200_4var_summary.json`: preliminary verifier downstream result.
- `results/qwen3vl8b_route_verifier_extra100_4var_summary.json`: extra verifier downstream result.
- `results/target_binding_verifier_601_summary.json`: first target-binding verifier summary.

## Omitted Large Raw Archives

The following raw third-party archives existed in earlier remote workspaces but are not committed:

- MathVista image archive.
- FormalGeo full raw archive.
- Full downloaded model checkpoints for Qwen3-VL and GDP-4B.

Reasons: large size, possible redistribution constraints, and redundant availability from original dataset/model providers. The repository keeps derived splits, parses, outputs, and adapters needed to inspect the paper results.

## Remote Notes

Historical remote servers used during the experiments included:

- `ubuntu@43.128.112.148`
- `ubuntu@43.166.7.63`

The current repository is designed so that a new host can reproduce the workflow from the committed scripts and [docs/TARGET_BINDING_VERIFIER_REPRO.md](docs/TARGET_BINDING_VERIFIER_REPRO.md), assuming access to required base models and source datasets.
