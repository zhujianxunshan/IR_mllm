# Artifact Manifest

Generated from remote server `ubuntu@43.128.112.148:~/topic_results` and local paper directory `/Users/gsy/Desktop/topic/paper_arxiv_demo`. Updated after the Geometry3K-601 safe-route evaluation completed on 2026-06-10.

## Included In GitHub

### Code

- `tools/gdp4b_formalgeo_parse.py`: GDP-4B parser runner for FormalGeo images.
- `tools/make_formalgeo_route_data.py`: builds route-generator SFT data from FormalGeo theorem sequences and GDP parses.
- `tools/train_qwen_plan_lora.py`: Qwen3-1.7B LoRA SFT trainer.
- `tools/qwen3vl_geometry3k_generated_route_eval.py`: Geometry3K downstream image-only vs image+generated-route evaluator.
- `tools/qwen3vl_formalgeo_generated_route_eval.py`: FormalGeo downstream evaluator for image-only, generated route, and oracle route.
- Additional earlier experiment scripts for construction-plan, GDP, trust, and multidataset evaluations are also preserved in `tools/`.

### Data

- `data_artifacts/formalgeo_route_data.zip`: FormalGeo route-generator SFT data, including 600 train, 100 validation, 200 downstream test examples, 900 GDP-4B parses, parser input rows, and 900 resized FormalGeo images.
- `data_artifacts/construction_plan_data.zip`: earlier construction-plan pseudo-label data.
- `data_artifacts/geometry3k_data.zip`: Geometry3K downstream data artifact used by the evaluation scripts.

### Models

- `models/formalgeo_route_qwen17b_lora/final/`: final LoRA adapter for GDP-parse-to-theorem-route generation.
- `models/formalgeo_route_qwen17b_lora/config.json`: training config.
- `models/formalgeo_route_qwen17b_lora/summary.json`: training summary.
- `models/gdp_to_plan_qwen17b_lora_8ep/final/`: final LoRA adapter for earlier GDP-to-construction-plan generation.
- `models/gdp_to_plan_qwen17b_lora_8ep/config.json`: training config.
- `models/gdp_to_plan_qwen17b_lora_8ep/summary.json`: training summary.

All included model adapter files are below GitHub's 100 MB single-file limit and are committed directly with Git.

### Results

`results/` contains root-level experiment outputs copied from the remote server, including:

- `qwen3vl8b_geometry3k_generated_route150_summary.json`
- `qwen3vl8b_geometry3k_generated_route150.jsonl`
- `qwen3vl8b_formalgeo_generated_route200.jsonl`
- `formalgeo_generated_route_cache_300.jsonl`
- `qwen3vl8b_generated_plan_downstream200_3var_summary.json`
- `qwen3vl8b_trust_full_601_summary.json`
- `qwen3vl8b_trust_stress_200_summary.json`
- `qwen3vl8b_answer_steer_200_summary.json`
- `results/safe_route_601/qwen3vl8b_safe_route_test601_raw_old_compact_summary.json`
- `results/safe_route_601/qwen3vl8b_safe_route_test601_raw_old_compact.jsonl`
- `results/safe_route_601/safe_route_compact_geometry3k_test601.jsonl`
- logs for the corresponding runs.

### Paper Demo

- `paper_arxiv_demo/main.tex`: English arXiv-style demo.
- `paper_arxiv_demo/main_zh.tex`: Chinese arXiv-style demo.
- `paper_arxiv_demo/main.pdf`: compiled English PDF.
- `paper_arxiv_demo/main_zh.pdf`: compiled Chinese PDF.
- `paper_arxiv_demo/figures/`, `paper_arxiv_demo/tables/`, `paper_arxiv_demo/results_snapshot/`.

## Omitted Large Raw Archives

The following raw third-party archives existed on the remote server but are not committed:

- `datasets/mathvista/images.zip`: about 866 MB.
- `datasets/formalgeo/formalgeo7k_v2.tar.gz`: about 521 MB.
- Other full raw dataset archives under `datasets/`.

Reason: these files are large, not necessary for inspecting the experiment outputs, and may have dataset-specific redistribution terms. For long-term archival, place them in Hugging Face Hub, Zenodo, or object storage, then add stable URLs here. Hugging Face upload was paused at the user's request; the current full local archive is kept under `/Users/gsy/Desktop/topic/archive_upload/remote_topic_results`, with final local adapters under `/Users/gsy/Desktop/topic/local_model_adapters`.

## Remote Paused Process

At the time this package was created, the FormalGeo-200 downstream run was paused:

```text
PID: 1036088
Command: python tools/qwen3vl_formalgeo_generated_route_eval.py ...
Output file: qwen3vl8b_formalgeo_generated_route200.jsonl
```

Resume on the remote server with:

```bash
kill -CONT 1036088
```
