# IR_mllm

Experiment artifacts for intermediate geometry language, GDP parses, theorem-route generation, and downstream MLLM evaluation.

Chinese README: [`README_zh.md`](README_zh.md)

## Contents

- `tools/`: training, parsing, data-construction, and downstream evaluation scripts.
- `data_artifacts/`: zipped experiment data artifacts, including FormalGeo route-generator SFT data, GDP-4B parses, resized images, construction-plan pseudo-labels, and Geometry3K downstream data.
- `results/`: JSON/JSONL/log outputs from trust calibration, answer steering, GDP, construction-plan, theorem-route, Geometry3K, MathVista, and FormalGeo experiments.
- `models/formalgeo_route_qwen17b_lora/`: Qwen3-1.7B LoRA adapter trained to map GDP parse + question to theorem route.
- `models/gdp_to_plan_qwen17b_lora_8ep/`: earlier Qwen3-1.7B LoRA adapter trained to map GDP parse + question to construction plan.
- `paper_arxiv_demo/`: English and Chinese arXiv-style demo sources, figures, tables, result snapshots, and compiled PDFs.

## Main Results Snapshot

### Safe compact theorem-route generator

Latest Geometry3K held-out 200 downstream evaluation:

| Variant | Correct | Total | Accuracy | Win vs raw | Loss vs raw |
|---|---:|---:|---:|---:|---:|
| image_only | 104 | 200 | 52.0% | -- | -- |
| old_route | 109 | 200 | 54.5% | 15 | 10 |
| compact_route | 112 | 200 | 56.0% | 12 | 4 |
| compact_route_qc | 108 | 200 | 54.0% | 7 | 3 |
| confidence_route | 109 | 200 | 54.5% | 14 | 9 |
| confidence_route_qc | 107 | 199 | 53.5% | 11 | 8 |

The main finding is that short, deduplicated theorem routes are more useful than longer unconstrained routes. The compact route generator improves raw Qwen3-VL accuracy by +4.0 points and reduces route-induced losses from 10 cases under the older route generator to 4 cases.

Full Geometry3K-601 downstream evaluation:

| Variant | Correct | Total | Accuracy | Win vs raw | Loss vs raw |
|---|---:|---:|---:|---:|---:|
| image_only | 323 | 601 | 53.74% | -- | -- |
| old_route | 329 | 601 | 54.74% | 40 | 34 |
| compact_route | 334 | 601 | 55.57% | 27 | 16 |

The 601-problem run confirms the same direction at a larger scale: compact routes give the best accuracy and reduce route-induced loss from 34 to 16 compared with the older route generator. The compact generator also constrains route length more aggressively (`avg_steps=1.18`, `max_steps=3`) than the older route generator (`avg_steps=3.07`, `max_steps=30`).

### GDP-to-theorem-route generator

- Training source: FormalGeo theorem sequences.
- GDP parsed images: 900.
- Split: 600 train / 100 validation / 200 downstream test.
- Base model: `Qwen/Qwen3-1.7B`.
- Final validation loss / perplexity: `0.13796 / 1.14792`.

Geometry3K-150 downstream:

| Variant | Correct | Total | Accuracy |
|---|---:|---:|---:|
| image_only | 76 | 150 | 50.67% |
| image + generated theorem route | 82 | 150 | 54.67% |

Pairwise against image-only: win 16, loss 10, net +6.

FormalGeo-200 downstream was intentionally paused at the user's request. The checkpoint file is `results/qwen3vl8b_formalgeo_generated_route200.jsonl`.

### Earlier construction-plan generator

Geometry3K-200 downstream:

| Variant | Correct | Total | Accuracy |
|---|---:|---:|---:|
| raw | 106 | 200 | 53.0% |
| image + GDP | 104 | 200 | 52.0% |
| image + generated construction plan | 108 | 200 | 54.0% |

## Large External Data

The remote server also contained larger third-party raw archives, including MathVista and FormalGeo source archives. They are intentionally not committed here because they are large and may have redistribution constraints. This repository keeps the derived experiment splits, parses, outputs, and model adapters needed to inspect and reproduce the reported experiments. The derived data directories are stored as zip archives under `data_artifacts/` to avoid committing thousands of small files.

## External Artifacts

Hugging Face upload was paused at the user's request. The current local archive keeps the final adapters and experiment results:

- Local final adapters: `/Users/gsy/Desktop/topic/local_model_adapters`
- Local result archive: `/Users/gsy/Desktop/topic/archive_upload/remote_topic_results`

See `MANIFEST.md` for a detailed inventory and omitted large-file notes.
