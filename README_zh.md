# IR_mllm 实验归档

本仓库保存几何题中间语言、GDP 自动解析、定理路线生成器，以及下游多模态大模型评测的代码、数据、结果和模型产物。

## 仓库内容

- `tools/`：GDP-4B 解析、数据构建、LoRA 训练、Geometry3K/FormalGeo 下游评测脚本。
- `data_artifacts/`：压缩后的派生实验数据，包括 FormalGeo route-generator SFT 数据、GDP-4B parses、resized images、construction-plan pseudo-labels 和 Geometry3K 下游数据。
- `results/`：所有实验产生的 JSON/JSONL/log 文件，包括 trust calibration、answer steering、GDP、construction plan、theorem route、Geometry3K、MathVista 和 FormalGeo 实验。
- `models/formalgeo_route_qwen17b_lora/`：从 `question + GDP parse` 预测 theorem route 的 Qwen3-1.7B LoRA adapter。
- `models/gdp_to_plan_qwen17b_lora_8ep/`：早期从 `question + GDP parse` 预测 construction plan 的 Qwen3-1.7B LoRA adapter。
- `paper_arxiv_demo/`：中英文 arXiv-style demo、PDF、图表和结果快照。

## 主要实验结论

### Safe compact theorem-route generator

最新 Geometry3K held-out 200 下游评测：

| 条件 | 正确数 | 总数 | 准确率 | 相对 raw 修正 | 相对 raw 带偏 |
|---|---:|---:|---:|---:|---:|
| image_only | 104 | 200 | 52.0% | -- | -- |
| old_route | 109 | 200 | 54.5% | 15 | 10 |
| compact_route | 112 | 200 | 56.0% | 12 | 4 |
| compact_route_qc | 108 | 200 | 54.0% | 7 | 3 |
| confidence_route | 109 | 200 | 54.5% | 14 | 9 |
| confidence_route_qc | 107 | 199 | 53.5% | 11 | 8 |

主要结论是：短、去重、目标相关的 theorem route 比较有用。`compact_route` 将 raw Qwen3-VL 从 52.0% 提高到 56.0%，同时把旧 route generator 造成的带偏样例从 10 个降到 4 个。

完整 Geometry3K-601 下游评测：

| 条件 | 正确数 | 总数 | 准确率 | 相对 raw 修正 | 相对 raw 带偏 |
|---|---:|---:|---:|---:|---:|
| image_only | 323 | 601 | 53.74% | -- | -- |
| old_route | 329 | 601 | 54.74% | 40 | 34 |
| compact_route | 334 | 601 | 55.57% | 27 | 16 |

601 题结果确认了同一趋势：`compact_route` 准确率最高，并且相比旧 route generator，将 route-induced loss 从 34 降到 16。它也更强地约束 route 长度，`avg_steps=1.18`、`max_steps=3`，而旧 route generator 为 `avg_steps=3.07`、`max_steps=30`。

### GDP-to-theorem-route generator

训练数据来自 FormalGeo theorem sequences：

- GDP-4B 自动解析图像：900 张。
- 数据划分：600 train / 100 validation / 200 downstream test。
- 基础模型：`Qwen/Qwen3-1.7B`。
- 最终验证 loss / perplexity：`0.13796 / 1.14792`。

Geometry3K-150 下游结果：

| 条件 | 正确数 | 总数 | 准确率 |
|---|---:|---:|---:|
| image_only | 76 | 150 | 50.67% |
| image + generated theorem route | 82 | 150 | 54.67% |

相对 image-only 的 pairwise flips：修正 16 题，破坏 10 题，净收益 +6。

FormalGeo-200 下游实验根据用户要求在 checkpoint 暂停，当前输出文件为：

```text
results/qwen3vl8b_formalgeo_generated_route200.jsonl
```

### 早期 construction-plan generator

Geometry3K-200 下游结果：

| 条件 | 正确数 | 总数 | 准确率 |
|---|---:|---:|---:|
| raw | 106 | 200 | 53.0% |
| image + GDP | 104 | 200 | 52.0% |
| image + generated construction plan | 108 | 200 | 54.0% |

## 数据说明

GitHub 仓库中包含派生实验数据、splits、GDP parses、实验输出和 LoRA adapter。远程服务器上更大的第三方原始数据包没有直接放入 GitHub，原因是体积较大且可能有再分发限制。

Hugging Face 上传已根据用户要求暂停。当前本机已经保存最终 adapter 和实验结果：

- 本机最终 adapter：`/Users/gsy/Desktop/topic/local_model_adapters`
- 本机结果归档：`/Users/gsy/Desktop/topic/archive_upload/remote_topic_results`

详细文件清单见 `MANIFEST.md`。
