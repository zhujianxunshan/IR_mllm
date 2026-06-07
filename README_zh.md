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

后续计划把较大的原始数据、派生数据和模型参数放到 Hugging Face，并在本仓库中补充下载链接。

详细文件清单见 `MANIFEST.md`。
