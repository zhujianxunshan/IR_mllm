# Selective Route Exposure

本仓库保存论文 **Selective Exposure of Intermediate Geometry Language** 的代码、论文草稿、LoRA adapter、实验结果快照和复现实验说明。

> 几何中间表示不是越结构化越有用。我们的实验显示，真正有价值的单位是紧凑的 theorem-level action；而它是否应该暴露给下游 MLLM，需要由 trust policy 判断。

[English README](README.md) · [英文论文 PDF](paper_arxiv_demo/main.pdf) · [中文论文 PDF](paper_arxiv_demo/main_zh.pdf) · [Verifier 复现说明](docs/TARGET_BINDING_VERIFIER_REPRO.md)

![Selective exposure overview](paper_arxiv_demo/figures/selective_exposure_overview.png)

## 仓库内容

- `paper_arxiv_demo/`：最新中英文 LaTeX 论文、PDF、图表、表格和结果快照。
- `tools/`：数据构建、GDP 解析、route 生成、Qwen3-VL 下游评测、route verifier 相关脚本。
- `target_binding_verifier/`：第二版 balanced target-binding verifier 脚本。
- `results/`：Geometry3K、GeoQA、PGPS9K、candidate、trust-policy 和 verifier 实验结果。
- `models/`：`Qwen/Qwen3-1.7B` LoRA adapter。
- `data_artifacts/`：压缩后的派生实验数据。
- `docs/`：远程实验复现文档和科研经历简介。

第三方原始大数据包未直接放入 GitHub，原因是体积较大且可能有再分发限制。详见 [MANIFEST.md](MANIFEST.md)。

## 核心观点

本文比较三类几何中间语言：

| 中间单位 | 作用 | 结论 |
|---|---|---|
| 描述性 facts | GDP-style 几何事实或 oracle predicates | 不稳定，可能无效甚至误导 MLLM。 |
| compact theorem routes | 短定理行动序列 | 有用，尤其能帮助较小 VLM，但会破坏部分 raw-correct 样例。 |
| theorem-equation candidates | 将 route 拆成定理绑定和方程候选 | 适合作为 route 置信度不足时的 fallback，不是 route screening 的替代品。 |

最终策略是级联式选择性暴露：

1. 先对 generated route 做轻量 trust screening；
2. 高置信时暴露 route；
3. 置信不足时降级为 theorem-equation candidates；
4. 仍不可信时回退到 raw image solving。

当前最强的 `verified-policy` 更准确地说是 **heuristic trust policy / lightweight route screening**，不是训练出来的 verifier。仓库中的 learned target-binding verifier 是初步版本，单独报告。

## 主要结果

### GeoQA271：route 有用，但需要 trust policy

| Qwen3-VL-8B 输入 | 正确数 | 准确率 | 相对 raw 修正 | 相对 raw 带偏 |
|---|---:|---:|---:|---:|
| image only | 83 / 271 | 30.63% | -- | -- |
| image + generated route | 103 / 271 | 38.01% | 42 | 22 |
| image + heuristic-policy route | 110 / 271 | 40.59% | 45 | 18 |

显著性检查：

- raw -> generated route：McNemar `p ~= 0.0175`，bootstrap CI `+1.48%` 到 `+12.92%`；
- raw -> heuristic-policy route：McNemar `p ~= 0.00105`，bootstrap CI `+4.43%` 到 `+15.50%`。

### GeoQA candidate interface

| 划分 | 方法 | 正确数 | 准确率 |
|---|---|---:|---:|
| GeoQA220 | direct route | 82 / 220 | 37.27% |
| GeoQA220 | equation candidates | 84 / 220 | 38.18% |
| GeoQA220 | hybrid theorem+equation candidates | 86 / 220 | 39.09% |
| GeoQA271 | direct route | 103 / 271 | 38.01% |
| GeoQA271 | strict gate | 86 / 271 | 31.73% |
| GeoQA271 | cascade route/candidate/raw | 91 / 271 | 33.58% |
| GeoQA271 | heuristic-policy route | 110 / 271 | 40.59% |
| GeoQA271 | equation candidates | 107 / 271 | 39.48% |
| GeoQA271 | hybrid candidates | 108 / 271 | 39.85% |

解释：candidate exposure 是有价值的 fallback 和诊断接口；当前最强干预仍是 route-level trust control。

### 跨模型 compact route

![Compact-route multi-model result](paper_arxiv_demo/figures/compact_route_multimodel_accuracy.png)

| 下游 VLM | 数据集 | Image Only | Image + Compact Route |
|---|---|---:|---:|
| Qwen2.5-VL-3B | PGPS9K + Geometry3K-400 | 151 / 400 | 159 / 400 |
| InternVL3-2B | PGPS9K + Geometry3K-400 | 63 / 400 | 127 / 400 |

compact route 更准确地说是推理时 theorem-planning assistance，而不是严格意义上的蒸馏。它对较小 VLM 增益最大，对强 8B 模型增益更小。

### Geometry3K compact route

| 划分 | 方法 | 正确数 | 准确率 | 修正 | 带偏 |
|---|---|---:|---:|---:|---:|
| Geometry3K-200 | image only | 104 / 200 | 52.00% | -- | -- |
| Geometry3K-200 | old route | 109 / 200 | 54.50% | 15 | 10 |
| Geometry3K-200 | compact route | 112 / 200 | 56.00% | 12 | 4 |
| Geometry3K-601 | image only | 323 / 601 | 53.74% | -- | -- |
| Geometry3K-601 | old route | 329 / 601 | 54.74% | 40 | 34 |
| Geometry3K-601 | compact route | 334 / 601 | 55.57% | 27 | 16 |

### 初步 learned exposure verifier

| 评测 | Image Only | Image + Route | Verifier Filter | Verifier Rewrite |
|---|---:|---:|---:|---:|
| Geometry3K-200 | 104 / 200 | 110 / 200 | 104 / 200 | 105 / 200 |
| Extra Geometry3K-100 | 51 / 100 | 52 / 100 | 51 / 100 | 54 / 100 |

这个 verifier 能降低 route-induced harm，但也拒绝了很多有用 route，因此还不能替代 heuristic trust policy。下一步应训练更强的 target-binding verifier，并加入错误类型监督。

## 数据集与模型

数据集：

- **GeoQA**：`GeoQA120`、`GeoQA220`、`GeoQA271` 是同一过滤顺序上的确定性前缀。
- **Geometry3K**：200/300/601 题评测与完整 test artifact。
- **PGPS9K + Geometry3K**：400 题跨模型 compact-route 评测。
- **FormalGeo**：用于 theorem sequence 监督训练 route generator，也用于 oracle-route sanity check。
- **MathVista-MC**：早期跨数据集 stress test，说明几何专门结构不自动迁移到通用视觉推理。

模型：

- 下游求解器：`Qwen3-VL-8B`、`Qwen2.5-VL-3B`、`InternVL3-2B`。
- route/plan generator：`Qwen/Qwen3-1.7B` LoRA。
- 图形解析来源：GDP-4B 自动结构语言。

## 环境

```bash
conda create -n route-exposure python=3.10 -y
conda activate route-exposure
pip install torch transformers peft accelerate bitsandbytes pillow tqdm numpy scikit-learn
```

Qwen3-VL 推理需要按照本地 Qwen-VL runner 安装额外依赖。远程实验使用 CUDA PyTorch 和 Hugging Face model cache。

## 复现关键流程

### 构建 route generator 数据

```bash
python tools/make_formalgeo_route_data.py \
  --outdir data/formalgeo_route \
  --train-n 600 \
  --val-n 100 \
  --test-n 200
```

### 训练 Qwen3-1.7B route generator

```bash
python tools/train_qwen_plan_lora.py \
  --model Qwen/Qwen3-1.7B \
  --train data/formalgeo_route/train.jsonl \
  --val data/formalgeo_route/val.jsonl \
  --outdir models/formalgeo_route_qwen17b_lora \
  --max-length 2048 \
  --batch-size 1 \
  --grad-accum 8 \
  --epochs 3 \
  --lr 1.2e-4 \
  --lora-r 16 \
  --lora-alpha 32
```

### 训练/评测 target-binding verifier

```bash
bash target_binding_verifier/run_target_binding_verifier_v2_balanced.sh
```

完整远程复现流程见 [docs/TARGET_BINDING_VERIFIER_REPRO.md](docs/TARGET_BINDING_VERIFIER_REPRO.md)。

## 论文自查

本次推送前重点检查了这些问题：

- candidate 被定位为 fallback，而不是比 verified route 更强的最终接口；
- `verified-policy` 明确写成轻量 heuristic trust policy，而不是训练 verifier；
- 避免使用不准确的“蒸馏”表述，改为 inference-time planning assistance；
- FormalGeo oracle-route 只作为定性 sanity check，不外推为 GeoQA 上界；
- GeoQA 子集嵌套关系、same-split candidate 对比和 strict-gate ablation 已写入论文。

## 引用

这是研究草稿和 artifact package。正式发表前引用请使用仓库链接和 commit hash。

```bibtex
@misc{selective_route_exposure_2026,
  title  = {Selective Exposure of Intermediate Geometry Language},
  author = {Anonymous},
  year   = {2026},
  note   = {Research artifact repository}
}
```
