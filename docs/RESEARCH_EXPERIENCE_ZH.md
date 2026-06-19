# 科研工作经历（约 300 字）

围绕多模态大模型的几何推理可靠性，我系统研究了“中间几何语言是否真正有用”这一问题。工作从 GDP-4B 自动结构化解析、Geometry3K/GeoQA/PGPS9K 多数据集评测出发，发现直接追加结构事实并不稳定，甚至会破坏模型原本正确的视觉判断。随后设计并训练基于 `Qwen/Qwen3-1.7B` LoRA 的 theorem-route generator，将图形解析压缩为紧凑的定理级行动，再在 Qwen3-VL、Qwen2.5-VL、InternVL 等下游模型上验证其效果。实验表明，compact route 对小模型有明显的推理时规划辅助作用，但 route 仍存在 win/loss 双刃剑效应。因此进一步提出 selective exposure 思路：先用轻量 trust policy 判断 route 是否值得暴露；不确定时降级为 theorem-equation candidates；低置信时回退原图。围绕该框架，我完成了 route/candidate/strict-gate/cascade 的对比实验、显著性检验、跨模型分析，并启动 target-binding verifier 训练，探索如何自动判断模型何时应相信中间推理结果。
