# MitoNet 补强准备（本轮不阻塞P0-P2）

已核验可复用的权重：`/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/official/MitoNet_v1.pth`。
上轮单轴网络与2D后处理实现：`/opt/data/3_117/tanghr/codex/mitotopo_20260909_round3/evaluate.py`；尺度敏感性检查：同目录`mitonet_scale.py`。

- M0：在官方模型实际配置/文档确认推荐物理尺度后推理；不能把原始像素网格自动称为推荐物理尺度。
- M1：仅用round4 validation来源选尺度/阈值，再冻结。
- M2：核对每volume原始ZYX spacing后，分别XYZ推理，将概率恢复原始网格再融合；融合权重只由validation选择。
- M3：先验证官方代码是否提供可训练权重与训练入口；TorchScript推理导出不等于完整微调训练包。若可行，使用相同31个监督训练体积、同等预算，并保留公平性差异。

目前只完成复用入口与实验约束准备，未运行M0-M3，不报告它们的MitoVerse分数。实例指标需单独的官方实例推理/拼接与实例真值评测流程。
