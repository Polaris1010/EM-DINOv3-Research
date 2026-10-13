# 验证滑窗批量化

P0预算pilot使用原始逐窗口版本（mv_probe_pilot_v1.py）；正式三模型使用同一批量化版本，每批16个256×256窗口。监督训练batch仍为1。

仅改变滑窗提交GPU的批量：stride128、原生XY、同一灰度归一化、encoder BF16、decoder FP32、mean softmax融合、0.5阈值与无TTA不变。官方Primus源码未改。LayerNorm不依赖batch统计，推理处于no_grad。

新脚本SHA记录在正式每组run_protocol/metrics的probe_source_sha256中。保留原pilot脚本以复查实际运行版本，不将已运行pilot描述为采用新批量化版本。

主协议batch_size为监督训练batch。验证tile batch是吞吐设置，不改变物理视野或评分网格。单seed完整结果和实际显存审计必须通过后才补其余seeds。
