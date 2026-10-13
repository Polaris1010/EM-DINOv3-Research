# MitoVerse：MitoNet 实测与修复版 P1 对比

2026-09-18 的阶段快照。验证集已完整覆盖 8 个原始体积、2 个来源；开发测试集仍在计算，seed2 尚未完成。

| 模型 | 全体积 validation 源等权 Dice |
|---|---:|
| MitoNet_v1 固定 M0 | 0.476611 |
| 修复 mask 后 P1 seed0 | 0.071986 |
| 修复 mask 后 P1 seed1 | 0.063903 |
| P1 已完成两个 seed 平均 | 0.067945 |
| MitoNet − P1 两 seed 平均 | +0.408666 |

| 来源 | MitoNet | P1 两 seed 平均 |
|---|---:|---:|
| guay21 | 0.019930 | 0.034780 |
| mekuc20 | 0.933292 | 0.101109 |

主要差距集中在 UroCell / mekuc20。Guay 两者均低分，P1 略高。本结果不支持“已接近 MitoNet”，也不支持将差距全归因于 Gram。

## 口径与证据边界

- 相同 validation 体积，全切片、原始网格、阈值 0.5；先对来源内体积等权，再对来源等权。
- MitoNet 使用官方未量化 TorchScript 权重，官方 mean/std 归一化；原始物理网格 input_scale=1，固定 512 tile / 256 stride（小图单窗口），不搜索尺度或阈值。这是明确实现的 2D M0，不等于官方完整 3D 实例推理基准。
- P1 为修复 mask 后重新训练的冻结编码器分割 probe；已完成 seed0/1 各 10k 更新。checkpoint 由预定验证规则选择。seed2 未计入均值，不能称为三 seed 结果。
- MitoNet 没有在本轮目标数据微调；两种模型预训练历史暴露未全部厘清，不能宣称严格未见域或模型架构的因果优势。
- 语义输出与本次 2D 实例后处理前景输出在这 8 个体积上的混淆计数一致；不是两次独立实验，也不代表 3D 实例质量相同。
- 本地复算 CSV 的每体积 TP/FP/FN/TN、Dice 和来源汇总均通过。完整测试结果尚不能报告。

[官方模型配置](https://raw.githubusercontent.com/volume-em/empanada-napari/main/empanada_napari/configs/MitoNet_v1.yaml)。实测证据：mitonet_comparison_volume.csv、mitonet_comparison_summary.json、mitonet_comparison_protocol.json。

## 正在执行

GPU1/3 分担 MitoNet 全体积推理，GPU2 继续 seed2。当前统一协调器完成后，已登记队列将使用 GPU1/2/3 分别评估修复版 P1 seed0/1/2 的完整 legacy_dev_test，完成新的匹配比较；中途资源保护停止则不自动重启。
