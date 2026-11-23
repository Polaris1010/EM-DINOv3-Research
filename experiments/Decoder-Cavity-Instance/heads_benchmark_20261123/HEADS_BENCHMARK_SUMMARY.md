# 冻结 C1 编码器 · 三组实例头与曝光量对照

三组新实验中，Head D · MultiScale-Cellpose 的 MB-1 原生实例 PQ 最高，相对 B 边界分水岭变化 +0.00766。B-Batch32 相对 B 的 MB-1 PQ 变化 -0.03757，Dice 变化 -0.05492。Batch32 同时改变了优化 batch、学习率和实际更新预算，效果用于工程全方案比较。

## 训练与后处理

统一使用冻结 C1、四层 ViT 特征、CEM-MitoLab 256² 裁块、seed 20261014。三组 100 步目标图、loss 下降、冻结状态哈希和显存余量复核通过。C 的独立 ASPP 语义/实例分支与质心投票遵循 Panoptic-DeepLab。D 的热扩散流和 Euler 汇点分组遵循 Cellpose 的机制。 [Panoptic-DeepLab](https://openaccess.thecvf.com/content_CVPR_2020/html/Cheng_Panoptic-DeepLab_A_Simple_Strong_and_Fast_Baseline_for_Bottom-Up_Panoptic_CVPR_2020_paper.html) · [Cellpose 实现](https://github.com/MouseLand/cellpose/blob/main/cellpose/dynamics.py)

Batch32 的冒烟稳态预测 40k 需 2.84 小时，阈值为 2.50 小时。按实际训练清单中 20,447 个图像项换算曝光等价 epoch。训练仍采用集合均衡的随机裁块。时长为实测冒烟、40k 或 20k 训练循环以及评测之和。

| 模型 | 物理 GPU | batch | 更新 | 裁块曝光 | 等价 epochs | 峰值保留 GiB | 合计 min | 正式训练 min | 评测 min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Head C · Panoptic-DeepLab | 1 | 4 | 40,000 | 160,000 | 7.83 | 2.08 | 57.6 | 44.0 | 13.5 |
| Head D · MultiScale-Cellpose | 2 | 4 | 40,000 | 160,000 | 7.83 | 1.84 | 48.7 | 34.5 | 14.1 |
| B-Batch32 | 3 | 32 | 20,000 | 640,000 | 31.30 | 5.58 | 96.6 | 84.5 | 11.6 |

每组后处理在相同 175 个 CEM 验证视野上锁定，再评完整 CEM 和 MB-1。下表控制项依次是 C 的中心置信度、D 的 Euler 迭代次数、B-Batch32 的边界置信度。

| 模型 | 前景阈值 | 控制项 | 最小面积 | CEM 调参子集 PQ | 权重 SHA-256 前缀 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Head C · Panoptic-DeepLab | 0.55 | 0.3 | 32.0 | 0.56489 | 4578bc799114ed0b |
| Head D · MultiScale-Cellpose | 0.55 | 200.0 | 32.0 | 0.52547 | 4457bd61b6376f6d |
| B-Batch32 | 0.5 | 0.35 | 16.0 | 0.57453 | 9f0d3a0904dc50d8 |

## 全模型对比

| 模型 | MB-1 Dice | r=3 内部漏分 | r=5 内部漏分 | MB-1 PQ | MB-1 AP50 | MB-1 MAE | CEM PQ | CEM MAE | HQ-5nm 方孔 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 Primus | 0.74820 | 10.96% | 9.63% | 0.35980 | 0.30473 | 18.41 | 0.35471 | 6.46 | 55 |
| A · MultiScale-Sem | 0.76927 | 7.17% | 5.55% | 0.48417 | 0.44721 | 4.86 | 0.46207 | 2.23 | 0 |
| B · 边界分水岭 | 0.76721 | 6.61% | 5.13% | 0.52316 | 0.49800 | 4.04 | 0.55446 | 1.57 | 0 |
| Head C · Panoptic-DeepLab | 0.77391 | 6.91% | 5.63% | 0.52564 | 0.50965 | 3.93 | 0.56723 | 1.35 | 0 |
| Head D · MultiScale-Cellpose | 0.78322 | 5.98% | 4.76% | 0.53082 | 0.50066 | 6.46 | 0.52412 | 2.33 | 1 |
| B-Batch32 | 0.71230 | 10.48% | 9.02% | 0.48559 | 0.46946 | 4.66 | 0.57908 | 1.46 | 0 |
| MitoNet-v1 · 官方 2D | 0.71215 | 9.84% | 8.84% | 0.46658 | 0.44375 | 6.77 | 0.76096 | 1.92 | 1 |

语义和空洞按固定 0.50 前景阈值计分，实例按各自锁定的后处理参数计分。C1 Primus 与 A 的实例采用连通域。所有 DINOv3 解码头采用 256 高斯窗口、stride 120、reflect padding。MitoNet-v1 采用发布权重的原生 Engine2d。旧 MB-1 缓存的权重哈希与本轮官方权重不一致，本轮因此用同一 Engine2d 重新推理 MB-1、CEM 和 HQ-5nm。MitoNet 在 CEM 的成绩按发布系统口径解读，其监督划分由发布协议确定。HQ-5nm 无 GT，方孔是预测掩膜形态统计。

## 新模型的二维实例对照

| 模型 | 数据 | 后处理 | AP50 | AP75 | PQ | 实例数 MAE | 接触对分开/总计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Head C · Panoptic-DeepLab | cem_validation | 原生 | 0.59891 | 0.46016 | 0.56723 | 1.35 | 120/179 |
| Head C · Panoptic-DeepLab | cem_validation | 连通域 0.50 | 0.48643 | 0.39055 | 0.47937 | 1.94 | 34/179 |
| Head C · Panoptic-DeepLab | cem_validation | 同阈值连通域 | 0.49180 | 0.39289 | 0.48340 | 2.07 | 34/179 |
| Head C · Panoptic-DeepLab | mb1 | 原生 | 0.50965 | 0.37448 | 0.52564 | 3.93 | 953/3779 |
| Head C · Panoptic-DeepLab | mb1 | 连通域 0.50 | 0.46232 | 0.35601 | 0.49645 | 4.80 | 499/3779 |
| Head C · Panoptic-DeepLab | mb1 | 同阈值连通域 | 0.46293 | 0.35294 | 0.49435 | 4.88 | 509/3779 |
| Head D · MultiScale-Cellpose | cem_validation | 原生 | 0.54676 | 0.42780 | 0.52412 | 2.33 | 141/179 |
| Head D · MultiScale-Cellpose | cem_validation | 连通域 0.50 | 0.47158 | 0.37037 | 0.46682 | 2.09 | 74/179 |
| Head D · MultiScale-Cellpose | cem_validation | 同阈值连通域 | 0.48051 | 0.37767 | 0.47338 | 2.04 | 83/179 |
| Head D · MultiScale-Cellpose | mb1 | 原生 | 0.50066 | 0.39701 | 0.53082 | 6.46 | 1316/3779 |
| Head D · MultiScale-Cellpose | mb1 | 连通域 0.50 | 0.47620 | 0.37513 | 0.51302 | 4.56 | 953/3779 |
| Head D · MultiScale-Cellpose | mb1 | 同阈值连通域 | 0.47833 | 0.37580 | 0.51336 | 4.45 | 950/3779 |
| B-Batch32 | cem_validation | 原生 | 0.59425 | 0.48453 | 0.57908 | 1.46 | 100/179 |
| B-Batch32 | cem_validation | 连通域 0.50 | 0.54245 | 0.43533 | 0.53426 | 1.69 | 86/179 |
| B-Batch32 | cem_validation | 同阈值连通域 | 0.54245 | 0.43533 | 0.53426 | 1.69 | 86/179 |
| B-Batch32 | mb1 | 原生 | 0.46946 | 0.34947 | 0.48559 | 4.66 | 736/3779 |
| B-Batch32 | mb1 | 连通域 0.50 | 0.43339 | 0.31883 | 0.45663 | 4.73 | 661/3779 |
| B-Batch32 | mb1 | 同阈值连通域 | 0.43339 | 0.31883 | 0.45663 | 4.73 | 661/3779 |

每个候选完整覆盖 CEM 1,424、MB-1 2,595、HQ-5nm 20 个视野。三组及 MitoNet 的评测 ID 逐项配对。语义、实例和空洞指标按组内汇总再按组等权，接触对汇报总数。逐组、封闭孔占比、网格对齐率、整颗漏检率与单项调参结果见各组 CSV。
