# Frozen C1 100k · MultiScale-Sem 与 MultiScale-BCD

两组使用同一 CEM-MitoLab 裁块序列、seed 20261014、batch 4、40k 步和匹配 AdamW 日程。B 的边界与距离损失权重均为 1，距离背景权重为 0.05。B 的分水岭参数依据预先固定的 CEM 验证样本与 collection 等权 PQ 锁定，MB-1 与 MV3 仅应用锁定值。

| ARM | GPU | 100 步前 20/末 20 损失中位数 | 40k 秒 | 峰值分配 GiB | 峰值保留 GiB | 编码器不变 |
|---|---:|---|---:|---:|---:|---:|
| arm_a_multiscale_sem | 1 | 1.3928 → 1.0365 | 1400.6 | 1.68 | 1.80 | True |
| arm_b_multiscale_bcd | 2 | 3.1788 → 2.4663 | 1768.5 | 1.69 | 1.85 | True |

BCD 五张训练目标的索引见冒烟回执。锁定参数 (tau_seed, tau_bnd, min_size) = [0.4, 0.65, 32]，CEM 验证搜索使用 175 个视野。实例分割在核心掩膜中执行，语义评分使用独立的 fg > 0.5。

## MB-1 七组等权

| 系统 | Dice | IoU | r3 内部漏分率 | r5 封闭孔占比 | r3 16px 网格对齐 | 整颗漏检 |
|---|---:|---:|---:|---:|---:|---:|
| C1 Primus (Stride-120) | 0.74820 | 0.64788 | 10.96% | 6.74% | 27.68% | 22.38% |
| 方案 A · Stride-120 | 0.76927 | 0.66977 | 7.17% | 1.35% | 19.19% | 22.24% |
| 方案 B · Stride-120 | 0.76721 | 0.66484 | 6.61% | 1.80% | 17.75% | 23.17% |
| MitoNet-v1 (3D+2D) | 0.74112 | 0.66070 | 14.11% | 0.91% | 17.29% | 23.00% |
| 方案 A · Baseline-128 | 0.76848 | 0.67049 | 7.70% | 1.44% | 20.48% | 22.24% |
| 方案 B · Baseline-128 | 0.76879 | 0.66855 | 6.96% | 1.78% | 20.93% | 22.97% |

## MB-1 逐组 r3 内部漏分率

| 来源 | 旧 Primus | A | B | MitoNet 3D+2D |
|---|---:|---:|---:|---:|
| TEM-100 | 8.64% | 5.48% | 5.42% | 5.89% |
| c_elegans | 14.19% | 2.74% | 3.08% | 21.71% |
| fly_brain | 3.52% | 1.21% | 1.01% | 0.50% |
| glycolytic_muscle | 2.16% | 7.07% | 6.34% | 0.89% |
| hela_cell | 8.16% | 3.18% | 3.60% | 3.09% |
| lucchi_pp | 1.61% | 0.89% | 0.72% | 2.25% |
| salivary_gland | 38.44% | 29.60% | 26.08% | 64.42% |

## CEM 与 MB-1 2D 实例指标

| 数据 | 模型 | 后处理 | 推理 | AP50 | AP75 | PQ | MAE | 接触对分开/总计 |
|---|---|---|---|---:|---:|---:|---:|---:|
| cem_validation | arm_a_multiscale_sem | cc | stride120 | 0.46579 | 0.36799 | 0.46207 | 2.22676 | 34/179 |
| mb1 | arm_a_multiscale_sem | cc | stride120 | 0.44721 | 0.34137 | 0.48417 | 4.86152 | 632/3779 |
| cem_validation | arm_a_multiscale_sem | cc | baseline128 | 0.46798 | 0.37455 | 0.46511 | 2.41859 | 23/179 |
| mb1 | arm_a_multiscale_sem | cc | baseline128 | 0.44575 | 0.34097 | 0.47881 | 5.08356 | 631/3779 |
| cem_validation | arm_b_multiscale_bcd | cc | stride120 | 0.49070 | 0.39214 | 0.48682 | 1.86593 | 43/179 |
| cem_validation | arm_b_multiscale_bcd | watershed | stride120 | 0.35718 | 0.25838 | 0.34459 | 6.85819 | 142/179 |
| mb1 | arm_b_multiscale_bcd | cc | stride120 | 0.44800 | 0.33566 | 0.48290 | 4.55434 | 689/3779 |
| mb1 | arm_b_multiscale_bcd | watershed | stride120 | 0.18177 | 0.04744 | 0.18660 | 27.53350 | 1520/3779 |
| cem_validation | arm_b_multiscale_bcd | cc | baseline128 | 0.49737 | 0.39817 | 0.49233 | 2.13865 | 30/179 |
| cem_validation | arm_b_multiscale_bcd | watershed | baseline128 | 0.36180 | 0.26309 | 0.34913 | 6.22597 | 149/179 |
| mb1 | arm_b_multiscale_bcd | cc | baseline128 | 0.44802 | 0.33662 | 0.47887 | 4.90533 | 649/3779 |
| mb1 | arm_b_multiscale_bcd | watershed | baseline128 | 0.18071 | 0.04073 | 0.18328 | 24.24282 | 1558/3779 |
| mb1 | Old Primus CC | cc | legacy128 | 0.30954 | 0.23100 | 0.36146 | 20.04477 | 1236/3779 |
| mb1 | Old center-offset head | head | legacy128 | 0.22920 | 0.14683 | 0.26873 | 58.87245 | 1847/3779 |
| cem_validation | Old Primus CC | cc | legacy128 | 0.46790 | 0.39121 | 0.46853 | 4.79297 | 51/179 |
| cem_validation | Old center-offset head | head | legacy128 | 0.43471 | 0.34790 | 0.42840 | 13.82910 | 157/179 |

旧 Primus 与旧中心偏移头采用历史 256/128 推理。A、B 提供 128 与 120 两种窗口结果，CC 对照直接作用于相同语义掩膜。CEM 验证曾用于 B 的分水岭选择。MB-1 已参与此前开发诊断。

## HQ-5nm 无标注 20 FOV

| 模型 | 推理 | 封闭空洞面积占比 | 方孔数 |
|---|---|---:|---:|
| arm_a_multiscale_sem | stride120 | 0.15% | 0 |
| arm_a_multiscale_sem | baseline128 | 0.19% | 0 |
| arm_b_multiscale_bcd | stride120 | 0.15% | 0 |
| arm_b_multiscale_bcd | baseline128 | 0.32% | 0 |

[逐组语义/内部漏洞](semantic_and_cavity.csv) · [实例对照](instance_metrics.csv) · [HQ 逐图](hq_per_image.csv)。MB-1 有 GT 的复核图在 visual_panels，HQ 输出为预测掩膜。

## 本轮结果裁决

- A 相对旧 Primus 的 MB-1 Dice 变化 +0.02107，r3 去边界内部漏分率变化 -3.79 个百分点。
- B 相对旧 Primus 的 MB-1 Dice 变化 +0.01901，r3 去边界内部漏分率变化 -4.36 个百分点。
- B 在 cem_validation 的分水岭 PQ 0.34459，同一语义前景的 CC PQ 0.48682。
- B 在 mb1 的分水岭 PQ 0.18660，同一语义前景的 CC PQ 0.48290。

HQ 的方孔形态统计无 GT；MB-1 的内部漏分和整颗漏检仍需分开判读。试验仅包含一个 decoder seed。
