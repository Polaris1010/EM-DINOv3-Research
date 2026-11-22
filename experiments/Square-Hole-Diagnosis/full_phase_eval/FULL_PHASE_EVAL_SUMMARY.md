# Full Patch-Phase-Decoupled Inference

状态 **完整**。物理 GPU 1。编码器 `/opt/data/3_117/tanghr/codex/pilot_c100_paired_20261028/runs/C1_Barrier_100k/checkpoints/step000100000/teacher_backbone.pth`，解码器 `/opt/data/3_117/tanghr/codex/mix_mitolab_mv_20261118/batch4/primus_cem/Barrier100/model_40000.pt`。SHA256 见 [protocol.json](protocol.json)。
推理窗口 256、阈值 >0.5、reflect 补边。原图边缘从每相位的最少覆盖窗口取预测。每个视野先计算 Baseline 并核验，再依次计算三个候选；同相位前向结果复用。表中秒数是各配置所用 pass 之和，可相互重叠，包含 tile 组装与融合。拓扑统计另外计算。全程墙钟耗时 3026.8 秒（未完成时为 0）。

## 步骤 0 · 前次 Shift-8 掉点

旧脚本使用 127.5 灰度常数补边，归一化后为 0。Reflect 是本次统一的新补边规则。掉点由四张图共同造成，Fly brain 是最大项。现有对照不能将旧掉点单独归因于补边。

| 视野 | 基线 Dice | Shift-8 Dice | 差值 |
|---|---:|---:|---:|
| MB1_TEM_7Z7jSCJZ23zSW2ZztTPJ_z0 | 0.84214 | 0.82020 | -0.02195 |
| MB1_fly_brain_z85 | 0.89662 | 0.73014 | -0.16648 |
| MB1_TEM_j1k88xV7tmKNMLM4bj1h_z0 | 0.71955 | 0.65912 | -0.06044 |
| MB1_TEM_cwpw4Zu4rqVzbTMPhql2_z0 | 0.95765 | 0.94963 | -0.00802 |

## HQ-5nm · 共同 20 FOV

| 配置 | FOV | 空洞面积占比 | 方孔总数 / 每张 | 封闭孔总数 | 窗口数 / 倍率 | 秒 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 20 | 2.995% | 139 / 6.95 | 15621 | 4500 / 1.000× | 31.0 |
| Gauss + Stride-120 | 20 | 1.734% | 55 / 2.75 | 10727 | 5120 / 1.138× | 33.0 |
| Gauss + Dual-Phase | 20 | 1.678% | 28 / 1.40 | 10894 | 9620 / 2.138× | 64.1 |
| Gauss + Quad-Phase | 20 | 1.150% | 22 / 1.10 | 8013 | 19220 / 4.271× | 125.7 |

## MB-1 汇总

Quad-Phase 全量估算 3134 秒。超过 900 秒时四相位仅应用于 TEM-100 和 Glycolytic muscle，倍率用同子集 Baseline 作分母。

Baseline 已核对 2595 planes，其间 257 个具有 reflect 补边的样本与旧四项混淆计数不同。无需补边的样本要求严格一致。

| 配置 | 范围 | N | Dice | IoU | Precision | Recall | 空洞占比 | 方孔 | 窗口数 / 倍率 | 秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
相同已完成视野按组等权的历史 Baseline Dice 为 0.74527，本次为 0.74537；历史 IoU 为 0.64629，本次为 0.64641。

| Baseline | ALL | 2595 | 0.74537 | 0.64641 | 0.83793 | 0.70490 | 0.407% | 49 | 104945 / 1.000× | 768.6 |
| Gauss + Stride-120 | ALL | 2595 | 0.74820 | 0.64788 | 0.84389 | 0.70799 | 0.371% | 39 | 110700 / 1.055× | 774.9 |
| Gauss + Dual-Phase | ALL | 2595 | 0.74906 | 0.64991 | 0.86526 | 0.69668 | 0.375% | 24 | 240834 / 2.295× | 1689.8 |
| Gauss + Quad-Phase | TEM-100+glycolytic_muscle | 402 | 0.89235 | 0.81286 | 0.93008 | 0.86430 | 0.305% | 2 | 37961 / 4.942× | 250.1 |

### MB-1 逐组

| 子组 | 配置 | N | Dice | IoU | Precision | Recall | 方孔 |
|---|---|---:|---:|---:|---:|---:|---:|
| TEM-100 | Baseline | 100 | 0.86063 | 0.76947 | 0.89543 | 0.84037 | 42 |
| TEM-100 | Gauss + Stride-120 | 100 | 0.86135 | 0.77059 | 0.89435 | 0.84454 | 31 |
| TEM-100 | Gauss + Dual-Phase | 100 | 0.87180 | 0.78590 | 0.90580 | 0.85247 | 19 |
| TEM-100 | Gauss + Quad-Phase | 100 | 0.87518 | 0.79115 | 0.90865 | 0.85644 | 2 |
| c_elegans | Baseline | 256 | 0.65892 | 0.51042 | 0.69093 | 0.64181 | 0 |
| c_elegans | Gauss + Stride-120 | 256 | 0.65892 | 0.51042 | 0.69093 | 0.64181 | 0 |
| c_elegans | Gauss + Dual-Phase | 256 | 0.66961 | 0.52031 | 0.71911 | 0.63629 | 0 |
| fly_brain | Baseline | 256 | 0.88650 | 0.80198 | 0.94961 | 0.83776 | 0 |
| fly_brain | Gauss + Stride-120 | 256 | 0.88650 | 0.80198 | 0.94961 | 0.83776 | 0 |
| fly_brain | Gauss + Dual-Phase | 256 | 0.88401 | 0.79892 | 0.96068 | 0.82613 | 0 |
| glycolytic_muscle | Baseline | 302 | 0.89636 | 0.81285 | 0.92822 | 0.86785 | 0 |
| glycolytic_muscle | Gauss + Stride-120 | 302 | 0.88976 | 0.80233 | 0.93334 | 0.85141 | 0 |
| glycolytic_muscle | Gauss + Dual-Phase | 302 | 0.90514 | 0.82728 | 0.94824 | 0.86693 | 0 |
| glycolytic_muscle | Gauss + Quad-Phase | 302 | 0.90951 | 0.83458 | 0.95152 | 0.87215 | 0 |
| hela_cell | Baseline | 256 | 0.77123 | 0.63689 | 0.86403 | 0.70317 | 0 |
| hela_cell | Gauss + Stride-120 | 256 | 0.77123 | 0.63689 | 0.86403 | 0.70317 | 0 |
| hela_cell | Gauss + Dual-Phase | 256 | 0.73269 | 0.58960 | 0.88594 | 0.63284 | 0 |
| lucchi_pp | Baseline | 165 | 0.91467 | 0.84398 | 0.93339 | 0.89966 | 0 |
| lucchi_pp | Gauss + Stride-120 | 165 | 0.91620 | 0.84663 | 0.92559 | 0.90980 | 1 |
| lucchi_pp | Gauss + Dual-Phase | 165 | 0.91778 | 0.84953 | 0.93633 | 0.90245 | 0 |
| salivary_gland | Baseline | 1260 | 0.22929 | 0.14927 | 0.60391 | 0.14366 | 7 |
| salivary_gland | Gauss + Stride-120 | 1260 | 0.25344 | 0.16631 | 0.64941 | 0.16748 | 7 |
| salivary_gland | Gauss + Dual-Phase | 1260 | 0.26236 | 0.17782 | 0.70072 | 0.15964 | 5 |

MB-1 各组先对切片或图像算均值，再对数据组等权。拓扑指标描述预测形状，Dice/IoU 使用人工真值。

[HQ CSV](hq5nm_summary.csv) · [MB-1 汇总 CSV](mb1_summary.csv) · [MB-1 逐组 CSV](mb1_by_group.csv) · [逐图 CSV](per_image.csv)
