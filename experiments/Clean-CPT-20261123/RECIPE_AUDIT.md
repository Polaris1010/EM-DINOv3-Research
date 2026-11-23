# Clean-CPT 最终配方与数据核查

Clean-C0 与 Clean-C1-0k 从官方自然权重起步，Clean-C1-30k 从 Clean-C0 的第 30,000 次更新恢复。物理 GPU 1、2、3 各处理 16 个样本，每次更新合计 48 个样本。历史配置与产物只读复用。

## D2 配方

| 项目 | 设置 |
|---|---|
| 模型 | DINOv3 ViT-L/16，官方 P0，精确 SHA-256 见下方 |
| 采样 | MitoVerse : CEM1.5M 为 50:50。前者按来源及该来源内的采集键均匀选择，后者按 collection 图像数平方根重加权，集合内均匀有放回 |
| 增强 | 原 D2 增强与 iBOT 掩码，global 256，8 个 local 112，SSL seed 0 |
| 优化 | AdamW，峰值 LR 2.5980762e-5，WD 0.04，betas 0.9 / 0.99，裁剪 30，layer-wise 衰减 0.98，patch embedding LR 倍率 0.2 |
| 历史日程 | 前 5k 线性 warmup，前 30k 段沿用 D2 的 30k cosine horizon，30k 起采用历史 100k 续训日程。最低 LR 1e-6 |
| 自监督 | DINO + iBOT + 0.1 KoLeo，teacher EMA 0.999，Sinkhorn-Knopp，teacher 温度 0.04→0.07（5k），BF16 autocast |

Layer 23 的空间 patch tokens 移除 CLS 和 registers，FP32 L2 归一化后求相似度矩阵。选择 Chebyshev 距离大于 2 的有序 patch 对，再选 anchor 相似度不超过候选中位数的对（含并列）。Barrier 取所选对的 `relu(student_affinity - anchor_affinity - 0.05)^2` 的平均值。Anchor 为同视图、无 mask、eval、no-grad，保留训练随机数状态。正式晚起点分支直接复用历史生产适配器，权重为 1。早起点分支固定 P0 anchor，权重从 0 至 5k 线性升到 1，检验同一次 backward 的实际系数。

## 最终清单

- MitoVerse 保留 119/153。排除 CellMap 六个官方测试采集对应 34 个体积，官方清单含 16 个 test crops。MV3 已退役，原 D2 合法 Z 范围照常使用。
- CEM1.5M 保留 597/627，1,390,869 张。CEM-MitoLab val 排除 30 个 collection、34,564 张。保留 432 个 NCI 内部 collection、427,093 张，以及两个先前宽泛排除的 collection。
- 已登记 MB-1 候选、关联果蝇采集、Lucchi、MitoEM 2.0 val/test 和 CellMap 测试采集的 ID 交集为零。MB-1 七组另进行过 dHash 抽样，组织图像无确认重复。NCI 来源使用作者的 MB-1 排除声明及机构来源记录。HQ-5nm 为独立采集，现有元数据缺少统一跨库采集 ID。声明以现有标识和抽样为证据范围。
- 双机清单 SHA-256 `f4eeb85ffa4868a58f2688af638433095a0ecf706ca19a0fa5013eee946ddb7f`。双机 P0 SHA-256 `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`。

正式启动依赖双机 200 步门禁回执。30k 全状态发布后，晚起点队列校验全状态与 EMA 权重的 SHA-256，再恢复优化器、训练日程、模型和各 rank RNG。

[MitoNet 原论文](https://www.cell.com/article/S240547122200494X/pdf) · [CEM 来源补充表](https://ars.els-cdn.com/content/image/1-s2.0-S240547122200494X-mmc3.xlsx)
