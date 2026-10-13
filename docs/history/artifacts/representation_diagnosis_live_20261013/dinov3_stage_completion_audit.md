# B_full 的 Gram / HRA 阶段完成核验

日期：2026-10-13。本轮读取正式配置、resolved YAML、trainer / sampler、三份主 checkpoint metadata，并完整扫描两轮训练的六份 rank 日志。

## 1. 已执行阶段

**B100k 与 B150k 均完成 vanilla EM 持续 SSL；这两轮未执行 Gram anchoring，也未执行独立 HRA。** 历史 P1 的 Gram 探索不计为 B_full 阶段。

| 项目 | B100k | B150k |
|---|---|---|
| 初始化 | 官方自然 ViT-L/16 | B100k full checkpoint 恢复续训 |
| optimizer updates | 1–100000 | 100001–150000 |
| 数据 | 全 MitoVerse：212 volumes、13 sources、48,722 slices | 同一 formal manifest |
| DINO / iBOT / KoLeo | 1 / 1 / .1 | 相同 |
| gram.use_loss | false | false |
| gram.ckpt | null | null |
| global / local / Gram teacher crop | 256 / 112 / null | 256 / 112 / null |
| local views | 8 | 8 |
| global batch | 48（3 GPU） | 48（3 GPU） |
| 学习率阶段 | 原 100k schedule | 固定原 min_lr=1e-6 继续 50k |
| 独立高分辨率 SSL | 未在正式任务中执行 | 未在正式任务中执行 |

源配置 SHA256：`159f45566320f9b583968aa8e9ffb07392d521bfa227b1ca0b4a4bde11f152d7`。正式 resolved YAML：`f1acbd49af827b80a457b64e28cc4dd912524c08c0676256f505cba59a5825e1`。150k 原样复制源配置；本轮 YAML 扫描所得 Gram / crop 值一致，详见 [supplement.json](evidence/supplement.json)。

原 trainer 和续训 trainer 的 check_config 均拒绝 gram.use_loss=True；sampler 使用 gram_teacher_crops_size=None。保存的 100k / 150k teacher metadata 对应相同 config hash、官方 weight hash 与 formal manifest hash。512×512 原始窗口读取随后形成 256 global views，不能把窗口读取大小记作 512 输入 HRA。

## 2. 完整日志扫描

| 正式任务 | rank 数 | 每 rank 条数 | 实际步数 |
|---|---:|---:|---|
| B100k | 3 | 100000 | 1–100000 |
| B150k extension | 3 | 50000 | 100001–150000 |

扫描共 450,000 条 rank 记录。loss_components 仅出现 dino_global_crops_loss、dino_local_crops_loss、ibot_loss、koleo_loss；未出现 Gram 字段，未发现非零 Gram loss 记录。每份文件的 SHA、条数及扫描结果保存在 [live_evidence.json](evidence/live_evidence.json)。

未记录 Gram 字段不等于测量到 Gram=0；本报告以 **false 配置 + runtime 拒绝条件 + 无 Gram checkpoint 引用 + 实际 loss 记录** 联合确认未启用。正式任务中没有 HRA 的多 global size 配置或独立阶段记录。此结论限于本次核验的 B_full 正式任务与续训目录，不声称服务器所有历史项目从未做过 Gram。

## 3. 真实 checkpoint inventory

以下 teacher_backbone 文件本轮均确认存在、计算 SHA256，并 mmap 读取 step / manifest / config / official weight metadata：[checkpoint_inventory_verified.json](evidence/checkpoint_inventory_verified.json)。

| 阶段 | 可用 encoder checkpoint steps |
|---|---|
| B_full 初次训练 | 1000、5000、10000、25000、50000、75000、100000 |
| B_full 150k extension | 100010、105000、125000、150000 |
| 30k / 60k | 正式 checkpoint 目录未发现，不能假设存在 |

每份 teacher export 大小 1,213,025,477 bytes；其 step 和 manifest 均与文件目录一致。D3 可以优先选 10k / 25k / 50k / 75k / 100k / 125k / 150k，无需要求缺失的 30k / 60k。

| 主权重 | SHA256 |
|---|---|
| 官方自然 P0 | `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035` |
| B100k teacher | `9603877244ba40f3cbeccdea18d69b0d3b818dc0fc6b09f839fd6d4189ce2de5` |
| B150k teacher | `9c67d08faf508a14d25418014422c0dbd641a51c122161dbd31f8af0a386d2ee` |

所有实际数值以 [checkpoint_metadata.json](smoke/checkpoint_metadata.json) 为准。Teacher hash 与 full optimizer checkpoint hash 不可互换。

## 4. 数据协议和上游能力

本轮取得实际正式 manifest 原件，其 SHA256 为 `3fd614082bf584c21e212884856875e328104ea09a293a95510c0f5f3117edb9`，见 [formal_pretrain_manifest.json](evidence/formal_pretrain_manifest.json)。212 volumes 中 legacy_split 为 train 31、validation 8、legacy_dev_test 16、excluded 157；B_full 的 full-data 训练池纳入这些条目，不能把 legacy_split 名称视为 SSL 隔离机制。

官方自然 ViT-L 权重可能继承其上游教师与蒸馏能力；本项目 EM 训练完成情况必须按这里的真实记录计算。不能因加载官方 DINOv3 就宣称在 EM 上做过 Gram/HRA，也不能把固定输入的高分辨率推理算成高分辨率自监督适应。

## 5. 当前决策

Gram / HRA 新训练保持未启动。100k→150k 没有一致的 Dice / NSD 提升，但这不足以证明 dense feature degradation。先做匹配的 readout 多种子、Linear Probe、目标相关密集特征与真实尺度诊断。

如果 D3 提供退化证据，再比较同起点、同数据与预算的 vanilla C10 / Gram G10，并在开发数据中锁定 reference 选择规则；如果 D4 支持 HR，再加入 vanilla HR 与 Gram HR，对照输入尺度和计算预算。B_clean 继续遵守现有数据治理 NO_GO，不能用这次取得 manifest 原件代替 acquisition / annotation 证据。
