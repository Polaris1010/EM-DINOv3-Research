# 本项目 DINOv3 阶段完成审计 · 2026-10-13

**B100k 和 B150k 完成的是 vanilla EM 持续 SSL；没有在这两轮中执行 Gram anchoring 或独立 HRA。** 当前判断有正式配置、限制条件、续训代码和完成记录支持；原始完整 resolved YAML、逐步日志和 checkpoint 二进制尚未在本轮从服务器重新读取，不能将历史证据说成实时检查。

## 配置、代码与完成证据

| 项目 | B100k | B150k |
|---|---|---|
| 初始化 | 官方自然 ViT-L/16，step 0 | B100k full optimizer checkpoint，恢复 student / teacher / optimizer / loss states / RNG |
| 正式训练 | 100,000 updates | 从 100,000 续训至 150,000 |
| 数据 | 全 MitoVerse 212 volumes / 13 sources / 48,722 legal slices | 同一 manifest |
| SSL objective | DINO=1、iBOT=1、KoLeo=.1 | 相同 |
| gram.use_loss | false | 相同配置 + 相同拒绝 Gram 的 check_config |
| gram.ckpt | null | 配置按原样复制，没有新增 Gram checkpoint |
| global / local / Gram teacher crop | 256 / 112 / null；8 local crops | 相同 |
| lr | 5k warmup、100k cosine | 原 min_lr=1e-6 固定续训；不是从头 150k cosine |
| 独立 HRA | 未在该任务中执行 | 未在该任务中执行 |

原件 `../stage1_B_full.yaml` 的 SHA256 为 `159f45566320f9b583968aa8e9ffb07392d521bfa227b1ca0b4a4bde11f152d7`，与 150k completion/admission 的 config hash 完全相同。`../stage1_train_B_full.py` 的 SHA256 为 `dfdd9db6c5bbfc2456e3eba19224cdf3eea305823743caadb00eeec309c2244c`，与续训 builder 对原 trainer 的硬校验一致。

trainer 的 check_config 明确拒绝 gram.use_loss=True；sampler 显式传入 gram_teacher_crops_size=None。150k builder 原样复制配置及 sampler，保留该检查，只改续训预算、恢复条件、LR 和 checkpoint milestones。150k 实际 completion 记录 trainer=`077e903bba5c22d88a1b210037dda861552a5920fe8a0c9207b58ef7bf5efd32`、sampler=`7c347451788969f5b49acc58d53f103cef0d771deb27b4a012443e9f6613b560`、world_size=3、global_batch=48，并完成 150,000 步。

100k 三 rank 日志的历史 audit 均为 100,000 连续条目，无步数缺口/非有限数。该摘要没有保存完整 loss dictionary，故**本轮非零 Gram loss 全量扫描为 UNVERIFIED**；不能把没有 Gram 字段与 Gram=0 等同。配置与 runtime admission 已足以确认这两轮禁止启用 Gram。

本轮没有发现属于 B_full 的 Gram reference checkpoint 或独立多分辨率 SSL 任务证据。项目旧 P1/Gram 历史目录确实存在，不能据此说整个项目从未尝试 Gram，也不能将其冒认为 B_full 阶段。

sampler 原始读取 512×512 灰度窗口（小图反射 padding），随后生成 256 global / 112 local views。读取 512 原图窗口不等于以 512×512 图像输入执行 HRA；Collate 使用固定 256/16 grid。

## 官方参照及上游知识

官方 DINOv3 源提交固定为 `91ffb6803c74aeb6927209b82560142743763dd7`。捕获的 [官方 HRA 配置](https://github.com/facebookresearch/dinov3/blob/91ffb6803c74aeb6927209b82560142743763dd7/dinov3/configs/train/dinov3_vit7b16_high_res_adapt.yaml) 中 gram.use_loss=true，global views 包括 512 / 768，对应 Gram teacher views 包括 768 / 1152，且按多组 local/global crop 组合采样。它与本项目固定 256 的 B_full 配置明确不同。不要只看 hrft.enabled 字段判断是否为 HRA：该官方文件本身的 hrft.enabled 也是 false。

官方自然权重可能继承上游教师训练与蒸馏的能力；这不能记为本项目在 EM 数据上执行了 Gram/HRA。官方 7B 的阶段步数与本项目 ViT-L 预算也不能直接互换。本报告未使用 Issue #102 的步数作为已核验项目事实。

## 可用 checkpoint

| 步数 | 本轮证据 |
|---:|---|
| 10k | 正式 B_full 代码安排 student/teacher exports；当前文件存在性、hash UNVERIFIED |
| 30k、60k | 不是正式 B_full milestone；未找到保存证据，不能假设可用 |
| 25k、50k、75k | 正式代码安排 exports；当前文件存在性、hash UNVERIFIED |
| 100k | final full checkpoint 与 teacher probe 使用记录确认存在过 |
| 125k | 续训代码安排 export；当前文件存在性、hash UNVERIFIED |
| 150k | 完成记录、teacher probe 使用与 export hash 确认存在过 |

100k full checkpoint SHA256：`12bbf09ffcd7f93cbad88c4e55df55d36b3a299990ad976bf29af51a118a265b`；150k teacher export：`9c67d08faf508a14d25418014422c0dbd641a51c122161dbd31f8af0a386d2ee`。100k teacher export 的独立二进制 hash 未出现在本轮选取的 metrics 中，不能拿 full checkpoint hash 替代它。完整候选表见 [checkpoint_availability.json](checkpoint_availability.json)。

正式训练保留 milestone exports、部分非 milestone full optimizer 文件被原策略清理；不能从保存间隔推断全部 checkpoint 今天仍存在。后续 D3 应先取得实时 inventory，再选 10k/25k/50k/75k/100k/125k/150k 等真正可用时间点，开发集选择 reference，禁止根据最终测试标签挑选。

## 启动判断

Gram/HRA 长训：**NO_GO**。尚缺多 seed 的 readout 波动、目标相关 dense feature 退化证据，以及固定物理 FoV 的真实权重分辨率诊断。干净 B_clean 仍遵守 Stage0 的 NO_GO manifest，未自动启动。仅按证据推进 D1–D4；若 D3 支持，再比较同起点 C10 与 G10；若 D4 支持，再比较同起点 256 续训与 HR，并保留无 Gram HR 对照、记录 GPU-hours / compute。
