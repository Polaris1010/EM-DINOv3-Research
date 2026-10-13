# MitoTopo-DINO 与 MitoNet：实测差距与探索实验

本轮尚未超越 MitoNet。已完成官方权重对照、推理消融、Gram／无 Gram 各 50 步联合预训练短测，以及四组各 200 步的冻结编码器解码器实验。完整 Gram 阶段已经启动，已超过 1,000 步，Gram 权重达到 2.0，且第 1,000 步的 student／EMA teacher／Gram teacher／优化器检查点已核验；目标为阶段内 200,000 步，它尚未完成，不能将短测检查点称为完整官方复现或最终模型。

**同数据实测结果**

固定随机种子 20260907，在原探索划分的 5 个 conrad23 验证体中各取 4 个 192×192 图块。3 个原图完全为零，单独标记为不可提供有效图像信息的样本；主表排除它们，使用剩余 17 个图块。这个排除规则只依据原图是否为常数，不依据标签或预测。剩余 salivary 图块也没有前景标注，因而该域不足以评估线粒体检出能力。原始 20 图块统计仍完整保留。

| 实测配置 | Dice | IoU | PQ | Recall |
|---|---:|---:|---:|---:|
| 官方 MitoNet v1 | 0.8140 | 0.6864 | 0.4800 | 0.7214 |
| 现有同结构权重，epoch 1 | 0.1014 | 0.0534 | 0.0000 | 0.4696 |
| 现有同结构权重，epoch 266 | 0.0174 | 0.0088 | 0.0000 | 0.0127 |
| epoch 266，原配置 224 补边＋翻转集成 | 0.0200 | 0.0101 | 0.0000 | 0.0143 |
| 冻结 EM 编码器＋轻量解码器 | 0.0815 | 0.0425 | 0.0044 | 0.0942 |
| 轻量解码器＋边界辅助监督 | 0.0679 | 0.0352 | 0.0043 | 0.0738 |
| 无 Gram 续训 50 步＋轻量解码器 | 0.0814 | 0.0424 | 0.0000 | 0.0935 |
| Gram 续训 50 步＋轻量解码器 | 0.0813 | 0.0424 | 0.0045 | 0.0937 |

Dice/IoU/Recall 在全部有效像素上累计；PQ 在全部匹配实例上累计，匹配要求 IoU > 0.5。未使用“空图 PQ=1”的简单平均作为主结果。GT 保留裁剪后的实例 ID，现有语义模型用连通域生成实例，MitoNet 用官方中心／偏移后处理。

MitoNet 对有效图块的 Dice 领先 epoch-266 原配置补边／翻转诊断约 79.4 个百分点。这个数值只代表本轮小样本跨域诊断，不代表完整体积、官方基准或模型架构的能力上限。没有完整 3D 重建、体素物理分辨率匹配、完整 sliding-window 评测、独立测试集或多种子统计检验。历史预训练是否见过这些数据仍有不确定性。

**差距在哪里**

MitoNet 提供语义、中心热图和像素偏移的实例分割链路；其官方工具还支持 3D 跟踪及多方向融合。现有模型是 ViT-L 的第 6/12/18/24 层特征，经通道频域门控后连接上采样解码器，输出前景／背景；没有原生实例头和经过验证的 3D 拼接链路。因此 Dice 接近也不意味着实例质量接近。相关依据见 [empanada 官方推理代码](https://github.com/volume-em/empanada/blob/main/empanada/inference/engines.py) 与本轮 source_snapshot。

本轮核对了 epoch-266 权重内嵌的 ImageNetNormalization、mitochondria 标签以及 Dataset131_CMCT 训练来源；严格加载没有缺失或形状不符。该训练集元数据只有 24 个训练 case，文件名包括 JRC mouse kidney/liver 的裁剪。历史 EMA≈0.970 是其自身训练流程的记录，不能当作这里的跨域分数。网络前向实现经过历史修改，缺少每份旧权重对应的完整历史源码快照，仍不能保证严格重现当年的推理行为。

本轮当前权重主要表现为漏检；epoch-1 权重还产生大量碎片假阳性。四次翻转、关闭频域门控、固定距离分水岭均未解决问题。门控沿特征通道做 FFT，不能把它解释成已完成图像空间的 Fourier 域适配。可查看 comparison_preview.png：原图、GT、MitoNet、epoch-266 预测按列对齐。

**Gram 已做什么，哪些还没有完成**

直接调用现有 DINOv3 仓库的 SSLMetaArch、DataAugmentationDINO、collate_data_and_cast 和 GramLoss，联合计算 DINO＋iBOT＋KoLeo＋Gram。训练权重为已校验的 EM-v2 teacher training_124999；首次短测将其装入 student 和 EMA teacher，重新初始化优化器，未冒充恢复原预训练优化器。完整续训则恢复本轮 50 步检查点中的 student、EMA teacher、Gram teacher 和优化器。

已通过 Gram 相同特征零损失、非相同特征非零有限梯度、真实数据多裁剪形状、调度数值、真实 GPU 前后向和检查点恢复检查。50 步中的第 1 步权重为 0，后续 49 步为正，最后权重 0.0118487。短测显存峰值约 7.9 GiB allocated。无 Gram 对照保留相同数据、初始化、步数和其他损失。

Gram 50 步后的轻量解码器 Dice 0.081308，与无 Gram 对照 0.081361 基本相同。PQ 的微小差异涉及极少数匹配实例，不能视为可靠提升；50 步尚处在官方权重爬升最初阶段，也不能据此否定完整 Gram 阶段。

[官方 Gram 配置](https://github.com/facebookresearch/dinov3/blob/main/dinov3/configs/train/dinov3_vit7b16_gram_anchor.yaml) 面向 ViT-7B，并使用从百万步预训练阶段延续的时间轴；官方 ckpt 字段仍需用户指定。这里保留原文件，并记录如下适配：

| 项目 | 本轮设置与原因 |
|---|---|
| 网络与预训练头 | 保留已有 ViT-L、0 storage tokens 和 65536 原型等可严格加载的结构；无法把 7B 权重形状直接套入 ViT-L |
| Gram 数学参数 | normalized=true、img_level=true、tokens=all、不截断负相关；bicubic，下采样 antialias=false |
| 裁剪 | global 256、local 112×8、Gram teacher 512，沿用官方设置 |
| 调度 | 本地步骤 s 读取官方 1,000,000+s；这不是声称已做百万步 EM 训练。实际旧 EM 训练为 125,000 步 |
| Gram 权重 | 本地 0→999 步按官方余弦由 0 升至 2，随后保持 2 |
| Anchor 更新 | 本地 10k 开始，每 10k 更新，最多 3 次，沿用官方规则 |
| 优化器 | AdamW β=(0.9,0.99)、WD=0.04、clip=30；教师温度0.07、EMA momentum0.999 |
| LR | 参考 LR=3e-5，并按官方 sqrt_wrt_1024 规则缩放实际 batch=2，得到 1.325825e-6 |
| 批量与精度 | 单卡 batch=2，是每个全局视图内 KoLeo 的最小非退化样本组；FP32 主参数、BF16 autocast、全块 activation checkpointing；train.compile=false，不使用 FSDP 分片 |
| 加载与缓存 | 无整卷 RAM 缓存。最初 workers=2 触发 swap 保护，改为 workers=0 后完成短测；完整续训按需从训练体读取 512 图块 |
| 保存 | 每 1000 步原子更新本轮 latest 完整检查点和 teacher export；中断保存。当前实现保留 latest，与官方多历史检查点保留策略不同 |

该实现是“官方损失和阶段调度的 ViT-L 单卡适配”，不是逐项不变的官方 7B 配置复现。

**探索方案与实际状态**

| 方案 | 解决的问题 | 当前状态 |
|---|---|---|
| G0：官方 Gram＋无 Gram 对照 | 修复稠密表征，检验增益来自 Gram 还是额外训练 | 各 50 步完成；完整 Gram 阶段运行中 |
| G1：早期 EM anchor 对照 | 检查后期 anchor 是否已经丢失细节 | 方案已写；早期 checkpoint 仍需审计与训练 |
| S0：冻结编码器＋轻量解码器＋原图浅层特征 | 减少解码器过拟合，补低层细节 | 200 步完成，未超过 MitoNet |
| S1：边界辅助监督 | 改善轮廓、降低粘连 | 语义轮廓版本 200 步完成；相邻实例边界版本待实现 |
| S2：MitoNet 式中心／offset 实例头 | 从语义检测升级为实例分割 | 已给官方损失和后处理参数；未训练 |
| S3：2.5D＋z-affinity | 改善跨切片连续性与 split/merge | 已给数据／分辨率／拼接方案；未训练 |
| S4：训练域增强与尺度条件 | 缓解灰度风格和体素尺度差异 | 已给无验证数据参与的方案；未训练 |
| I0：TTA、去门控、分水岭 | 验证是否只需改推理就能补差距 | 已完成，无有效改进 |

下一轮监督训练应先在训练集内确认标注覆盖及前景采样比例，再扩展经过 specimen 审计的训练域；目前只用 jiang25_vol1 的短测不足以证明泛化能力。所有阈值和实验选择仍属于探索性验证；正式“超越”需要锁定独立 specimen 测试集，在相同 3D 协议下比较，并复核预训练重叠。

**交付与正在运行的作业**

服务器目录：`/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/`。

- `gram_full_01/steps.jsonl`：逐步损失；`resource_log.csv`：资源记录；`progress.json`：每次保存后的进度。
- `gram_full_01/checkpoint_latest.pth`、`teacher_latest.pth`：每 1000 步更新；首次保存前可用 `gram_smoke_workers0/checkpoint.pth`。
- `guard_result.json` 仅在守护结束后生成；需结合进程状态判断正在运行或已终止。
- `metrics_summary.json`／`comparison_table.csv`：累计指标与分域结果；`patch_quality.json`：纯黑图块记录。
- `plans/*.json`、`experiment_plans.json`：方案与参数；`source_snapshot/`：实际调用源码。
- `light_decoder_01/*.pth`：四个轻量解码器检查点。大型预训练检查点保留在服务器，未重复打入交付包。

本轮只使用 GPU3；GPU1/2 的已确认网站进程未受影响，GPU0 未使用。多次 swap 连续增长时已停止本轮子进程；低内存配置的短测完成。长训仍由相同守护监控，资源条件不满足会自行停止，不保证其能不间断完成。交付包是进行中的快照，不包含未来训练结果，也没有部署或替换线上模型。

MitoNet 权重来源为[官方配置所指向的 Zenodo 发布](https://zenodo.org/records/6861565)，SHA-256：`487d63dd1deb971a3e374785b350673ee5427e0ffcd44880bf9a54f6c2c4bf04`。现有 EM 权重 SHA-256：`53f759c7019f8753d9d18b52aa79168b51b7480466934df47f446bc3d5f696524`。
