# Decoder 实际结构审计 · 2026-10-13

**结论：P0 已完成评测；历史 frozen probe 使用经过修改、含频域 gates 的 Primus_Multiscale。** P0、B100k、B150k 的同协议单种子对照已经存在，无须将 P0 当作尚未运行。当前服务器状态未重新读取，本报告基于本地保存的服务器执行记录、哈希匹配源码与官方源码；新真实权重 forward 为 UNVERIFIED。

## 证据及模型

完整导入类名：`nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus.Primus_Multiscale`。

历史服务器源码：`/opt/data/3_117/tanghr/Med_DINOv3/MedDINOv3/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/dinov3/models/primus.py`。

本地逐字节匹配快照：`../mitotopo_first_round/remote/source_snapshot/primus.py`；SHA256 为 `d394bd9655f4a2e67a0e5e464b6fd486597061b4cce33b0ae18dba898d81ce97`，与 `../mitoverse_full_stage2a2_control_20261011/protocol_match.json` 中实际依赖哈希相同。其依赖记录说明自早期 B probe 后未修改。150k 使用同一模块导入和显式 gated forward；150k 执行时单独的模块文件哈希未保存在其 metrics 中，因此该时点源文件身份没有独立复核。

与固定提交的 [MedDINOv3 官方 Primus 源码](https://github.com/ricklisz/MedDINOv3/blob/a2460c2547899283446e520b84523454f7e25608/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/dinov3/models/primus.py) 比较，官方多层拼接后直接解码，未包含 `LaSt_FrequencyGate` 或 `last_gates`。本项目版本新增四个 gates，并在 probe 的训练循环中显式执行；不是 dormant 模块。

| 项目 | 实际配置 |
|---|---|
| Backbone | 官方 DINOv3 ViT-L/16；P0 与两个 B checkpoint 同架构 |
| embedding / patch / storage tokens | 1024 / 16 / 4 |
| interaction_indices | `[5,11,17,23]`；get_intermediate_layers 的索引从 0 开始，即第 6、12、18、24 个 block |
| 多层拼接 | 四层、4096 个通道 |
| 下游类别 | 两个 logits：background / mitochondria；语义分割 |
| 下游训练输入 | 224×224；不可与 SSL global view 的 256 混淆 |
| 解码路径 | 四个频域 gates → 拼接 → 四级转置卷积、LayerNormNd、GELU → 1×1 输出层 |
| encoder 训练 | 所有参数 requires_grad=False；训练时 encoder.eval()，特征提取 no_grad |
| 训练精度 | 历史 encoder 特征 BF16，gates 和解码器 FP32 |

`LaSt_FrequencyGate` 对每个空间 token 的**通道维**执行 FFT 平滑；学习 sigma、gamma 和 channel_projector。它不是空间 FFT，也不能仅凭名称将它等同于 SSL 的 LAST 聚合机制。B_full 的 SSL 无 LAST 与其下游存在 LAST-inspired gate 可以同时成立。

## 参数摘要及 smoke

| 模块 | 参数数 | 历史是否训练 |
|---|---:|---|
| up_projection | 44,576,514 | 是 |
| 四个 last_gates | 2,114,568 | 是，包括 sigma / gamma / channel_projector |
| 合计可训练 head | 46,691,082 | 是 |
| decoder 辅助容器 | 0 | 仅保存 deep_supervision=False |
| 真实 encoder 参数总数 | UNVERIFIED | 本轮未加载真实 encoder checkpoint |

计数由原始 class body 在 PyTorch meta device 上实际构造得到，见 [参数摘要](model_parameter_summary.json)。LayerNormNd 使用已捕获的 [官方参考实现](https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/building_blocks/patch_encode_decode.py)，含每通道 weight/bias；服务器安装版本仍待读取。因此这是源码构造计数，不能冒充当前服务器模型重新实例化的结果。

[CPU smoke](decoder_cpu_smoke.json) 使用真实 head 维度 1024、32×32 合成输入和明确命名的 SyntheticEncoder：前向、loss、backward 均有限；gates 和 up_projection 均有有限梯度；encoder 无梯度、参数哈希不变且保持 eval。此测试不验证 P0/B checkpoint 的特征数值。参考框架 base 与初始化 helper 在 CPU harness 中替换，完整 nnU-Net 环境未复现。

## P0 / B100k / B150k 是否匹配

六份完成的 metrics 整理于 [historical_probe_evidence.csv](historical_probe_evidence.csv)，原件保留。

| 开发验证集 | train / val cases | P0 Dice | B100k Dice | B150k Dice |
|---|---:|---:|---:|---:|
| CMCC | 27 / 3 | 0.785344 | 0.886671 | 0.879463 |
| CMCT | 22 / 2 | 0.956144 | 0.961455 | 0.961065 |

所有模型的 decoder seed 为 20260907，训练 3,000 steps；encoder 冻结，每个 encoder 重新初始化并训练自己的 head。共同初始化哈希为 `4b19c61be6451662481678ca87ffac1162812df24c31422c0d8d3f63acc358cf`。P0 / 150k metrics 直接保存哈希，100k 由协议审计及未执行优化的初始化重建确认；不能把这两种证据来源混为一个新的实时测量。

P0 与 B100k 的训练循环、sampler/loss/validation/seed AST 一致，原 B 前向 compatibility 差异为 0，首个训练样本一致。150k 构建脚本从该控制脚本派生，只更换 encoder 路径、步数与记录字段，执行记录的 split / seed / head 哈希一致。

共同训练协议：batch 1，AdamW lr=1e-4、weight_decay=.05、betas=(.9,.98)，50-step warmup 后 cosine、梯度裁剪 12；50% 前景坐标采样与 50% uniform；CE+foreground Dice，负标签忽略。每 1,000 步用 stride-20 验证 Dice 选择 head；最终 stride-1、四翻转 TTA 评价完整验证体积。旧脚本原样保存，后续显式 label1 版本另存。

## 标签与结论范围

`../mitoverse_CMCC_CMCT_label1_20261012/label_audit.json` 确認两个 dataset.json 均声明 background=0、mitochondria=1。五个完整验证标签仅 0/1，54 个预处理文件仅 -1/0/1；训练中的 -1 被忽略。label1 重算明确使用 GT==1 / prediction==1，结果与原结果一致，当前成绩确实是 mitochondria-only，不是多细胞器总分。

**标注完整性 UNVERIFIED**：只发现二值标签不能证明所有真实线粒体都标全，也不能确认预处理 -1 是否完整表达原始标注有效区域。现有 raw-label 评价把 0 当作背景，未证实应用独立 annotation-valid mask。

fold-0 仅能确认 case ID 不交叉，不能证明原始 acquisition / ROI 隔离。三个 CMCC 与两个 CMCT 验证 crop 在 MitoVerse 中有对应来源，B_full SSL 已将其来源 crop 纳入训练候选池；原始与预处理像素几何映射尚未重新核验。它们属于开发与 transductive 对照，不是严格 unseen-acquisition 测试。

历史 SD 来自验证体积，**不是三个 decoder seeds**。NSD 容差 1，spacing=(1,1,1)，HD95 为 voxel-equivalent 单位，不应写成 nm。100k→150k 的平均 Dice 下降不能独立证明 dense feature degradation。强 gated decoder 也可能影响预训练差异的可读出性，下一轮应增加统一 Linear Probe，并将 ungated Primus 明确登记为新的对照，避免悄然改写历史模型定义。
