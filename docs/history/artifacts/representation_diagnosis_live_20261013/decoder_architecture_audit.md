# Decoder 实际结构核验

日期：2026-10-13。证据来自本轮服务器实际文件、checkpoint 加载、GPU3 前向与梯度测试。此前同日的本地预审目录保留，本报告补充实时证据。

## 结论

**P0、B100k、B150k 的单种子冻结评测均已完成。实际解码器是包含四个频域门控的修改版 Primus_Multiscale。** 因此“自然权重尚未完成同协议评测”与“当前没有本科频域模块残留”均不符合现存执行证据。

P0 为官方 LVD-1689M 自然图像 ViT-L/16，不是 MedDINOv3 的医学持续预训练权重。本轮三模型真实加载成功，均使用相同架构、相同 decoder 初始参数和冻结 encoder。

## 1. 实际代码与模型构造

完整类名：`nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus.Primus_Multiscale`。

服务器源码：`/opt/data/3_117/tanghr/Med_DINOv3/MedDINOv3/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/dinov3/models/primus.py`。

本轮快照：[8_primus.py](sources/8_primus.py)。SHA256：`d394bd9655f4a2e67a0e5e464b6fd486597061b4cce33b0ae18dba898d81ce97`。实际导入路径通过 inspect 校验，依赖 LayerNormNd 的服务器文件也已捕获：[10_patch_encode_decode.py](sources/10_patch_encode_decode.py)。

| 项目 | 实际值 |
|---|---|
| Encoder | DINOv3 ViT-L/16，24 blocks |
| embed_dim / patch_size / storage tokens | 1024 / 16 / 4 |
| interaction_indices | `[5,11,17,23]`，0 起始，即第 6、12、18、24 个 block |
| 输出特征 | 四个 1024 通道特征图，拼接为 4096 通道 |
| 历史下游输入 | 224×224，灰度重复为三个通道；读取 nnU-Net 预处理图像 |
| 上游 SSL global/local | 256 / 112；与下游输入不同 |
| Head | 四个 LaSt_FrequencyGate → concat → PatchDecode |
| PatchDecode | 四级 2D 转置卷积，每级 LayerNormNd / GELU，末端 1×1 输出卷积 |
| 输出类别 | background / mitochondria 两个 logits |
| decoder 辅助容器 | deep_supervision=False，无参数 |

与已捕获的[固定提交官方 Primus](https://github.com/ricklisz/MedDINOv3/blob/a2460c2547899283446e520b84523454f7e25608/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/dinov3/models/primus.py) 比较，官方直接拼接四层特征，没有 `LaSt_FrequencyGate`。本项目新增四个门控，训练循环和模型 forward 都实际使用它们。官方源码快照保留于此前预审目录 `../representation_diagnosis_audit_20261013/sources/official_primus.py`。

门控沿 token 的**通道维**执行 FFT，并训练 sigma、gamma 和 channel_projector。它属于下游 LAST-inspired 频域模块，不能等同于上游 SSL 的 LAST。B_full 预训练 LAST=OFF 与下游含门控是两个独立事实。

## 2. 当前服务器真实参数计数

三模型分别真实实例化，计数完全一致，详见 [model_parameter_summary.json](smoke/model_parameter_summary.json)，含全部 trainable 参数名称与形状。

| 模块 | 参数数 | requires_grad / 实际梯度 |
|---|---:|---|
| Encoder | 303,154,176 | 全部 False；无梯度 |
| up_projection | 44,576,514 | 训练；梯度有限 |
| 四个频域门控 | 2,114,568 | 训练；sigma / gamma / projector 均获得有限梯度 |
| 合计 head | 46,691,082 | 全部训练 |
| 整体模型 | 349,845,258 | 仅 head 更新 |
| 新 Linear Probe | 2,050 | 最后一层 1024→2 的 1×1 Conv；仅 head 更新 |

历史训练显式 `encoder.eval()`、`torch.no_grad()`，BF16 提取特征，FP32 训练 gates / decoder。本轮也逐参数验证冻结，在每种 head 两个 optimizer updates 后 encoder 的完整 state_dict 哈希不变。此处不是使用 SyntheticEncoder 或 meta device 得到的估计。

## 3. P0 / B100k / B150k 执行与协议匹配

[实时证据](evidence/live_evidence.json) 收录服务器六份 complete 的 metrics 文件；原件逐份复制到 evidence 目录。服务器当前 150k STATUS 同样为训练完成、评价完成。

| Dataset | train / val | P0 选择步数 | B100k 选择步数 | B150k 选择步数 |
|---|---:|---:|---:|---:|
| CMCC | 27 / 3 | 3000 | 2000 | 2000 |
| CMCT | 22 / 2 | 3000 | 3000 | 3000 |

所有组训练预算均为 3,000 steps、seed 20260907；按相同验证选择规则选择 head，选择到不同步数不等于训练预算不同。

共同 head 初始化 SHA256：`4b19c61be6451662481678ca87ffac1162812df24c31422c0d8d3f63acc358cf`。P0 / B150k 历史记录直接保存；B100k 原 metrics 没有该字段，后续重建报告保存。本轮用三份真实 encoder 再次独立构造 head，均复现此哈希。

训练采用 batch=1，AdamW lr=1e-4 / weight_decay=.05 / betas=(.9,.98)，50-step warmup 后 cosine，clip=12；前景坐标采样概率 50%，其余 uniform，CE+foreground Dice。每 1000 steps 用 stride-20 验证选择 head；最终完整验证体积、stride-1 与四翻转 TTA。P0/B100k 的控制审计此前已完成；本轮读取原脚本、控制脚本、150k 派生脚本和匹配的 split / seed / 初始化记录，未发现协议分叉。

**可归因范围：** 同一 gated decoder 内可以比较三份 encoder；这些结果不能称为“未修改官方 Primus 复现”，也不能与本科不同 decoder 的结果直接归因比较。后续 ungated Primus 必须独立命名，对所有 encoder 使用一致的新初始化与训练协议。

## 4. 标签、有效区与历史指标

本轮两份 dataset.json 均为 background=0、mitochondria=1。实际训练 smoke 的 CMCC_crop20 标签为 -1/0/1，-1 忽略。此前 2026-10-12 label1 审计检查了五个完整验证 GT（仅 0/1）和 54 个预处理 case（仅 -1/0/1），并明确用 GT==1 / prediction==1 重算三指标。该报告和结果现从服务器再次复制，见 evidence/final_label_audit.json、evidence/final_results.json。

| Dataset | Encoder | Dice ↑ | HD95 ↓ | NSD@1 ↑ |
|---|---|---:|---:|---:|
| CMCC | P0 | 0.785344 ± 0.079056 | 166.13 ± 126.69 | 0.175351 ± 0.113098 |
| CMCC | B100k | 0.886671 ± 0.065791 | 77.00 ± 95.35 | 0.312016 ± 0.190296 |
| CMCC | B150k | 0.879463 ± 0.076584 | 71.36 ± 77.15 | 0.302451 ± 0.197386 |
| CMCT | P0 | 0.956144 ± 0.016103 | 30.80 ± 12.55 | 0.674437 ± 0.132182 |
| CMCT | B100k | 0.961455 ± 0.014152 | 27.38 ± 20.34 | 0.717415 ± 0.133223 |
| CMCT | B150k | 0.961065 ± 0.015052 | 27.01 ± 21.64 | 0.712862 ± 0.143823 |

这是既有 label1-only 重算结果的汇总，**本轮没有重新训练或计算上述六组 benchmark**。SD 是验证体积间 sample SD（ddof=1），不是 seeds 的 SD。距离使用原协议 spacing=(1,1,1)，HD95 为 voxel-equivalent，NSD 容差为 1，不是已校准 nm。

标注完整性、原始 annotation-valid mask、同 acquisition 隔离及 raw/preprocessed 图像的精确几何对应仍为 **UNVERIFIED**。二值标签本身不能证明 0 都是可靠背景。MitoVerse 的 mito 正实例 ID 不能套用 CMCC/CMCT 的 label==1 语义规则。

## 5. 本轮 smoke 的意义

三模型各进行了 Primus 与 Linear 两步 head 优化；真实训练输入、loss 和有限梯度已通过。测试 head 仅存于内存，未覆写历史 head。另有三模型×两种 FoV×三个分辨率，共 18 次真实 EM 输入 forward，输出及四层特征均有限。详见 [real_weight_smoke.json](smoke/real_weight_smoke.json)。

这些测试确认实现和冻结约束可用；不构成 Linear Probe 最终成绩、HRA 性能提升或 150k dense 退化的证据。
