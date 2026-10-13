# 原版 MedDINOv3 Primus 冻结评测

六组重新初始化 head 并训练完成。固定官方源码，无频域门控。

| Dataset | Encoder | Dice ↑ | HD95 ↓ | NSD@1 ↑ |
|---|---|---:|---:|---:|
| CMCC | P0 | 0.784606 ± 0.079449 | 166.25 ± 127.11 | 0.174574 ± 0.112801 |
| CMCC | B100k | 0.889716 ± 0.061090 | 77.40 ± 96.74 | 0.317518 ± 0.183327 |
| CMCC | B150k | 0.878806 ± 0.077265 | 70.42 ± 75.34 | 0.301053 ± 0.196836 |
| CMCT | P0 | 0.954885 ± 0.015951 | 33.35 ± 12.93 | 0.666005 ± 0.129070 |
| CMCT | B100k | 0.960962 ± 0.014212 | 27.69 ± 20.72 | 0.713795 ± 0.133351 |
| CMCT | B150k | 0.960518 ± 0.015057 | 27.48 ± 21.89 | 0.709115 ± 0.143459 |

## 原版 − 历史门控版均值差

| Dataset | Encoder | ΔDice | ΔHD95 | ΔNSD |
|---|---|---:|---:|---:|
| CMCC | P0 | -0.000739 | +0.12 | -0.000777 |
| CMCC | B100k | +0.003045 | +0.40 | +0.005502 |
| CMCC | B150k | -0.000656 | -0.94 | -0.001398 |
| CMCT | P0 | -0.001259 | +2.55 | -0.008432 |
| CMCT | B100k | -0.000493 | +0.31 | -0.003620 |
| CMCT | B150k | -0.000546 | +0.47 | -0.003747 |

## 协议

- Encoder 冻结，官方 ViT-L/16，层索引 [5,11,17,23]；官方 Primus 参数 44,576,514，无 LAST/频域 gates。
- 全部 head 初始化 SHA 一致，seed 20260907，batch1，3000 steps，AdamW/CE+Dice/采样/验证选择/四翻转 TTA 与历史协议匹配。
- 标签仅 GT==1 / prediction==1，HD95 与 NSD 使用固定 MedDINOv3 官方测试入口。NSD 容差1，距离为 voxel-equivalent，未校准 nm。
- SD 为 CMCC 3 / CMCT 2 个验证体积间样本标准差，不是 seeds 的波动；单 seed 不作显著性结论。
- B_full 为 transductive 开发资产；标注完整性、物理 spacing 与 acquisition 隔离尚待补证，尚不能证明超过 MitoNet。
- 既有 checkpoint、数据 manifest、历史门控版结果和网站代码保持原样；本次只训练新官方解码器。

## 同一原版 decoder 内的 encoder 均值差

| Dataset | 对比 | ΔDice | ΔHD95 | ΔNSD |
|---|---|---:|---:|---:|
| CMCC | B100k minus P0 | +0.105110 | -88.85 | +0.142944 |
| CMCC | B150k minus B100k | -0.010909 | -6.98 | -0.016466 |
| CMCT | B100k minus P0 | +0.006077 | -5.66 | +0.047790 |
| CMCT | B150k minus B100k | -0.000444 | -0.21 | -0.004680 |

上述差值已按逐体积结果独立复算；这些是单种子开发验证结果，不能作显著性或最终独立泛化结论。

## 本轮结论与后续基线

- 官方原版 Primus 下，B100k 相比自然 P0 在 CMCC、CMCT 的 Dice、HD95、NSD 三项指标均改善。
- 100k→150k：CMCC Dice -0.010909、NSD -0.016466；CMCT Dice -0.000444、NSD -0.004680。HD95 分别改善 6.98、0.21 voxel-equivalent。150k 未显示一致的整体收益，建议 B100k 作为下一阶段主基线，150k 保留作对照。
- 更换 decoder 后，没有观察到本科频域门控的稳定正收益；其效应在本轮单 seed 中较小且方向不一致。后续本研究统一使用原版 Primus，不以历史门控版结果作为原版基线。
- 下一项优先补充 Frozen Linear 与匹配 seeds，继续诊断 dense features / resolution；不据此直接启动 Gram/HRA 或延长到 200k。

实现来源：[MedDINOv3 固定提交官方 Primus](https://github.com/ricklisz/MedDINOv3/blob/a2460c2547899283446e520b84523454f7e25608/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/dinov3/models/primus.py)。GitHub 接口复核通过；runtime 快照仅文件末尾多一个 LF，函数与全部 AST 完全一致。
