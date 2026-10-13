# CMCC / CMCT：仅 label == 1 的线粒体冻结评测

已对三个编码器的现有预测重新计算，GT 和预测均明确使用 `== 1`。其他非负语义标签视为背景。

## 三模型结果

| 数据集 | 编码器 | 验证体积 n | Dice ↑ | HD95 ↓ | NSD ↑ |
|---|---|---:|---:|---:|---:|
| CMCC | 官方 DINOv3 自然图像权重 | 3 | 0.7853 ± 0.0791 | 166.13 ± 126.69 | 0.1754 ± 0.1131 |
| CMCC | B_full 100k | 3 | 0.8867 ± 0.0658 | 77.00 ± 95.35 | 0.3120 ± 0.1903 |
| CMCC | B_full 150k | 3 | 0.8795 ± 0.0766 | 71.36 ± 77.15 | 0.3025 ± 0.1974 |
| CMCT | 官方 DINOv3 自然图像权重 | 2 | 0.9561 ± 0.0161 | 30.80 ± 12.55 | 0.6744 ± 0.1322 |
| CMCT | B_full 100k | 2 | 0.9615 ± 0.0142 | 27.38 ± 20.34 | 0.7174 ± 0.1332 |
| CMCT | B_full 150k | 2 | 0.9611 ± 0.0151 | 27.01 ± 21.64 | 0.7129 ± 0.1438 |

SD：完整验证体积之间的样本标准差，ddof=1；不是多种子标准差。CMCC n=3，CMCT n=2。HD95 数值按 voxel 等效单位报告；NSD 固定容差 1，读取原标签 spacing=(1,1,1)。

## 本次标签核查

实际使用的两个目录是：

- `/opt/data/3_117/tanghr/Med_DINOv3/nnUNet_raw/Dataset111_CMCC/labelsTr`
- `/opt/data/3_117/tanghr/Med_DINOv3/nnUNet_raw/Dataset131_CMCT/labelsTr`

两个 dataset.json 均声明 `background: 0, mitochondria: 1`。5 个完整验证标签都只有 0/1，当前输入中不存在其他正类别。全部 54 个预处理训练/验证标签文件均仅含 -1、0、1，-1 为忽略值。

| 数据集 | 体积 | 标签 1 体素数 | 其他正类别体素数 |
|---|---|---:|---:|
| CMCC | CMCC_crop113 | 3962742 | 0 |
| CMCC | CMCC_crop110 | 1501278 | 0 |
| CMCC | CMCC_crop3 | 15215697 | 0 |
| CMCT | CMCT_jrc_mus-liver_crop145 | 4190433 | 0 |
| CMCT | CMCT_jrc_mus-kidney_crop156 | 6297238 | 0 |

此前正式 Dice、HD95、NSD 的 MedDINOv3 指标入口已经使用 `label_array == c`，本轮只传入一个类别，c=1。冻结脚本中曾使用的 `GT > 0` 在当前二值输入上与 `GT == 1` 完全相同。此次重新导出明确的 label1 掩码并执行同一指标入口后，六组均值和 SD 与原结果一致。

没有根据标签纠正重训 encoder 或 decoder：已核实当前训练标签没有额外正类别。另已保存后续使用的 `frozen_probe_label1.py`，训练目标显式将 1 映射为前景、其他非负标签映射为背景、负值映射为忽略值；验证 GT 使用 `== 1`。原始历史脚本保留以保证已有实验可追溯。

## 结论

- 100k 相比官方自然权重，在两组的平均 Dice、HD95、NSD 均更好。
- 150k 相比 100k，两组平均 HD95 改善，但平均 Dice 和 NSD 下降；原来的比较结论不因此次明确 label1 重算而改变。
- 当前仍是小样本、单种子的验证集冻结测量，不能据此声称已超过 MitoNet 或已达到预训练上限。

完整精度和逐体积结果见 results.json、summary.csv、per_volume_metrics.csv；标签值计数、文件路径和哈希见 label_audit.json。
