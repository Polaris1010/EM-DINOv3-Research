# 冻结评测：Dice、HD95、NSD（2026-10-12）

使用 MedDINOv3 官方 `compute_segmentation_metrics` 测试函数，已验证与上游函数 AST 相同。沿用已保存的 B_full / 官方自然图像初始化预测，不重新训练。

| 数据集 | 冻结编码器 | 验证体积 n | Dice ↑ | HD95 ↓（voxel 等效单位） | NSD ↑（τ=1.0） |
|---|---|---:|---:|---:|---:|
| CMCC | 官方 DINOv3 自然图像权重 | 3 | 0.7853 ± 0.0791 | 166.13 ± 126.69 | 0.1754 ± 0.1131 |
| CMCC | B_full（EM 100k） | 3 | 0.8867 ± 0.0658 | 77.00 ± 95.35 | 0.3120 ± 0.1903 |
| CMCT | 官方 DINOv3 自然图像权重 | 2 | 0.9561 ± 0.0161 | 30.80 ± 12.55 | 0.6744 ± 0.1322 |
| CMCT | B_full（EM 100k） | 2 | 0.9615 ± 0.0142 | 27.38 ± 20.34 | 0.7174 ± 0.1332 |

统计说明：每个完整验证体积先计算指标，再等权平均；± 为体积间样本标准差（pandas std，ddof=1），不是多随机种子标准差，也不是标准误。CMCC n=3、CMCT n=2。

距离说明：官方入口读取标签文件 spacing，当前所有文件均为 1×1×1，NSD 的固定容差 1.0 数值上等于 1 voxel。NIfTI 头标有 mm，但 EM 真实物理尺度尚未校准，因此本表按 voxel 等效单位说明，不解释为真实毫米。没有重采样。

实现说明：入口实际调用 MONAI HD95 和 surface Dice；没有使用同文件中未被入口调用的 SimpleITK compute_hd95 / compute_nsd 辅助函数。MONAI NSD 默认 use_subvoxels=False。

评测范围：二值线粒体语义分割、原始网格完整 3D 体积。冻结 encoder、同一 decoder 协议、单 seed、原验证划分与原 decoder checkpoint 不变。官方自然图像权重不是 MedDINOv3 CT-3M 权重；不把这次两模型冻结评测与参考表中的完整训练模型混为一组。

该结果属于验证集测量，SSL/下游数据重叠审计仍待完成，不能声称独立测试或未见来源泛化。P 继续暂缓，不启动长训。

源码版本：[a2460c254789](https://github.com/ricklisz/MedDINOv3/blob/a2460c2547899283446e520b84523454f7e25608/nnUNet/nnunetv2/compute_metrics.py)；MONAI 1.5.0。完整精度与逐体积结果见 results.json、summary.csv、per_volume_metrics.csv。
