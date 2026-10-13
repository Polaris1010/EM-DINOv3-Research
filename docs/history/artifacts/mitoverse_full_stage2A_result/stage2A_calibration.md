# Full-MitoVerse Stage 2A 校准结论

**STAGE2A_GATE = FAIL；AUTO_LAUNCH_T1 = false；T1 未启动。**

时间：2026-09-30T09:49:10.007144+00:00。Stage 1 只读；GPU 0 未用于本轮计算。网站 HTTP 200。

## 已通过

- 与 Stage 1 相同的 424 张切片（212 volumes × 2），完整 manifest 匹配。
- 官方 SAM ViT-H 权重 SHA-256 `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`，424 张相同样本全部生成 6 种几何变换 MAM，未使用标签。
- ViT-B 与 ViT-H 固定 prior 的 5 种 flip/90° rotation 逆变换全部逐像素相同；H0/H1 bottleneck 全部为 0。ViT-H 为 2,120/2,120 次。
- ViT-H 在线重新编码仍不等变：hflip MAM MAE 中位数 0.1141，H0/H1 bottleneck 中位数 0.1576/0.1568。正式算法必须用固定离线 prior。
- ViT-H 原始 uint8 1024 tile 编码中位数 0.239 秒，三卡纯编码粗估 1.90 小时，未计入 IO。
- crop/context PH score 有动态范围；在训练可用尺度内，较强 crop 分数更差的配对比例 92.5%。

## 阻断项

按 B_full global crop scale 0.32–1.0，使用同一样本、同一 crop 形状的 4 个空间位置，共 1,696 个候选；context 扩张 1.5 倍，H0/H1 各 0.5 权重，按 source × scale bin 做 median/IQR 归一化，小 source 回退全局统计。

MAM 低 affinity 对应邻接特征差异，故用 `1-MAM` 在 crop 边框 5% 带的均值除以 crop 总体均值，作为完全无标签的结构截断代理。**同一张样本内的 score/proxy Spearman 中位数为 0.000、均值 -0.019；最低分候选比最高分候选代理更低的比例只有 47.9%，最低分候选代理均值反而高 0.0155。** 跨样本 pooled Spearman 为 -0.114。

Q85/Q90 fallback 分别为 6.1%/4.5%，但 Q90 选择后代理均值 1.1226 与随机 1.1272 只相差约 0.4%。当前分数主要区分 crop 尺度的破坏程度，缺乏选择**相同形状下更少截断的位置**的证据。执行单要求该关系合理，因此该硬门槛失败。

## 执行边界

已停止全量 MAM 缓存、B/T 100k crop 模拟与 T1 的准备和启动。坐标映射、native grid、dtype、缓存读取、正式训练开销等仍未通过，不应视为 PASS。要继续需重新设计并验证有正向位置判别能力的无标签指标，然后重做 Stage 2A；不得直接沿用本轮 Q85/Q90/Q95 阈值进入 T1。

证据：`topology_calibration/stage2A_gate.json`、`crop_score/crop_score_pilot_424x4.csv/json`、`crop_score/crop_score_diagnostic_424.json`、`equivariance/vit_b_equivariance_424.json`、`equivariance/vit_h_equivariance_424.json`、`sam_vit_h/paired_B_H_summary.json`、`cache_benchmark/vit_h_raw_tile_10.json`。
