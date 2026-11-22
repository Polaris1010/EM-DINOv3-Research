# Argon 实验结果索引 · 2026-11-22

## 本轮归档

| 实验 | 审核入口 | 口径 |
|---|---|---|
| C0/C1 100k → 120k Gram → 130k HRA | [训练配置、完成回执与结果](../experiments/review-results-20261122/c0_c1_gram_hra_20261120/) | C0/C1 两条续训轨迹，HRA 采用显存预检通过的 512/112/768。 |
| C0/C1 130k 的 MB-1 | [C0](../experiments/review-results-20261122/c0_c1_gram_hra_20261120/mb1_130k_seed20261014/C0_Vanilla_100k/summary.csv) · [C1](../experiments/review-results-20261122/c0_c1_gram_hra_20261120/mb1_130k_seed20261014/C1_Barrier_100k/summary.csv) | CEM-MitoLab 匹配 Primus，40k、seed 20261014，每种窗口 2,595 个切片或二维视野，七组等权。 |
| 大窗口同中心诊断 | [100k 诊断](../experiments/MB1-Matched-Window/summary.csv) · [C0 Gram 120k 快测](../experiments/review-results-20261122/c0_c1_gram_hra_20261120/gram_screen/summary.csv) | 固定 52 个 MB-1 视野。快测复用同一 C0 100k Primus，指标只覆盖共同中央区域。 |
| HQ-5nm | [原网格](../experiments/HQ-5nm-ZeroShot/HQ-5nm_80_predictions_PNG.zip) · [输入降采样](../experiments/HQ-5nm-ZeroShot/HQ-5nm_half_input_80_predictions_PNG.zip) · [512/1024 大窗口](../experiments/HQ-5nm-ZeroShot/HQ-5nm_Primus_512_1024_predictions_PNG.zip) · [130k 四组预测](../experiments/HQ-5nm-ZeroShot/HQ5nm_C0_C1_130k_256_512_PNG.zip) | 无标注未知域。130k 含肺在内共 25 张视野，其余归档覆盖当时可用的 20 张。PNG 是预测掩膜。 |
| CEM-MitoLab 与 Mix 监督适配 | [初始 CEM](../experiments/review-results-20261122/cem_mitolab_three_20261118/) · [batch 4 汇总](../experiments/review-results-20261122/mix_mitolab_mv_20261118/batch4/all_batch4/) · [C0 消融](../experiments/review-results-20261122/mix_mitolab_mv_20261118/batch4/c0_matched/paired_metrics.csv) | 分开查看 batch 1、batch 4、CEM-only 与 Mix。三 seed 汇总以对应报表与实际完成回执为准。 |
| 官方 MitoNet 与 Primus 语义重测 | [2D 逐组与逐图](../experiments/review-results-20261122/mix_mitolab_mv_20261118/zeroshot_repeat_20261120/) · [官方 MitoNet 3D 报告](../experiments/MitoNet-v1-3D/report.md) | 3D 为发布权重的三轴共识，TEM-100 使用 2D。 |
| MitoNet 论文流程复现 | [复现表](../experiments/review-results-20261122/mitonet_paper_20261119/benchmark/Table1_reproduction.csv) · [配置和评分脚本](../experiments/review-results-20261122/mitonet_paper_20261119/) | 独立训练与论文体数据流程，单独对照发布版权重评测。 |
| 二维实例指标与新增实例分支 | [原模型重测](../experiments/review-results-20261122/mix_mitolab_mv_20261118/instance_2d_repeat_20261120/instance_metrics.csv) · [学习实例头 40k 报告](../experiments/review-results-20261122/barrier_primus_instance_20261120/final_report.md) | 旧 Primus 的连通域实例仅为后处理参照。学习实例头使用 CEM-MitoLab 原实例 ID。 |
| Pilot C100 三 decoder seeds | [完整报告与逐项结果](../experiments/review-results-20261122/pilot_c100_decoder_seeds_20261102/) | C0、C1、Trajectory-Lerp05 三头种子比较，编码器权重固定。 |

## 已发布的早期实验

| 系列 | 远端入口 |
|---|---|
| MV-1、MV-2 | [主分支实验注册表](https://github.com/Polaris1010/EM-DINOv3-Research/blob/main/experiments/registry.yaml) |
| MV-3 | [Generalist 监督训练](https://github.com/Polaris1010/EM-DINOv3-Research/tree/codex/mv3-generalist-training) |
| MV-4、MV-4B、MV-4C | [官方 MitoNet](https://github.com/Polaris1010/EM-DINOv3-Research/tree/codex/mv4-official-mitonet) · [Semantic/Panoptic 诊断](https://github.com/Polaris1010/EM-DINOv3-Research/tree/codex/mv4b-semantic-panoptic) · [输入溯源](https://github.com/Polaris1010/EM-DINOv3-Research/tree/codex/mv4c-input-provenance) |
| MV-5、原 MB-1 | [MitoNet 适配](https://github.com/Polaris1010/EM-DINOv3-Research/tree/codex/mv5-mitonet-adaptation) · [原 MB-1 benchmark](https://github.com/Polaris1010/EM-DINOv3-Research/tree/codex/mb1-official-benchmark) |
| D1/D2 与 Coverage-NR | [D1/D2 30k](https://github.com/Polaris1010/EM-DINOv3-Research/tree/codex/d12-ssl-diversity) · [100k](https://github.com/Polaris1010/EM-DINOv3-Research/tree/codex/d12-100k-continuation) · [三 seeds](https://github.com/Polaris1010/EM-DINOv3-Research/tree/codex/d12-three-seed-validation) · [Coverage-NR](https://github.com/Polaris1010/EM-DINOv3-Research/tree/codex/coverage-nr-p0-training) |
| Pilot A、A2、B、C | [A](https://github.com/Polaris1010/EM-DINOv3-Research/tree/codex/feature-mechanism-pilot-a) · [A2](https://github.com/Polaris1010/EM-DINOv3-Research/tree/codex/pilot-a2-readout-calibration) · [B](https://github.com/Polaris1010/EM-DINOv3-Research/tree/codex/pilot-b-results-public) · [C 45k](https://github.com/Polaris1010/EM-DINOv3-Research/tree/codex/pilot-c2-paired-45k) · [C 100k](https://github.com/Polaris1010/EM-DINOv3-Research/tree/codex/pilot-c-paired-100k) |

审阅时按数据划分、监督预算、seed、窗口尺寸和语义/实例口径对齐。MB-1 已参与开发诊断，HQ-5nm 的前景比例为描述性统计。CEM/Mix 的 batch 4 多 seed 原始评价文件中，seed 20261015 有八组，seed 20261016 可核对到三组 CEM-only Primus。其他 seed 20261016 组合在本次归档中缺少评价文件。大型权重、原始显微图像和运行缓存保留在原实验存储中。
