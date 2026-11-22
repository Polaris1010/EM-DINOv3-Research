# Argon 实验结果索引 · 2026-11-22

## 本轮归档

### 方孔与 GT 内部漏洞复核

| 阶段 | 已验证的结果 |
|---|---|
| 形态及编码器特征 | [诊断报告](../experiments/Square-Hole-Diagnosis/DIAGNOSIS_SUMMARY.md)。380 张掩膜，五个编码器，10 个特征对照视野。 |
| 单窗与拼接追踪 | [日志口径与逐孔测量](../experiments/Square-Hole-Diagnosis/fusion_trace/FUSION_TRACE_SUMMARY.md)。P0 与 C1 的 12 张历史掩膜逐像素复现。 |
| 半 patch 偏移 | [四图 GT 快测与 HQ-5nm 相位结果](../experiments/Square-Hole-Diagnosis/phase_shift_test/PHASE_SHIFT_SUMMARY.md)。 |
| 破相位全量推理 | [HQ-5nm 与 MB-1 报告](../experiments/Square-Hole-Diagnosis/full_phase_eval/FULL_PHASE_EVAL_SUMMARY.md)。前三组各 2,595 个 MB-1 视野。四相位的 MB-1 口径仅为 TEM-100 与 Glycolytic muscle 的 402 个视野。 |
| GT 去边界内部漏洞 | [r=3/5 报告](../experiments/Square-Hole-Diagnosis/gt_interior_holes/GT_INTERIOR_HOLES_SUMMARY.md)。四种配置各 2,595 个视野，MitoNet 为六个 3D 体数据加 TEM-100 官方 2D 的混合参照。 |

人工复核拼图的原件留在服务器独立诊断目录。审核分支保留样本索引、方孔坐标、复算脚本和逐项指标。

### 冻结 C1 双解码头

[最终报告](../experiments/Decoder-Cavity-Instance/EXPERIMENT_SUMMARY.md) · [训练协议](../experiments/Decoder-Cavity-Instance/PROTOCOL.md) · [独立核查](../experiments/Decoder-Cavity-Instance/audit.json)。两组已完成 40k，训练、双窗口评测及逐视野结果均已归档。每个组合覆盖 MB-1 2,595 个视野、CEM 验证 1,424 个视野、MV3 38 个区域和 HQ-5nm 20 个视野。两组均降低 MB-1 内部漏分率。B 的分水岭分开了更多接触实例，同一前景掩膜的连通域 PQ 更高。

### 历史实验结果

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
