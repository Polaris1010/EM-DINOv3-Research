# Stage MV-1：MitoVerse Generalist Representation Benchmark v1

首批已完成：Guay/Jiang/BetaSeg，4个train volumes、4个dev volumes。原版MedDINOv3 Primus固定源码；P0/B100k/B150k encoder冻结。完成共同预算pilot和3个固定decoder seeds，每组20,000 steps，共9组训练、36组完整开发体积指标。

## 状态入口

服务器：`/opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013`。
服务：`mv-generalist-stage1-20261013.service`。`STATUS.json`是运行状态的权威入口；`logs/`保留各阶段记录。当前本地文件不是持续自动同步的状态监控。

最终服务器状态为complete。结果已下载并独立复算，`evidence/delivery_verification.json`为PASS：混淆矩阵公式、原始GT前景计数、来源均值、体积SD、seed SD、配对差值及三seed初始化/采样一致性均通过。受保护文件哈希不变；网站结束检查HTTP 200。GPU仅使用1/2/3。

## 已完成的训练前核验

- 212-volume eligibility/acquisition registry。未知物理采集体保留UNVERIFIED，不推断为独立。
- 首批8volumes完整像素QC，图像/标签元数据一致；读取`mito>0`实例前景，未读reserved test labels。
- 三encoder原版Primus真实三来源训练/开发ROI smoke，encoder参数不变，decoder初始化一致。
- slice-uniform与source-balanced实现及抽样核验：首轮固定source-balanced。
- 6项负向准入检查通过，覆盖train/dev源文件重叠、保留测试集、轴方向、越界、ignore政策和不支持的泛化声明。

## 解释边界

这是**transductive source-volume development**，不证明unseen acquisition。proofread与语义规则来自curator contract；数值QC不能证明人工标签穷尽。历史GT诊断曝光不清零，reserved test不等于untouched test。Stage0严格clean NO_GO未改变。

Guay/Jiang各只有1个dev volume，其体积SD为NA。仅BetaSeg有2个dev volumes满足primary source-macro门槛；三来源平均作为descriptive摘要。三个seed的SD与体积SD分开。

NSD@16nm为预设工程容差，附8/32nm和1voxel敏感性。HD95按原生ZYX nm spacing计算；不与CMCC/CMCT旧voxel指标直接混比。

本批MitoVerse的`mito`数组是线粒体实例标签，前景为`mito>0`。这与历史CMCC/CMCT原始多类别标签仅取`label==1`的规则不同；不能对实例ID机械使用`==1`。

共同预算锁定为20,000 steps，但P0预算pilot到达预设上限仍未满足收敛规则，因此结果不代表decoder已充分收敛，也不代表encoder的最终监督适配上限。

## 结果与决策

| 来源 | P0 Dice | B100k Dice | B150k Dice |
|---|---:|---:|---:|
| Guay | 0.756738 ± 0.003564 | 0.745874 ± 0.007008 | 0.746829 ± 0.007622 |
| Jiang | 0.867301 ± 0.001013 | 0.883872 ± 0.000704 | 0.882137 ± 0.001072 |
| BetaSeg/Muller | 0.893847 ± 0.003825 | 0.883981 ± 0.002957 | 0.884007 ± 0.003065 |

表中SD为三个decoder seeds之间的来源均值标准差，不是体积SD。B100k相对P0的Dice变化分别为−1.086、+1.657、−0.987个百分点，三个seed的方向一致。收益具有来源差异；100k到150k没有一致提升。BetaSeg的HD95改善而Dice/NSD下降，应分别解释各指标。

保留B100k作为EM预训练主对照，P0作为同等重要的自然权重对照；B150k保留作训练长度对照。下一轮先检查decoder共同预算是否充分、Guay/BetaSeg小目标漏检和真实输入尺度敏感性，并扩充可核验来源及开发体积；再依据密集特征证据决定Gram/HRA等试验。本批未评测MitoNet，尚不能回答是否超过MitoNet。

## 交付

固定数据与decoder：`mitoverse_supervised_eligibility.csv`、`mitoverse_acquisition_registry.csv`、`MV-Dev-v1.json`、`original_primus_config.yaml`、`MV_frozen_probe_protocol.yaml`、`split_exposure_audit.md`。

已完成结果：[完整分析](source_failure_analysis.md)、[逐体积指标](MV_frozen_probe_results.csv)、[体积SD](source_metrics_by_seed.csv)、[seed SD](source_metrics_three_seeds.csv)、[配对差值](paired_volume_deltas.csv)、[首seed目标分层](target_strata_seed1.csv)和`failure_panels/`。独立复核脚本为`scripts/verify_delivery.py`。

不训练额外SSL、Gram、HRA、LAST或TopoMask；P暂缓；不用GPU0，不停网站与未知进程。
