# Feature Mechanism Pilot A Report

## 实验发现

本轮比较P0、D1/100k、D2/30k、D2/100k与Coverage-NR/100k，四层[5,11,17,23]。所有模型使用同一预先锁定坐标、native256视野和官方post-norm patch tokens；CLS与4个register tokens单独缓存并从patch统计剔除。

| 问题 | 裁决 | 直接观察 |
|---|---|---|
| 1 | INCONCLUSIVE | Glycolytic muscle: both train-only scores decline at layers []; both increase at [11]. |
| 2 | INCONCLUSIVE | Layerwise deltas and block intervals are reported; no whole-encoder collapse inferred. |
| 3 | SUPPORTED | Layer23 linear crop Dice delta=-0.218372; remote FP-rate delta=+0.094149. |
| 4 | SUPPORTED | Descriptive within-source association only; contrast/gradient/position strata and partial ranks reported. |
| 5 | INCONCLUSIVE | Only a research rationale, never evidence that a proposed SSL constraint works. |

裁决是固定样本上的探索性诊断。问题1按两种train-only读出的四层变化是否多数同向判断，块bootstrap区间列在原表；问题4表示可观察的描述性关联。它们不提供独立acquisition的因果证据。

### D2同轨迹30k→100k：肌肉patch AUROC

| 层 | 读出 | 30k | 100k | 差值 | Z块bootstrap 95%区间 |
|---|---|---:|---:|---:|---|
| 5 | linear | 0.9609 | 0.9470 | -0.0138 | [-0.0196, -0.0089] |
| 5 | prototype | 0.1092 | 0.1116 | 0.0024 | [-0.0004, 0.0049] |
| 11 | linear | 0.8547 | 0.8901 | 0.0354 | [0.0201, 0.0508] |
| 11 | prototype | 0.9845 | 0.9898 | 0.0053 | [0.0024, 0.0091] |
| 17 | linear | 0.9602 | 0.9577 | -0.0026 | [-0.0104, 0.0076] |
| 17 | prototype | 0.9470 | 0.9891 | 0.0422 | [0.0201, 0.0658] |
| 23 | linear | 0.9811 | 0.9727 | -0.0083 | [-0.0212, -0.0008] |
| 23 | prototype | 0.5138 | 0.9105 | 0.3968 | [0.3536, 0.4501] |

### 相同随机视野上的肌肉分割

| 模型 | 读出 | Dice | Precision | Recall | 远端FP率 |
|---|---|---:|---:|---:|---:|
| P0 | 23 | 0.4635 | 0.6515 | 0.3597 | 0.0003 |
| P0 | Primus | 0.7440 | 0.9201 | 0.6245 | 0.0003 |
| D1-100k | 23 | 0.5498 | 0.6256 | 0.4904 | 0.0004 |
| D1-100k | Primus | 0.6475 | 0.9363 | 0.4948 | 0.0000 |
| D2-30k | 23 | 0.5090 | 0.5362 | 0.4845 | 0.0031 |
| D2-30k | Primus | 0.4785 | 0.7388 | 0.3539 | 0.0026 |
| D2-100k | 23 | 0.2906 | 0.1775 | 0.8019 | 0.0972 |
| D2-100k | Primus | 0.2018 | 0.1373 | 0.3804 | 0.0750 |
| Coverage-NR-100k | 23 | 0.0910 | 0.0483 | 0.7844 | 0.4838 |
| Coverage-NR-100k | Primus | 0.4182 | 0.5335 | 0.3439 | 0.0073 |

## 抽样、拟合与统计口径

抽样seed20261027；MV3 train全部156区域各2个随机视野，五个诊断来源各64个随机视野，另1个历史D2误检案例。历史案例不进入随机总体统计。所有坐标在首次特征提取前锁定，未用D2失败筛选随机样本。

所有特征以bfloat16 autocast计算、官方post-norm后统一保存为float16缓存；初始receipt称“精确表示BF16输出”不准确，原输出dtype未被捕获，另有cache_precision_note保留更正。相同量化配置用于全部模型，不将缓存视为全精度特征。

每层1x1 logistic probe只在MV3 train拟合，固定500 AdamW更新、lr0.03、weight decay0.001、source/class-balanced minibatch4096、阈值0.5。按train计算来源均衡channel mean/std和前景/背景原型；所有probe hash固定后才读取开发与MB1 GT。没有SSL或Primus训练。

前景为有效patch中GT占比≥0.5；远端背景为patch无GT且≥95%面积在同一标注XY平面距GT>180nm。所有远端背景指锁定随机视野中的全部合格patch，不是原始体积全域。随机远端背景每来源固定最多1024个；D2特定FP集合为这些远端背景中旧D2/100k Primus预测占比≥0.5，单列为事后有偏错误集合。

谱统计每模型/来源/层使用相同512个有效patch位置，3个固定子抽样重复；有效秩上限511。以中心化特征协方差的非零特征值计算有效秩、top1方差占比和归一化谱分布，另报类内/类间余弦、train中心化余弦、Fisher ratio、各向异性及总方差。三个子抽样不是三次训练或独立生物重复。没有对权重做SVD或修改。

周期性为固定64×64 Hann窗在4–32像素周期频带内的top4非DC功率比例。记录对比度、梯度能量、图像位置、周围GT密度和native spacing/FOV；对比度分位和梯度分层阈值仅由train估计。报告来源内/volume内Spearman、固定分层及partial rank；同来源spacing常量不能估计尺度效应，不跨来源合并制造相关。

AUROC与部分主对照相关区间按volume_id和Z//8块重采样200次，仅作为这些已暴露数据中的不确定性描述。patch不是独立acquisition；OpenOrganelle多个crop共享acquisition。图像块重叠、单volume肌肉和已暴露MB1限制结论范围。

## 仍无法解释的现象

有效秩下降本身不能区分有害语义混淆与正常各向异性变化；固定线性读出与原版Primus具有不同容量和空间上下文。频谱关联亦可能反映未观测的结构、染色和物理尺度差异。即使本轮SUPPORTED，也不能认定周期纹理是因果机制，或Topo-Anchor/Gram一定有效。

## 唯一后续机制建议

优先研究冻结表征到分割读出的背景响应与校准失配；本轮证据未同时支持稳定表征退化及简单读出复现，不据此实施Topo-Anchor。先保留这个机制判断，不自动追加实验。

## 交付和停止

完整CSV在tables目录：全部背景集合的separability、spectrum与归一化谱分布、逐crop/逐来源readout指标、周期性控制变量和分层相关、D2逐层配对变化。缺少前景、远端背景或方差时保留NA，不按结果修改阈值或来源。特征缓存、线性权重及完整window响应仅保留独立服务器目录。

图表：AUROC_all_sources.png、effective_rank_all_sources.png、D2_paired_AUROC_delta.png；原图/GT/预测示例muscle_examples_REVIEW_ONLY.png为人工复核派生图，不提交原图到Git。代码commit、五模型SHA、环境、样本坐标、train-only读写证据随本轮归档。

Coverage-NR是2026-10-22已完成历史资产。Pilot A使用GPU1/2/3的确认空闲计算资源；GPU0禁用、网站保持。Pilot B/C、权重融合、权重SVD、Gram或新SSL均未启动。完成后停止。

