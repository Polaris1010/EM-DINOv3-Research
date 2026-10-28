# Pilot C：固定45k Vanilla / Barrier配对实验

**研究决定：不将当前Barrier配置继续作为MICCAI方法主线。** C1相对同预算C0的开发集Dice下降0.003790、NSD下降0.003432；MB-1七组Dice增加0.003175，但Glycolytic muscle的Dice下降0.075061、AP下降0.051623、Recall下降0.090534。相对Pilot B中点，开发集和MB-1均值分别下降0.008421和0.018905。当前配置没有取得兼顾内部开发表现与跨域肌肉分割的净收益。

这是一项已完成的负结果：两组各从原D2/30k完整状态更新15k至45k，各自完成原版冻结Primus40k、Linear23和全部固定评价。没有按中途成绩取消Primus、调阈值或改变训练参数。

## 五系统总结果

以下Primus均为decoder seed20261014、固定step40k、TTA OFF；历史三系统复用同seed结果，**不是三seed均值**。MV3先按区域求来源均值，再对12来源等权；MB-1先对二维图像/切片求组内均值，再对七组等权。两套集合分别报告，不混成一个总分。

| 系统 | MV3 Dice | MV3 NSD@16nm | MB-1 Dice | 肌肉Dice |
|---|---:|---:|---:|---:|
| D2-30k | 0.829407 | 0.731270 | 0.461614 | 0.460026 |
| D2-100k | 0.841379 | 0.748817 | 0.475754 | 0.215677 |
| Trajectory-Lerp05 | 0.838631 | 0.745975 | 0.505823 | 0.468187 |
| C0_Vanilla_45k | 0.834000 | 0.741066 | 0.483744 | 0.462184 |
| C1_Barrier_45k | 0.830210 | 0.737635 | 0.486919 | 0.387123 |

| C1净变化 | MV3 Dice | MV3 NSD | MB-1 Dice | 肌肉Dice | 肌肉AP | 肌肉Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 相对C0 | -0.003790 | -0.003432 | +0.003175 | -0.075061 | -0.051623 | -0.090534 |
| 相对Midpoint | -0.008421 | -0.008341 | -0.018905 | -0.081063 | -0.046441 | -0.106481 |

主因果对照是同45k预算的C1−C0。中点与30k/100k是实用参照，预训练预算不同，不作为严格等预算loss消融。

## 开发集逐来源损益

C1相对C0在7/12来源Dice提高、5/12下降；最大改善为Han +0.012336，最大退化为Turner −0.064686。多数正向变化较小，Turner退化使来源等权均值下降。Han沿用用户确认的整幅GT协议，不读取annotation mask。

| 来源 | C0 | C1 | 中点 | C1−C0 Dice | C1−中点 Dice | C1−C0 NSD |
|---|---:|---:|---:|---:|---:|---:|
| casser20 | 0.935277 | 0.935719 | 0.935254 | +0.000442 | +0.000465 | +0.000414 |
| guay21 | 0.765414 | 0.765957 | 0.761066 | +0.000542 | +0.004891 | -0.000458 |
| haberl18 | 0.538618 | 0.540218 | 0.547271 | +0.001600 | -0.007053 | +0.004324 |
| han24 | 0.696995 | 0.709332 | 0.678697 | +0.012336 | +0.030634 | +0.005487 |
| jiang25 | 0.881015 | 0.881877 | 0.879882 | +0.000862 | +0.001995 | +0.002513 |
| kunduri22 | 0.952129 | 0.953032 | 0.952491 | +0.000902 | +0.000540 | +0.006291 |
| mekuc20 | 0.932948 | 0.933465 | 0.932114 | +0.000517 | +0.001351 | +0.000888 |
| muller21 | 0.895673 | 0.894360 | 0.896649 | -0.001312 | -0.002289 | -0.003770 |
| openorganelle | 0.789981 | 0.794689 | 0.805610 | +0.004707 | -0.010921 | -0.001108 |
| turner22 | 0.763735 | 0.699049 | 0.816389 | -0.064686 | -0.117341 | -0.051958 |
| wei20 | 0.927213 | 0.926325 | 0.928210 | -0.000888 | -0.001885 | -0.003290 |
| xiao18 | 0.928995 | 0.928496 | 0.929939 | -0.000499 | -0.001443 | -0.000517 |

OpenOrganelle相对C0提高0.004707，但仍低于中点0.010921；Guay仅提高0.000542，Müller下降0.001312。当前结果不支持跨来源统一改善。38区域同GT计数配对变化见[PilotC_paired_dev_deltas.csv](../experiments/Pilot-C2/PilotC_paired_dev_deltas.csv)。

## MB-1七组固定诊断

| Benchmark | C0 | C1 | 中点 | C1−C0 | C1−中点 |
|---|---:|---:|---:|---:|---:|
| c_elegans | 0.192216 | 0.238584 | 0.206498 | +0.046368 | +0.032085 |
| fly_brain | 0.551179 | 0.562738 | 0.655546 | +0.011559 | -0.092808 |
| glycolytic_muscle | 0.462184 | 0.387123 | 0.468187 | -0.075061 | -0.081063 |
| hela_cell | 0.509568 | 0.540318 | 0.514462 | +0.030751 | +0.025857 |
| lucchi_pp | 0.861104 | 0.861493 | 0.867514 | +0.000389 | -0.006021 |
| salivary_gland | 0.064750 | 0.071718 | 0.082040 | +0.006968 | -0.010322 |
| TEM-100 | 0.745204 | 0.746455 | 0.746516 | +0.001251 | -0.000061 |

C1在6/7组相对C0提高，C. elegans +0.046368、HeLa +0.030751较明显；肌肉−0.075061抵消了大部分增益。七组均值的+0.003175不能替代目标失败域的实际退化。C1相对中点仅C. elegans与HeLa提高，Fly brain下降0.092808、肌肉下降0.081063。Salivary gland仍很弱。

完整Dice、IoU、Precision、Recall、HD95 nm、NSD@16nm、预测前景比例及体素汇总IoU见[PilotC_MB1_per_dataset.csv](../experiments/Pilot-C2/PilotC_MB1_per_dataset.csv)；两套逐图CSV各2595行。TEM空间标定不足的nm指标保留空值，空预测导致的HD95=inf原样保留。体素汇总IoU仍来自逐层二维推理，不等同原论文3D实例流程。

## 肌肉：远端FP下降伴随Recall损失

| 系统 | Primus Dice | AP | AUROC | Recall | 远端背景FP率 |
|---|---:|---:|---:|---:|---:|
| D2-30k | 0.460026 | 0.517261 | 0.927059 | 0.356516 | 0.3441% |
| D2-100k | 0.215677 | 0.242346 | 0.698756 | 0.380833 | 7.6468% |
| Trajectory-Lerp05 | 0.468187 | 0.468882 | 0.859489 | 0.391853 | 0.5723% |
| C0_Vanilla_45k | 0.462184 | 0.474064 | 0.881961 | 0.375906 | 0.5241% |
| C1_Barrier_45k | 0.387123 | 0.422441 | 0.869546 | 0.285373 | 0.3426% |

C1远端FP率由0.5241%降到0.3426%（相对减少34.64%），但Recall由37.59%降到28.54%（−9.05个百分点），Dice下降7.51个百分点，AP也下降。这个方向不构成成功修复。

C0/45k肌肉Dice0.462184与30k的0.460026接近，远端FP0.5241%远低于100k的7.6468%；它没有复现100k严重Dice/FP退化。但C0的Primus AP已从30k的0.517261下降到0.474064，不能将45k描述为所有指标毫无退化。C1优于历史100k的肌肉成绩，不能据此归功于Barrier阻止了长期遗忘。

Primus AP/AUROC为完整302层逐层计算后的均值；远端FP率汇总全部距离GT>180nm的二维有效背景。下面Linear23使用另一个固定的64视野/来源集合、原生像素汇总；两个读出的绝对AP不能直接视为同一像素总体。

## 匹配Linear23

只在312个MV3-train视野拟合500步，seed20261027；归一化统计、source/class采样和初始化均固定。MB-1 GT只在拟合完成且probe SHA固定后用于评分。五个随机诊断来源各64视野，错误导向案例不混入总体。

| 来源 | C0 AP | C1 AP | ΔAP | C0 Dice | C1 Dice | ΔDice |
|---|---:|---:|---:|---:|---:|---:|
| openorganelle | 0.855726 | 0.837930 | -0.017796 | 0.753557 | 0.717886 | -0.035672 |
| guay21 | 0.420961 | 0.341392 | -0.079569 | 0.450540 | 0.386988 | -0.063552 |
| muller21 | 0.557136 | 0.558075 | +0.000939 | 0.533593 | 0.538944 | +0.005351 |
| glycolytic_muscle | 0.592327 | 0.568414 | -0.023913 | 0.603283 | 0.562401 | -0.040883 |
| hela_cell | 0.536928 | 0.491348 | -0.045580 | 0.498217 | 0.472297 | -0.025919 |

C1肌肉Linear23 AP下降0.023913、Dice下降0.040883，方向与Primus一致。其Linear远端FP反而从0.3428%升到0.9599%，Recall略升；Primus则FP与Recall同时下降。两类读出的错误变化不同，不能将现象统一解释成Encoder背景响应被普遍抑制。Probe各自拟合，排序变化仍包含表征与读出方向的交互。全部五系统结果及新模型谱统计均已保存，未将AP用作Primus门槛。

## 实际执行配置与完整恢复

两组都恢复原D2/30k完整checkpoint：student、EMA teacher、DINO/iBOT状态、AdamW和三rank的Python/NumPy/Torch/CUDA RNG；optimizer step30000→45000。原训练没有独立scheduler或GradScaler对象：按global step继承真实100k分段LR函数、WD0.04及组倍率，继续BF16 autocast。没有重置optimizer或新建15k cosine。global batch48、global256/local112×8、原增强、seed0及DINO+iBOT+KoLeo cfg0.1（含原crop scaling）保持。数据是同一个D2图像池与排除清单，MitoVerse153体积/34143合法Z、CEM1425433图像，未读CEM监督标签。

C1唯一新增项固定λ1、margin0.05、bottom50%、Chebyshev>2、Layer23 post-norm patch token；每global视野取anchor非近邻相似度中位数及以下的有序patch对。CLS/register排除，affinity与hinge/reduction为FP32：

`L_Barrier = sum_selected max(0, A_student - stopgrad(A_anchor) - 0.05)^2 / selected_count`

固定30k EMA anchor对同一global图像做unmasked eval/no-grad前向；student保留原iBOT mask和DropPath。Barrier加到同一个总loss，仅一次实际backward和optimizer update。正式模式没有额外Student诊断forward或VJP；raw-loss梯度hook每步确认upstream=1，anchor始终无梯度且未改变。patch总梯度范数只作健康记录，不冒充Barrier独立梯度范数。

45,000组rank-step全部global/local crop、mask/indices/weights、采样链、模型前后RNG、LR/WD逐项相等；每组恰好15k更新，没有恢复重放。C0首20步loss与C-1逐值相同；C1生产/诊断预检首20步loss最大差5.63e-5，如实记录，不宣称数值逐位相等。数学核心与已接受C-1文件SHA相同。

两head沿用MV-Generalist-v2的156 train/38 dev，原版Primus_Multiscale，层[5,11,17,23]，encoder全程冻结，同head初始化及全部40k采样hash相等。seed20261014、原AdamW与40k日程、native256/stride128/threshold0.5、TTA OFF，固定step40k。除Han整幅GT规则外，其他有效区保持。没有Gram/HRA/TopoCL/LAST或其他loss、decoder变体。

运行源码commit `b3c101680eaa451b74963d66433ba70bd6077adc`，13个运行源码文件及protocol SHA已在结束后重核。生产GPU集成测试通过；首次测试后的GPU释放延迟导致调度零更新退出，随后仅修复有界idle等待，原始失败记录保留，未增加正式预算。

## 耗时、显存与资产

| 模型 | SSL墙钟 | SSL img/s | 全程step中位（rank0） | 每卡peak allocated | Primus40k |
|---|---:|---:|---:|---:|---:|
| C0_Vanilla_45k | 3.249 h | 61.55 | 754.1 ms | 9.53 GiB | 37.25 min |
| C1_Barrier_45k | 3.495 h | 57.22 | 807.3 ms | 10.72 GiB | 37.26 min |

C1 SSL实际增加7.57%墙钟（约14.76分钟）；每步中位数统计来自rank0，整体吞吐含checkpoint开销。两head并行。全流水线28979.45秒（8.05小时），新目录共177.44GiB，含全部新恢复checkpoint和预测缓存。物理GPU1/2/3执行训练、推理与加速评分，CPU仅负责I/O、调度和汇总。

rank0 Barrier loss中位0.001521，激活比例中位10.76%；最后1000步均值分别0.001250和8.83%。所有rank/step数值有限；原clip30保持，rank0 clip前梯度范数峰值C0=114.86、C1=309.42，均通过同一裁剪流程。没有因这些观测临时改配置。

原30k完整checkpoint SHA256 `cce40d90b9e4f03eae9231f7516cd1a81d49e8b369db9380dbb3b30f7d82967f`。两新完整checkpoint、正式EMA、head与probe的真实路径映射和完整SHA见[artifact_hashes.json](../evidence/Pilot-C2/artifact_hashes.json)；原始大文件仅留研究服务器，不放入Git。

C0 EMA SHA256 `54a20de4c373bec12af858c810b4edf7eb337aaff62dd3631ff3a00b8a69e021`；C1 EMA SHA256 `bd38b158fbd16cddee6863b5092a57a27a127616a82d443d4afd016886219a38`。原30k完整文件/EMA前后SHA不变；其他历史checkpoint、指标及已有PR均未修改。

## 科学裁决与停止

**SUPPORTED：** 固定45k预算下，这个Barrier配置带来多来源权衡：六组MB-1小幅改善、肌肉质量和开发均值下降。肌肉Linear与Primus的Dice/AP下降方向一致。

**CONTRADICTED（本配置、本协议）：** “Barrier兼顾同预算Vanilla的开发表现和肌肉修复，且优于固定轨迹中点”的实用主张。

**INCONCLUSIVE：** 长期关系遗忘的具体机制、100k时能否防止严重退化、其他损失定义或参数的作用。Student包含iBOT mask/DropPath，anchor为unmasked eval，激活不能归因为遗忘；即使某些来源改善，也不能单凭此对照确定机制。只有一个匹配decoder seed；MB-1已暴露并参与前期决策，作为开发压力诊断，不称盲测或独立泛化证明。

**明确建议：结束当前Barrier配置，不调整λ/margin、不自动延长到100k，不继续把它作为MICCAI主方法。保留C0/45k和Pilot B中点作为后续研究的强基线，避免把早停本身已有的肌肉表现归功于新loss。** 本轮不再增加诊断或方法阶段。

截至2026-10-28 08:43 HK，服务inactive、退出0，所有新训练/评分任务已结束，网站HTTP200、GPU0未用。完成源码、指标、Markdown及独立draft PR交付后STOP；不自动合并、不运行任何后续训练。

复核：337份收集文件逐个SHA验证；190条五系统开发区域、60条来源、35条MB-1组、25条Linear、1510条肌肉逐层及两套各2595条逐图记录齐备；来源/组均值独立重算与summary误差<1e-12。配对、训练与停止证据保存在evidence/Pilot-C2。提交前CPU检查记录保存在cpu_checks.json。
