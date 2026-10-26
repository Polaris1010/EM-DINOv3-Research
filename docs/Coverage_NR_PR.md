Coverage-NR从官方P0完成100k SSL与固定seed20261014的40k冻结Primus，在38开发区域和七组2595张MB1上完成GPU推理/评分。保留的1425433张CEM图像ID覆盖100%，训练和评测已停止。

与相同head初始化、监督采样及GT的D2-100k首seed比较，开发Dice0.831821对0.841379，MB1为0.440027对0.475754。OpenOrganelle是开发差距主要来源；肌肉改善0.215677→0.393368，其余六组MB1下降。开发HD95改善、NSD略降。完整报告提供Dice/IoU/Precision/Recall/HD95/NSD/前景占比及逐来源差值，保留MB1空前景∞和TEM物理NA。

本轮变量是固定全覆盖配额、collection内无放回抽样和从P0连续100k LR；历史D2有不同集合分布和分段LR，作为系统参考。下游保持MV-Generalist-v2的156train/38dev、原版Primus、冻结encoder、40k选模与TTA OFF。NR仅一个decoder seed，D2三seed作为背景；MB1为开发诊断。

提交含采样/恢复/训练/推理和exact GPU评分源码、原始小型指标、checkpoint路径/SHA、完整报告及每轮Markdown。显存不足使用主存配合完整轴CUDA EDT。GPU校验误触未知PID保护的中断与94000完整状态恢复如实留档，原训练器和总预算保持。最终服务inactive/MainPID0/退出0，GPU0未用，网站200，历史资产保护检查通过。

验证：GPU距离场最大误差0、有/无ignore的HD95/NSD一致；最终CPU复核38区域/2595逐图GT、匹配初始化/采样和两级指标聚合；registry check、repository validator、unittest及git diff检查。详见docs/Coverage_NR_final_report.md与docs/work_logs/20261022_36_coverage_nr_complete.md。本轮完成，不自动新实验；保持draft，不合并。
