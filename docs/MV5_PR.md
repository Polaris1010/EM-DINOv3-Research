MitoNet-MV 已在 MV-Generalist-v2 的相同156训练区域上完成一个 seed 的40k实际 optimizer updates，并完成38开发区域/12来源的全部推理与评测。现有G-P0/G-B100三个seed和未适配官方MitoNet的266条记录原样复用，本轮新增38条，总计304条。

来源等权Dice：G-P0 0.813741，G-B100 0.826449，官方MitoNet 0.610209，MitoNet-MV 0.728833。适配增益+0.118624，B100对适配后差距+0.097616；当前开发结果中B100在12/12来源数值更高。Guay/Müller/Han明显恢复，但OpenOrganelle/Haberl等下降；不能把总体增益解释为全面改善。

使用官方scripted训练forward、PanopticLoss和中心/偏移/PointRend目标；源均衡native256 crop、batch4、一个seed20261014，10k/20k/40k保存，固定40k选模，TTA OFF。二值crop使用2D连通分量伪实例的规则明确披露。与冻结DINOv3相比，结构、上游数据、优化方式和监督曝光次数不同；这是当前适配配方下的同监督区域系统开发比较，不是严格等预算预训练消融，也不是MitoNet能力上限。

[最终报告](MV5_final_report.md)、[304条指标](../experiments/MV-5/MV5_per_region_metrics.csv)、[逐来源差距](../experiments/MV-5/MV5_source_deltas.csv)、[本轮运行记录](work_logs/20261015_11_mv5_mitonet_adaptation.md)及全部checkpoint哈希已收取。原结果提交09335403f53cdce18c661e35db07a484e375026e已推送；本次完成状态复核保留独立Markdown记录。

服务器服务inactive、MainPID=0、退出状态0；网站200，GPU0未用于研究任务。历史40条registry和266条旧指标不变。任务结束，没有启动新实验。本PR仍为draft，base为未合并MV4C分支，不自动合并。
