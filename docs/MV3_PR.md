# MV3：完成多来源冻结Primus与三seed比较

旧少来源head不能说明充分多来源监督后的分割能力。本PR完成MV-Generalist-v2的156train/38dev区域、12来源训练：官方DINOv3 ViT-L/16冻结，原版Primus_Multiscale，[5,11,17,23]，TTA=OFF，G-P0/G-B100三个匹配decoder seeds各40k，保存10k/20k/40k。

来源等权Dice：P0 0.813741±0.000827，B100 0.826449±0.000832；同seed配对Δ+0.012707±0.001361，三个seed均正。9个来源三个seed均提高，Haberl基本持平；Guay/Müller均仍略低。最明显收益在OpenOrganelle/Han/Turner。最终报告包含所有来源的Dice、Precision、Recall、HD95、NSD、预测前景比例，逐区域配对、失败案例、训练曲线与18个checkpoint SHA。

当前数据协议仅对Han按用户明确指令忽略mask、整幅GT参与loss/采样/指标；官方test保留，不混合MV2/v1协议。旧预测在相同新heldout/GT上仅作描述性扩展收益参照，包含监督覆盖和20k→40k预算的共同改变。B100此前无标签暴露全量图像，结论限定transductive development；decoder seeds不增加独立acquisition。

六组40k及228条指标已实际完成。评分末段将串行15秒轮询改为GPU1/2/3各一worker、最多三进程且保守限制RAM；只改变执行调度，evaluator和训练/数据SHA不变，18个checkpoint及189条已有指标保持。GPU0未用，网站200、环境未改，历史26条及6条v1注册快照原样保留。

验证：17项CPU回归测试、registry/provenance、228条覆盖和混淆矩阵算术、12来源/3seed汇总、18个checkpoint元数据关联及三个seed初始化/采样匹配均通过。service已inactive/MainPID0，任务结束。下一项决策是MitoNet同图像GT对照，本轮未自动运行新实验。PR保持draft，未自动合并。
