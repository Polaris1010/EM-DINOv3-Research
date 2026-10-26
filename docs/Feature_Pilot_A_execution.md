# Feature Mechanism Pilot A执行协议

用户于2026-10-27授权修订Pilot A，仅运行冻结特征诊断和轻量线性探针。Coverage-NR在2026-10-22完成，保持历史资产。独立目录/opt/data/3_117/tanghr/codex/feature_mechanism_pilot_a_20261027。

固定五模型P0/D1-100k/D2-30k/D2-100k/Coverage-NR-100k及官方EMA SHA，特征层[5,11,17,23]，官方post-norm，CLS/register单独保存。原MV输入256/native grid、灰度重复3通道和((x/255)-0.5)/0.25不变。

在特征之前锁定156训练区域各2个随机视野，五个诊断来源各64个随机视野；历史z150肌肉错误案例单列。开发与MB1 GT只在所有probe保存并固定SHA后离线读取。所有远端背景指锁定视野中的全部合格patch，另列随机远端背景和D2特定远端FP，不将事后错误集合当无偏总体。

每模型每层1025参数的1x1 logistic probe：500固定AdamW更新，train-only来源/类别均衡采样，lr0.03、weight decay0.001、batch4096，阈值0.5。均值/方差、前景/背景原型以及周期性分层cutoffs只用train，已在协议预先固定，不用MB1选参。

各来源谱统计同512个token、3个共同子抽样。所有有效秩、余弦、Fisher、AUROC均单独解释。周期性用固定64px Hann窗、4–32px频带top4功率份额；记录对比度、梯度、位置、native spacing和上下文GT密度，来源内/volume内、固定分层和partial rank相关分开。Z块bootstrap仅描述固定采集内不确定性，无独立patch p值或因果宣称。

只使用检查后空闲的GPU1/2/3，显存上限约每worker7GiB；GPU0不可见。数值smoke由本轮controller管理，避免触发其他任务未知PID保护。保留网站进程，不杀未知进程；新占用出现时暂停调度，失败只停本轮直接child。8小时上界，不自动新研究阶段。

结果输出Feature_Mechanism_Pilot_Report.md、CSV与科学图表；权重/特征cache留服务器。原图GT示例为review artifact不提交原始数据。最终逐问题标记SUPPORTED/CONTRADICTED/INCONCLUSIVE，独立PR，不自动合并。此文档为锁定配置，实际执行以run_index与服务器收据为准。
