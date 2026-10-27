# Pilot A2执行记录

用户明确批准执行，独立分支以2a2a347为base，历史Pilot A及设计保持。先读锁定设计/父实验源码/记录，重连服务器确认GPU3空闲、网站200、父实验complete、两组概率/probe SHA及样本SHA一致。复用本地Anaconda的SSH库，无依赖升级。

本轮只估计两个全局偏置，不重训模型。按用户最新顺序，原始评分进程先读诊断GT算指标；独立fit进程禁止读取diagnostic GT、MB1路径与baseline文件，算法和参数在原始评分前以源码commit锁定。最后冻结偏置统一评分；不将分数变化归因encoder共模漂移。

源码实现GPU exact tie-aware AUROC/AP、同像素mask/Brier、自然类别比例source-equal标量BCE、固定200次Z块配对bootstrap。权重、特征、预测和父指标只读；输入前后SHA验证。代码中的数值smoke使用人工fixture，不碰历史GT。当前为准备事实，不将计划写为已执行；启动/结果/验证后续追加。

## 实际完成

执行源码225e3c6，服务pilot-a2-readout-calibration-20261027.service，物理GPU3。GPU数值smoke通过；baseline2.42秒、标量拟合7.57秒，整个有界服务50.22秒完成并inactive/MainPID0/退出0。11份父输入前后SHA一致，网站200。收取24份小文件并逐SHA核验，无新模型或额外超参数运行。

偏置b30=-1.63222310、b100=-1.60702706，仅312train视野、12来源自然像素比例source-equal BCE；残差约1e-12。原始评分和fit隔离，fit明确禁止读取评测标签与baseline文件。

同像素肌肉AUROC下降0.022186、AP下降0.181580；100k偏置后Dice+0.075991，远端FP相对约少80.11%，Recall下降0.403359；其余四来源Dice点估计均下降。双校准差距改善R约0.2922，其中约74%来自30k退化，不能称完全恢复。决定关闭全局偏置主修复方向，不以此启动空间共模方法。完整数据与结论见正式报告。

收尾验证：24份服务器小文件SHA匹配；1280条视野、20条聚合、20条排序、85项配对及区间复算一致；baseline原始计数与Pilot A精确一致；60条历史registry未变。CPU检查registry61生成一致、validate_repository PASS（2207文本）、unittest51项/3平台skip为OK、diff检查PASS。三张统计图完成预览。报告/指标/代码/本轮Markdown随本次结果commit交付，随后核实独立PR链接。

Git交付已核实：结果commit792d0d90db869de665cd890664710679152caee4已推送；独立draft PR14，base为codex/feature-mechanism-pilot-a，未合并。PR创建回执归档GitHub_PR_created.json。本次补交仅记录实际链接和交付状态，所有实测CSV、偏置及源码结果保持。完成后停止。
