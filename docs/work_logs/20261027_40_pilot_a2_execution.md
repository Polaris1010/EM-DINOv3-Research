# Pilot A2执行记录

用户明确批准执行，独立分支以2a2a347为base，历史Pilot A及设计保持。先读锁定设计/父实验源码/记录，重连服务器确认GPU3空闲、网站200、父实验complete、两组概率/probe SHA及样本SHA一致。复用本地Anaconda的SSH库，无依赖升级。

本轮只估计两个全局偏置，不重训模型。按用户最新顺序，原始评分进程先读诊断GT算指标；独立fit进程禁止读取diagnostic GT、MB1路径与baseline文件，算法和参数在原始评分前以源码commit锁定。最后冻结偏置统一评分；不将分数变化归因encoder共模漂移。

源码实现GPU exact tie-aware AUROC/AP、同像素mask/Brier、自然类别比例source-equal标量BCE、固定200次Z块配对bootstrap。权重、特征、预测和父指标只读；输入前后SHA验证。代码中的数值smoke使用人工fixture，不碰历史GT。当前为准备事实，不将计划写为已执行；启动/结果/验证后续追加。
