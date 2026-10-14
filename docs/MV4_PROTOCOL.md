# MV-4：Official MitoNet vs DINOv3 Generalist

用户2026-10-14明确授权本轮，继承MV3完成状态并启动独立官方MitoNet对照。研究问题：官方MitoNet在同38开发区域上的来源等权Dice是多少，在哪些来源/目标上与G-B100存在差距？不训练任何模型、不增加诊断或SSL任务。

输入为不可变MV-Generalist-v2（156监督训练区域的历史协议、38开发区域/12来源）。G-P0/G-B100直接复用三个seed的40k预测与228条指标，不选择最佳seed。官方MitoNet_v1使用Zenodo6861565发布权重，文件221241200 bytes，发布者MD5 `b56fce28d877a3c35b7ca970cec73bc9`，校验后另计算SHA256。权重不入Git。

官方代码入口为empanada/empanada-napari，使用PyPI官方empanada-dl0.1.7发布包及hash锁定的Engine2d、Preprocessor原定义。无Qt入口只通过AST原样编译所需定义；没有修改网络、实例分离、归一化或阈值。记录整个原文件SHA及官方来源地址；未知upstream Git commit不凭空填写。

采用官方二维batch工作流：原生XY尺度、scale1、padding16、confidence0.5、center0.1、NMS3、label divisor10000、coarse boundaries、fill holes OFF、TTA OFF，采用官方默认整图路径`tile_size=0`。最大4096×4096图像已通过GPU3 smoke。最初tile1280尝试在官方Tiler后处理处因兼容错误失败，尚无正式预测或GT指标；旧配置与失败证据完整归档，不用GT调节。所有原生Z都推理；GT/annotation mask值不进入推理，GT box禁止使用。完整配置见[MV-4_protocol.json](../configs/evaluation/MV-4_protocol.json)。

仅评测读取与MV3相同GT、有效区与native physical spacing。Han沿用用户“取消mask限制、整幅GT”指令；其他来源不改。Dice/Precision/Recall用相同有效像素，HD95 nm/NSD@16nm调用原MV2距离实现，不重写指标。来源内等区域，12来源等权；DINOv3先平均三个decoder seeds，MitoNet为固定官方系统。undefined/inf指标保留。

保留官方逐层panoptic输出、跨层ID不重复的逐层instance输出和semantic foreground；不将逐层ID称为3D追踪，不用二值GT连通分量冒充实例GT。典型错误图在推理完成后选取，每来源最低MitoNet Dice区域及最大FP+FN平面；只做结果展示，不反馈推理。

这是完整系统竞争力开发对照。官方MitoNet监督训练数据/预算不同，具体与MitoVerse的重叠尚未核验；B100有transductive无标签暴露；三decoder seeds不代表独立acquisition。Han完整性按用户确认处理，Turner物理spacing继续UNVERIFIED并附限制。严格隔离测试属于后续论文任务，不阻塞当前对照。

服务器独立目录 `/opt/data/3_117/tanghr/codex/mv4_official_mitonet_20261014`，MitoNet推理使用独立venv、NumPy1.26.4/Numba0.59.1兼容官方旧发布包；额外包只装到该目录。评分与报告继续用原meddinov3环境，保留MV3原始NumPy2.4.4及MONAI/EDT实现；既有环境未修改。物理GPU0禁止，仅1/2/3各一个worker。保留网站和未知进程。最多24小时、唯一service和锁；仅本轮38区域推理/评分及报告，完成后停止，不自动新实验。

最终产物：`MV4_final_report.md`、38条MitoNet指标+228条复用指标、逐来源表、逐区域配对差、官方权重/源码/配置哈希、原始实例资产位置、12来源典型图、实际环境和保护MV3输入回执。代码/配置/小表和每轮Markdown一并commit/push；大数组/权重保留服务器。
