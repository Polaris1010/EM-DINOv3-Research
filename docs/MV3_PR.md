# MV3: matched multisource frozen Primus generalist training

MV2用只在4体积/3来源监督的旧head评测12来源，无法回答充分多来源监督后DINOv3的适配能力。此PR建立MV-Generalist-v1并按共同40k预算训练G-P0/G-B100的新原版Primus head；不再扩展旧9个head，不增加SSL模块或全面数据审计。

新train156/dev38区域，均覆盖12来源；保留正式test和已映射官方validation，将130个OpenOrganelle探索性区域转train，按现有sample组留出22 dev。Han沿用已知非零末端mask有效规则；当前train/dev不交叉，单acquisition的空间holdout限制明示。MV1/MV2原文件和26条历史注册记录原样保留，新增6条MV3记录。

两组冻结encoder+官方原版Primus，[5,11,17,23]，TTA OFF，共同seed/初始化/source-aware采样/预算，固定40k选模，保存10k/20k/40k。首seed完整开发验证后再补另外两seeds。新目录中独立保存weights、预测、每来源指标、配对变化、前景比例和失败案例；最终报告后停止，不自动启动MitoNet或新预训练。

验证：9项CPU测试、registry/provenance检查与train/dev/test边界检查通过；真实GPU smoke和后续训练进度见experiments/MV-3、带时间戳evidence及本轮工作日志。计划预算和脚本存在不等于训练完成；最终模型结果尚未提交时不宣称新Dice收益。首次启动遇不可见GPU0宿主机PID，在模型训练前修复，未操作GPU0进程。

服务器既有环境和网站不改动；仅物理GPU1/2/3。仓库提交代码、配置、元数据和Markdown，weights/预测/原始图像仍留服务器。每轮交接从START_HERE读取当前分支和最新实际状态。
