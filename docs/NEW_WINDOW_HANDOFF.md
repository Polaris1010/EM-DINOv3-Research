# 新窗口交接

**当前接手MV-3（2026-10-14）：** 用户已授权多来源监督训练，读取 `docs/MV3_PROTOCOL.md`、最新工作记录和 `experiments/MV-3/run_index.json`。分支 `codex/mv3-generalist-training`；服务器新根目录 `/opt/data/3_117/tanghr/codex/mv_generalist_v1_20261014`，执行器只允许2 encoders×3 seeds×40k并在最终报告后停止。先核查现有服务/进度，不重复启动，不恢复旧watcher。smoke和完成状态以实际快照为准。

Han24本轮核验见Han24_mask_review_20261014.md与最新工作记录；原论文局部末端语义及已有GT-mask空间证据支持非零mask有效。三份并排照片在本地artifacts/mv3_control_20261014/Han24_review。未参考模型Dice，未改变其他来源或训练配置。

本地被动收取器只等待这次MV3终止，源码scripts/watch_mv3_results.py。查看artifacts/mv3_control_20261014/collector_pid.json、collector.log、collector_result.json；不要重复启动。完成时会收取并推送报告/指标/哈希，若工作区有其他改动则留在collected_results。服务器有界执行器不依赖本地窗口保持打开；本地collector不能访问时，直接从服务器本轮目录只读收取实际产物。

MV-2已完成1692条核心指标，不追加旧9个head的验证或诊断。当前train156/dev38区域、12来源，官方test/val保留，部分旧探索性dev已转train；Han非零mask有效。历史登记原样保留，新增MV3条目单列。既有“初始化不训练”约束已被本次明确训练任务取代；GPU0禁用、网站/环境/历史不可修改继续生效。

以下内容为上一轮交接，旧MV2运行中记录不再代表当前状态。

当前研究仓库与网站分开。首PR保存服务器只读盘点、26条实验记录、115份精选来源快照；本轮另补664份本地历史源码/配置/记录，其中76份Markdown。精选实际运行代码位于 `src/`，本地历史代码位于 `archive/research/`，历史Markdown位于 `docs/history/`。每份历史文件的来源、字节数和SHA位于 `evidence/history_file_index.json`；归档本身不证明实验已运行。

首PR交付进度与实际合并回执以 [GITHUB](GITHUB.md) 和 `evidence/github_delivery.json` 为准。旧 `evidence/github_publication.json` 是首次发布的历史快照，不是实时状态。

服务器 `root@192.168.203.97`，研究根目录 `/opt/data/3_117/tanghr`。具体环境、模型和路径见 [SERVER_PATHS](SERVER_PATHS.md)、[ENVIRONMENTS](ENVIRONMENTS.md) 和registry。密码不入库。

P0、B100k、B150k及MV-1九个head已核验；B150k是100k+50k continuation。原版Primus已与锁定upstream按AST比对。无标签预训练暴露包括开发/测试图像，因此结论限定transductive development。保留Stage2A/Stage2A2的失败及负面结果。

最近服务器快照来自2026-10-13 19:55:21香港时间：已有 `mv-dev-v2-notta-20261013.service` 正在运行，partial inferred507/scored251。这不是当前实时状态，也不能写成完成。本次补交代码和记录没有连接服务器或新增实验。

按 [NEXT_ACTION](NEXT_ACTION.md) 只读收取现有MV-2无TTA结果，先确认是否完成，再核对3模型×3seeds及每组188个条目的覆盖。保持encoder、九个head、原版Primus、TTA=OFF及全部推理协议。旧四翻转目录已废弃，保留档案，不用于正式汇总。

`src/` 和 `archive/research/` 的脚本含原服务器绝对路径、依赖关系及历史watcher。归档未改变代码，也未赋予启动权限；不要批量执行。可直接运行的本仓库CPU校验见README。

每轮记录在 `docs/work_logs/`。结束前补充实际操作、验证、commit/PR、未完事项；事实变化时更新PROJECT_STATE、registry及NEXT_ACTION。先核验当前Git状态和远程，再继续工作。
