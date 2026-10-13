# 新窗口交接

当前研究仓库与网站分开。首PR保存服务器只读盘点、26条实验记录、115份精选来源快照；本轮另补664份本地历史源码/配置/记录，其中76份Markdown。精选实际运行代码位于 `src/`，本地历史代码位于 `archive/research/`，历史Markdown位于 `docs/history/`。每份历史文件的来源、字节数和SHA位于 `evidence/history_file_index.json`；归档本身不证明实验已运行。

首PR交付进度与实际合并回执以 [GITHUB](GITHUB.md) 和 `evidence/github_delivery.json` 为准。旧 `evidence/github_publication.json` 是首次发布的历史快照，不是实时状态。

服务器 `root@192.168.203.97`，研究根目录 `/opt/data/3_117/tanghr`。具体环境、模型和路径见 [SERVER_PATHS](SERVER_PATHS.md)、[ENVIRONMENTS](ENVIRONMENTS.md) 和registry。密码不入库。

P0、B100k、B150k及MV-1九个head已核验；B150k是100k+50k continuation。原版Primus已与锁定upstream按AST比对。无标签预训练暴露包括开发/测试图像，因此结论限定transductive development。保留Stage2A/Stage2A2的失败及负面结果。

最近服务器快照来自2026-10-13 19:55:21香港时间：已有 `mv-dev-v2-notta-20261013.service` 正在运行，partial inferred507/scored251。这不是当前实时状态，也不能写成完成。本次补交代码和记录没有连接服务器或新增实验。

按 [NEXT_ACTION](NEXT_ACTION.md) 只读收取现有MV-2无TTA结果，先确认是否完成，再核对3模型×3seeds及每组188个条目的覆盖。保持encoder、九个head、原版Primus、TTA=OFF及全部推理协议。旧四翻转目录已废弃，保留档案，不用于正式汇总。

`src/` 和 `archive/research/` 的脚本含原服务器绝对路径、依赖关系及历史watcher。归档未改变代码，也未赋予启动权限；不要批量执行。可直接运行的本仓库CPU校验见README。

每轮记录在 `docs/work_logs/`。结束前补充实际操作、验证、commit/PR、未完事项；事实变化时更新PROJECT_STATE、registry及NEXT_ACTION。先核验当前Git状态和远程，再继续工作。
