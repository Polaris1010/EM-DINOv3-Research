# 新窗口从这里接手

独立 private 仓库：[Polaris1010/EM-DINOv3-Research](https://github.com/Polaris1010/EM-DINOv3-Research)。本地：`E:\mira_website_demo\EM-DINOv3-Research`。使用此仓库的 main，不要把网站仓库当作研究仓库。

按顺序读取：

1. [AGENTS.md](AGENTS.md)：权限、GPU、协议和记录规则。
2. [PROJECT_STATE](docs/PROJECT_STATE.md) 与 [实验注册表](experiments/registry.yaml)：已核验研究事实。
3. [新窗口交接](docs/NEW_WINDOW_HANDOFF.md)、[每轮记录](docs/work_logs/README.md) 与 [GitHub交付](docs/GITHUB.md)：交付和未完事项。
4. [NEXT_ACTION](docs/NEXT_ACTION.md)：下一项科学问题；[历史Markdown索引](docs/history/INDEX.md)：历次原始记录。

可直接粘贴给下一个窗口：

> 请接手 private 仓库 Polaris1010/EM-DINOv3-Research 的 main。本地路径 E:\mira_website_demo\EM-DINOv3-Research。先读 START_HERE.md 和 AGENTS.md，再读当前状态、registry、交接与最新工作记录。按实际证据区分计划、执行和失败，不重复初始化。现有 MV-2 无 TTA 任务需只读核实最终状态；不要启动训练、推理、旧 watcher 或修改网站/环境；GPU 0 禁用。保留原版 Primus、九个 head 和固定协议。每轮更新工作记录及交接，将本轮代码、配置和 Markdown 提交并推送；汇报实际 commit、分支和 PR 状态。

仓库不包含服务器密码；认证使用已有安全凭据或交互输入。
