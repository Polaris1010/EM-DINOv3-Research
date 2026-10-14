> **MV3 v2已完成并停止。** 新窗口先读docs/MV3_final_report.md、docs/NEW_WINDOW_HANDOFF.md和最新工作记录；六组40k、228条指标、18份checkpoint已完成，禁止重复启动或自动开始新实验。

# 新窗口从这里接手

独立 private 仓库：[Polaris1010/EM-DINOv3-Research](https://github.com/Polaris1010/EM-DINOv3-Research)。本地：`E:\mira_website_demo\EM-DINOv3-Research`。使用此仓库的 main，不要把网站仓库当作研究仓库。

按顺序读取：

1. [AGENTS.md](AGENTS.md)：权限、GPU、协议和记录规则。
2. [PROJECT_STATE](docs/PROJECT_STATE.md) 与 [实验注册表](experiments/registry.yaml)：已核验研究事实。
3. [新窗口交接](docs/NEW_WINDOW_HANDOFF.md)、[每轮记录](docs/work_logs/README.md) 与 [GitHub交付](docs/GITHUB.md)：交付和未完事项。
4. [NEXT_ACTION](docs/NEXT_ACTION.md)：下一项科学问题；[历史Markdown索引](docs/history/INDEX.md)：历次原始记录。

可直接粘贴给下一个窗口：

> 请接手 private 仓库 Polaris1010/EM-DINOv3-Research，本地 E:\mira_website_demo\EM-DINOv3-Research，当前MV3分支codex/mv3-generalist-training，现用MV-Generalist-v2；Han按用户明确指令忽略mask、整幅GT参与loss/采样/评测，旧v1已停止保留。读取experiments/MV-3/v2/run_index.json。先读START_HERE、AGENTS、MV3_PROTOCOL、最新工作记录、当前状态、registry和交接。用户已明确授权MV3：仅G-P0/G-B100 frozen encoder+original Primus，共同40k预算、3匹配decoder seeds、TTA=OFF，首seed全来源指标后再补另两seed。先查现有MV3服务/进度，继续同一已授权任务，不重复启动。MV2已完成，不追加旧9个head验证或新审计/SSL模块；GPU0禁用，不改网站/环境/历史。每轮代码和Markdown提交推送，最终输出MV3_final_report.md并停止。

仓库不包含服务器密码；认证使用已有安全凭据或交互输入。
