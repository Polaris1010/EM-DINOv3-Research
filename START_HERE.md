> MV4已完成并停止（2026-10-14香港时间23:15:04核验）。同38开发区域/12来源，来源等权Dice：P0 0.813741、B100 0.826449、官方MitoNet 0.610209；B100在8/12来源更高。当前分支codex/mv4-official-mitonet，独立draft PR3，base为未合并MV3分支。先读docs/MV4_final_report.md、experiments/MV-4/run_index.json和work_logs/20261014_08_mv4_official_mitonet.md。服务inactive/MainPID0，指标266行核验PASS；不重启MV3/MV4或旧watcher，不自动启动新实验。

# 当前交付：MV4已完成

完整报告：[MV4_final_report.md](docs/MV4_final_report.md)。G-P0/G-B100复用全部三个decoder seeds，没有重训/重推；MitoNet使用固定官方权重、默认整图二维推理、TTA OFF，实例输出保留服务器。Han整幅GT继续如实披露，Turner nm指标保留spacing限制；这是监督数据和训练暴露不同的完整系统开发对照。

实际运行代码db2ae4e；完整指标experiments/MV-4/MV4_per_region_metrics.csv，来源差距MV4_source_deltas.csv，最终校验MV4_results_verification.json。服务器根/实例路径见run_index.json和最终报告。代码、指标、本轮Markdown及交接随PR3提交推送；PR未合并，不假设main已包含本轮。

下一项仅为决策：优先考虑跨来源稳健性，关注Guay/Müller仍存在的B100相对P0退化，同时保留MitoNet在Casser/Mekuc/Turner/Wei的优势作为实际改进目标。用户后续明确授权前停止所有实验；不新增审计、诊断、预训练或延长head训练。

## 以下保留MV3及更早阶段的历史状态/计划

> **MV3 v2已完成并停止。** 新窗口先读docs/MV3_final_report.md、docs/NEW_WINDOW_HANDOFF.md和最新工作记录；六组40k、228条指标、18份checkpoint已完成，禁止重复启动或自动开始新实验。

# 新窗口从这里接手

独立 private 仓库：[Polaris1010/EM-DINOv3-Research](https://github.com/Polaris1010/EM-DINOv3-Research)。本地：`E:\mira_website_demo\EM-DINOv3-Research`。本轮完成结果在 codex/mv3-generalist-training（draft PR2）；先读该分支，不假定main已合并本轮结果。不要把网站仓库当作研究仓库。

按顺序读取：

1. [AGENTS.md](AGENTS.md)：权限、GPU、协议和记录规则。
2. [PROJECT_STATE](docs/PROJECT_STATE.md) 与 [实验注册表](experiments/registry.yaml)：已核验研究事实。
3. [新窗口交接](docs/NEW_WINDOW_HANDOFF.md)、[每轮记录](docs/work_logs/README.md) 与 [GitHub交付](docs/GITHUB.md)：交付和未完事项。
4. [NEXT_ACTION](docs/NEXT_ACTION.md)：下一项科学问题；[历史Markdown索引](docs/history/INDEX.md)：历次原始记录。

可直接粘贴给下一个窗口：

> 请接手 private 仓库 Polaris1010/EM-DINOv3-Research，本地 E:\mira_website_demo\EM-DINOv3-Research，当前MV3分支codex/mv3-generalist-training，现用MV-Generalist-v2；Han按用户明确指令忽略mask、整幅GT参与loss/采样/评测，旧v1已停止保留。读取experiments/MV-3/v2/run_index.json。先读START_HERE、AGENTS、MV3_PROTOCOL、最新工作记录、当前状态、registry和交接。用户已明确授权MV3：仅G-P0/G-B100 frozen encoder+original Primus，共同40k预算、3匹配decoder seeds、TTA=OFF，首seed全来源指标后再补另两seed。先查现有MV3服务/进度，继续同一已授权任务，不重复启动。MV2已完成，不追加旧9个head验证或新审计/SSL模块；GPU0禁用，不改网站/环境/历史。每轮代码和Markdown提交推送，最终输出MV3_final_report.md并停止。

仓库不包含服务器密码；认证使用已有安全凭据或交互输入。
