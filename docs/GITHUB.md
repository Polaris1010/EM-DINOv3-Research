# MV3三seed最终交付

最终报告与指标版本commit `ab625b43c52707aa865ed0d4ed428df9650687f3`已推送至private仓库的`codex/mv3-generalist-training`。[PR #2](https://github.com/Polaris1010/EM-DINOv3-Research/pull/2)为open draft，未合并；main不代表本轮最终结果。两项CI均success：[PR CI](https://github.com/Polaris1010/EM-DINOv3-Research/actions/runs/38541655448)、[push CI](https://github.com/Polaris1010/EM-DINOv3-Research/actions/runs/38541647912)。后续交付回执提交单独看对应head，不套用旧CI状态。

六组40k、228条完整指标、18个checkpoint及三seed核验已完成，报告为docs/MV3_final_report.md；本轮记录为docs/work_logs/20261014_07_mv3_completion.md。服务器service inactive、collector退出，网站200、GPU0未使用，没有自动新实验。实际GitHub回执见evidence/MV-3/v2/github_final_report_delivery_ab625b4.json。

以下为先前阶段的交付快照。

# 当前MV3交付

分支`codex/mv3-generalist-training`，当前[PR #2](https://github.com/Polaris1010/EM-DINOv3-Research/pull/2)为open draft，未合并。当前MV-Generalist-v2仅对Han按用户指令取消mask，整幅GT用于loss/采样/指标；其他配置保持。运行回执、每轮工作记录、状态、NEXT_ACTION与新窗口交接一并随代码提交推送。

部署代码commit `1e139513576984416c070a754eb62ad15f532e2f` 已推送，两个Actions均success：[PR CI](https://github.com/Polaris1010/EM-DINOv3-Research/actions/runs/38513854780)、[push CI](https://github.com/Polaris1010/EM-DINOv3-Research/actions/runs/38513851805)。后续回执提交的CI须看对应head，不能套用此成功记录。当前private核验与draft状态见evidence/MV-3/v2/github_publication_1e13951_20261014.json。

以下为初始化阶段保留的历史交付记录。

# GitHub交付

远程：[Polaris1010/EM-DINOv3-Research](https://github.com/Polaris1010/EM-DINOv3-Research)。可见性：**private**（API核验），owner Polaris1010，default branch main。

本地独立仓库 `E:\mira_website_demo\EM-DINOv3-Research`；首PR分支 `chore/bootstrap-research-registry`；主初始化commit `b2d5189`。该commit包含26条实验记录与115份来源快照，不含权重、数据、凭据或网站代码。浏览器账户授权经用户明确确认，凭据由既有Git Credential Manager管理，不写入本仓库。

首PR：[Archive research code, state registry and window handoff #1](https://github.com/Polaris1010/EM-DINOv3-Research/pull/1)，**已合并**，2026-10-13 20:29:24香港时间，merge commit `1baeacf0fde9e5cfac6876f550a9ddf9fe02cdc3`。本地已切换main并快进同步。

补交head `bd7f04c06c4e792520f3996d8de7744b0637dc8c` 包含664份本地历史档案（其中76份Markdown）、START_HERE、新窗口交接和每轮工作日志。该head的 [PR CI](https://github.com/Polaris1010/EM-DINOv3-Research/actions/runs/38397462657) 与 [push CI](https://github.com/Polaris1010/EM-DINOv3-Research/actions/runs/38397456993) 均success。后续commit实时状态以GitHub Actions为准，避免把历史CI成功套到新提交。

main已包含完整代码、26条实验记录、115份精选来源快照、664份本地历史档案和新窗口文件。新窗口从 [START_HERE](../START_HERE.md) 进入；[每轮日志](work_logs/README.md) 和 [历史索引](history/INDEX.md) 随代码保存。

实际合并回执见 `evidence/github_delivery.json`。旧 `evidence/github_publication.json` 保留首次发布时open状态，不是实时状态。本次纠正main只有scaffold的交付缺口；没有新增实验、修改服务器/网站/环境或上传凭据。下一项科学行动见NEXT_ACTION。
