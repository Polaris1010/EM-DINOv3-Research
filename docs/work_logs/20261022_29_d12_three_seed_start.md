# D1/D2 三seed执行记录

用户授权补齐四个encoder的固定划分三decoder-seed重复。读取真实checkpoint登记和seed20261014完成记录，当前每组确实只有单seed。新增两个seed各40k，共8次head，旧seed复用。独立根、原版trainer/evaluator、40%进程显存限制与Coverage-NR监督接管方案已编码。CPU55项通过（3项已有skip）；正式启动及实测进度在本记录后追加。

## 实际启动

UTC2026-10-21T16:57:27 / 香港2026-10-22 00:57：单一服务active/MainPID4154944；smoke通过，首批3个head均100/40000，loss有限、同step采样hash一致；物理GPU1/2/3总用量14623/14623/14162MiB，GPU0未新增本轮进程。原SSL rank PID3930786/87/88不变，Coverage-NR前进至42340，网站200。原监督器临时移交健康检查，新执行器维护心跳，SSL未重启。代码86492fc。

部署连接曾同时读取同一队列；已停止本轮重复的本地SSH桥，仅保留原连接。原CPU检查留下的已知部署PID4153160及shell4153159已核实身份并清理后，才单独启动唯一systemd服务。自动审核曾拒绝在旧部署仍可能启动服务时另行启动；本轮通过先清理旧部署的安全方式完成。小体积历史预测的CPU/GPU exact指标逐值一致；Han全体积额外CPU检查取消，不阻塞既定训练。

最终三seed均值尚未产生。服务负责完整8次head、304条新增区域指标和20760条新增MB1逐图指标，合并历史seed后为456/31140。全量MB1表在服务器保留并输出12个无损小CSV分片以满足Git单文件限制。

## 运行及交付复核

香港01:01：首批三组均2600/40000；decoder初始化SHA一致17d5a0fc…、同step采样hash一致；loss中位数约0.26–0.31。每个head实际约3360MiB，tile16 smoke峰值allocated6379032576 / reserved9458155520 bytes，40%进程上限提供共享安全空间。原SSL至42550、PID不变，网站200。全部部署源码SHA与Git一致，实际trainer/loader/Primus SHA继承原版本。

代码及记录已推送1cec40c，draft PR12（未合并）。CPU检查55项、3项已有POSIX skip，registry生成与仓库校验通过，原59条登记逐条一致。新代码不含未知来源的scripts/phase0_complementarity.py。被动终态收取器已连接，实际返回COLLECTOR_WAITING；其仅收取、校验、提交推送，不启动实验。若后续切换或修改该分支，它将保存收取文件并停止自动提交，下一窗口按collector receipt补交。

## 终态收取

D1/D2三个decoder seed任务终态：complete，UTC 2026-10-22T01:20:53.479333+00:00。 详见docs/D12_three_seed_report.md与本轮STATUS；固定划分、四个已有encoder，没有新SSL。 本轮结束停止。Coverage-NR是独立已授权服务，按其既有任务继续，不重复启动。
