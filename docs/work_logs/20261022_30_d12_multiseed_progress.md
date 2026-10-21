# 多seed进度复核

2026-10-22 02:56香港实测：多seed新增head完成4/8（seed20261015四encoder均40k）；seed20261016的D1/D2-30k均39500、D1-100k=1000、D2-100k排队。服务active，GPU1/2/3均100%，开发完整指标与MB1均尚未开始，三seed均值尚无。Coverage-NR继续47490/100k、网站200。只继续现有有界服务，禁止重复启动。

## 已完成head

D1-30k / D2-30k / D1-100k / D2-100k，seed20261015均已training_complete，encoder未变；用时分别约63.8、63.8、63.8、53.9分钟。40k checkpoint SHA继承现场progress记录，本轮未重新哈希大文件。

## 当前运行

seed20261016：两组30k head=39500/40000；D1-100k=1000/40000；D2-100k未启动。GPU1总用约14.3GiB，GPU2/3约18.6/18.2GiB，利用率均100%。网站HTTP200，GPU0无本轮任务。控制器日志无错误。按现场吞吐估计，剩余head训练约1小时，之后仍需38区域全量推理、固定七组MB1和exact距离评分，不能把head完成等同于全部评测完成。

新seed完整开发指标0条、MB1完成0组；40k内置monitor为固定子切片检查，不作为全38区域最终成绩。现有旧seed结果保留，尚不计算/发布不完整的三seed均值。

## 交付

真实快照：evidence/D12-three-seeds/progress_20261021T185634Z.json；更新8个新增head登记，保留原59条记录不变。原代码/Markdown已推送draft PR12；本轮继续提交进度与交接。终态收取器进程仍在运行，未出现terminal_delivery/collector_error；不改参数、不新启动任务。
