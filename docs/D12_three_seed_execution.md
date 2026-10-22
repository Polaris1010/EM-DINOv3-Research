# D1/D2 三 decoder seed 重复实验

用户于2026-10-22授权立即补齐D1/D2的三seed比较。历史30k、100k四个encoder各只有decoder seed20261014；本轮各补20261015、20261016，共8个新head。旧seed指标与预测直接复用，不重新训练SSL，不改变监督划分。

固定MV-Generalist-v2（156 train、38 dev、12来源）；冻结ViT-L/16、原版Primus_Multiscale、[5,11,17,23]、40k steps，10k/20k/40k保存、固定40k选模、原预处理/采样/loss/optimizer/推理tile16、TTA OFF。Han整幅GT协议沿用。四个EMA路径和SHA从已提交registry读取，启动前重新计算完整SHA。

## 并行执行

独立服务器根`/opt/data/3_117/tanghr/codex/d12_three_seeds_20261022`；服务`d12-three-seeds-20261022.service`。只使用物理GPU1/2/3，每卡一个head进程，CUDA allocator限制40%总显存，保留既有Coverage-NR与网站进程。先运行原版12来源loader/loss/冻结/存档smoke，并验证原tile16峰值，再启动正式训练。

现有Coverage-NR监督器会将任何新GPU进程判为外来任务。新服务临时接管运行健康/网站/心跳检查，仅SIGSTOP原Python监督器，**不暂停、不重启torchrun或任一SSL rank**，不改变训练权重、随机状态或配置。新CUDA任务退出后SIGCONT原监督器。systemd control-group清理后ExecStopPost也执行恢复，防止错误退出留下挂起监督器。运行身份含PID创建时间，防PID复用。接管最长36h，全服务48h，无自动重启。原STATUS在接管期间可能陈旧，实际SSL进度读formal/progress.json，新服务STATUS同时收录它。

先完成8次head训练，再全38区域推理、七组MB1固定诊断；之后恢复原监督器，使用原validation.distances与MONAI的CPU exact EDT路径完成304个新区域距离指标，减少VRAM争用。CPU/GPU exact EDT仅计算后端不同，已用一个历史小体积预测确认CPU/GPU HD95和NSD逐值一致；额外Han全体积CPU检查已取消；不更改阈值、尺度或TTA。

## 汇总与停止

最终12个run共456条区域指标、31140条MB1逐图指标。先各来源内平均，再12来源等权；MB1先组内平均，再7组等权。输出各encoder三个seed的均值、样本标准差（ddof=1）、范围；同seed配对D2−D1（30k/100k）及100k−30k（D1/D2），保留逐来源、逐benchmark与前景比例变化。非有限距离保留，不以有限值均值掩盖失败。

输出D12_three_seed_report.md、summary、paired_summary、sources_summary、datasets_summary、seed_macros、checkpoint_hashes及原始逐区域/逐图CSV。大逐图表保留服务器并分片收取，禁止checkpoint/预测进Git。代码、运行快照、指标、Markdown与交接提交推送，服务完成后停止。

这是固定划分的decoder seed重复，衡量head初始化/监督采样波动；不是三折交叉验证，也不是三次SSL seed。MB1仍是已暴露开发诊断集。尚无三seed结果时不将历史单seed均值写成新结论。

实际启动证据：2026-10-22 00:57香港，首批3个head均100/40000，原SSL42340且继续。见experiments/D12-three-seeds/launch_snapshot.json。

## 2026-10-22 GPU距离评分接续

2026-10-22 08:25香港：已按用户要求将剩余HD95/NSD切到GPU1/2/3，实测286/304条完成；新服务d12-three-seeds-gpu-score-20261022.service active，只接续缺失距离评分。保留277条既有指标SHA不变，无训练/推理重跑。Han CPU实测173–182秒，GPU21–22秒（同体积不同decoder seed，非严格同预测速度基准）。全距离场数值对照误差0，MONAI HD95/NSD一致。Coverage-NR继续60440、网站200，GPU0未用。最终完整报告待评分结束，禁止重启旧服务。

实际指标定义、物理spacing、预测/GT与MONAI函数未变。超显存的全局float64距离场在主存保存，按完整轴分批执行原CUDA kernel，再全局传递到下一轴，禁止局部空间截断。每作业数值验证及真实后端留档。原训练执行代码86492fc不变；评分加速39a56ce、只评分接续682574d。
