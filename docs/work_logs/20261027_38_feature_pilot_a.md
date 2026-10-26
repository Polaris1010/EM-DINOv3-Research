# 修订Pilot A：冻结表征与周期纹理诊断

用户授权仅Pilot A；禁止Pilot B/C、SSL与Primus训练。创建独立分支codex/feature-mechanism-pilot-a，以02a5ea0（Coverage-NR完成及公开可见性）为基线，不动SARR窗口。

已读AGENTS/交接/状态/registry、D12肌肉复核、Coverage-NR完成报告及原始配置，定位五正式encoder资产。服务器初始snapshot确认Coverage-NR complete、网站200；GPU3完全空闲，1/2仅网站少量context、util0，GPU0禁用。

代码将规划、train数据准备、冻结特征+train-only probes、offline GT评分、GPU谱/可分性/周期性统计分阶段；拟合阶段不读取诊断GT，全部probe SHA固定后才允许offline标签读取。详见协议与execution文档。此时尚未声称实验已启动或有结果。

本轮数字/图表应以实际CSV为依据，原图/大缓存/权重不提交。代码、锁定坐标、环境和结果最终推送独立PR；PR不自动合并。完成后停止。后续在本记录追加实际运行、检查与交付事实。

正式启动：源commit772f0fb；服务feature-mechanism-pilot-a-20261027.service，MainPID2993673。锁定633个视野（312train+320random eval+1historical case），样本SHA18622e1374467fe8a480f1b8a13e15370b51ac3386039d389bc4f55351c11458。五encoder实际SHA通过、GPU数值smoke通过；20个探针各500步均完成，训练坐标hash52112acd1aca39ebbf17bd943efa09cfe9bef677dda93f6f3328d79c02b2ba54。12来源均具有前景/背景训练patch。当前GPU统计分析运行，未声称有最终机制结论。

缓存精度核对：实际feature cache为float16；原receipt描述“精确表示BF16输出”不准确，输出norm后dtype未在初始运行捕获。已另存cache_precision_note，所有模型采用相同float16缓存；保留原receipt与源码commit，不伪造输出dtype或隐藏量化。实际Torch2.5.1/CUDA12.4，DINO源码91ffb6803c74aeb6927209b82560142743763dd7，无环境升级。
