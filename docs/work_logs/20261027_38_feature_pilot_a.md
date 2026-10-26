# 修订Pilot A：冻结表征与周期纹理诊断

用户授权仅Pilot A；禁止Pilot B/C、SSL与Primus训练。创建独立分支codex/feature-mechanism-pilot-a，以02a5ea0（Coverage-NR完成及公开可见性）为基线，不动SARR窗口。

已读AGENTS/交接/状态/registry、D12肌肉复核、Coverage-NR完成报告及原始配置，定位五正式encoder资产。服务器初始snapshot确认Coverage-NR complete、网站200；GPU3完全空闲，1/2仅网站少量context、util0，GPU0禁用。

代码将规划、train数据准备、冻结特征+train-only probes、offline GT评分、GPU谱/可分性/周期性统计分阶段；拟合阶段不读取诊断GT，全部probe SHA固定后才允许offline标签读取。详见协议与execution文档。此时尚未声称实验已启动或有结果。

本轮数字/图表应以实际CSV为依据，原图/大缓存/权重不提交。代码、锁定坐标、环境和结果最终推送独立PR；PR不自动合并。完成后停止。后续在本记录追加实际运行、检查与交付事实。
