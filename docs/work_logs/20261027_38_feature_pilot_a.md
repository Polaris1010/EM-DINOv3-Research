# 修订Pilot A：冻结表征与周期纹理诊断

用户授权仅Pilot A；禁止Pilot B/C、SSL与Primus训练。创建独立分支codex/feature-mechanism-pilot-a，以02a5ea0（Coverage-NR完成及公开可见性）为基线，不动SARR窗口。

已读AGENTS/交接/状态/registry、D12肌肉复核、Coverage-NR完成报告及原始配置，定位五正式encoder资产。服务器初始snapshot确认Coverage-NR complete、网站200；GPU3完全空闲，1/2仅网站少量context、util0，GPU0禁用。

代码将规划、train数据准备、冻结特征+train-only probes、offline GT评分、GPU谱/可分性/周期性统计分阶段；拟合阶段不读取诊断GT，全部probe SHA固定后才允许offline标签读取。详见协议与execution文档。此时尚未声称实验已启动或有结果。

本轮数字/图表应以实际CSV为依据，原图/大缓存/权重不提交。代码、锁定坐标、环境和结果最终推送独立PR；PR不自动合并。完成后停止。后续在本记录追加实际运行、检查与交付事实。

正式启动：源commit772f0fb；服务feature-mechanism-pilot-a-20261027.service，MainPID2993673。锁定633个视野（312train+320random eval+1historical case），样本SHA18622e1374467fe8a480f1b8a13e15370b51ac3386039d389bc4f55351c11458。五encoder实际SHA通过、GPU数值smoke通过；20个探针各500步均完成，训练坐标hash52112acd1aca39ebbf17bd943efa09cfe9bef677dda93f6f3328d79c02b2ba54。12来源均具有前景/背景训练patch。当前GPU统计分析运行，未声称有最终机制结论。

缓存精度核对：实际feature cache为float16；原receipt描述“精确表示BF16输出”不准确，输出norm后dtype未在初始运行捕获。已另存cache_precision_note，所有模型采用相同float16缓存；保留原receipt与源码commit，不伪造输出dtype或隐藏量化。实际Torch2.5.1/CUDA12.4，DINO源码91ffb6803c74aeb6927209b82560142743763dd7，无环境升级。

## 完成与裁决复核

2026-10-27香港完成核验：STATUS complete，服务inactive/MainPID0/ExecMainStatus0，网站200，五历史encoder结束SHA一致。五冻结模型、20个probe均完成；GPU分析738.9秒，未运行SSL、Primus、Pilot B/C或新阶段。

收取45份小型结果与图件并逐SHA核对。三张统计SVG随Git归档，四张PNG保留本机review artifacts；原始图像、权重与大缓存不进入public仓库。原始自动report/verdicts保留不变，人工复核以reviewed_verdicts和最终报告为准。

肌肉D2/30k→100k Linear23 Dice0.5090→0.2906、远端FP0.31%→9.72%；相同随机视野Primus Dice0.4785→0.2018。23层有效秩33.3074→21.8166，但linear AUROC仅0.9811→0.9727、prototype AUROC0.5138→0.9105。五来源23层均谱压缩而语义排序变化不同向，不能认定整体语义坍塌。

自动Q4的SUPPORTED只基于正向点估计；肌肉Primus相关rho0.0914且Z块区间跨0，linear原始相关为负、partial接近0，最终裁决INCONCLUSIVE。所有肌肉窗口同属train最低对比度/梯度档，固定分层仅剩两个Y半区；不根据结果改cutoffs或追加分析。Q2的SUPPORTED仅限定深层谱压缩；语义退化层位仍未定。

唯一候选机制：跨域背景响应相对于固定决策边界的偏移，作为后续建议而非已确认原因。本轮不实施Topo-Anchor或任何新约束。

交付独立draft PR13，base codex/coverage-nr-p0-training，初始分支5d6d689已推送；最终报告、指标、代码、Markdown和CPU检查随后在同一PR收尾。GitHub本地DNS返回不可达虚拟IP，发布使用HTTPS验证的DoH地址和单命令DNS override，保持证书校验；未改系统网络配置。

CPU收尾检查：registry生成一致；validate_repository PASS（60条记录）；unittest 51项、3项平台跳过，结果OK。独立复算8025条crop、125条来源聚合、40条配对AUROC及300份谱分布全部一致，59条历史registry原样保留。Markdown写入产生的CRLF已在本轮作者文件修正为LF，git diff --check通过；原始服务器文件保持SHA。检查收据closeout_checks.json。最终结果、代码和本轮Markdown随同一分支提交；以PR实际head和交付回执核实推送，不自动合并。

最终审阅版以本Git分支docs报告和reviewed_verdicts为准。结果下载后SSH会话失效，附加的服务器reviewed副本未创建；不影响已完成GPU实验及45份已验证下载，不为附加副本重连或重复计算。服务器自动报告原件保持。
