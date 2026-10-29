> Pilot C100已完成并停止（2026-10-30香港复核）：C0/C1各自45k full续训55k至100k，匹配Primus40k/Linear500/38dev/七组MB1齐备；165000组输入/RNG/日程配对，无重放。C1−C0开发Dice−0.001007、MB1+0.049892；肌肉Dice+0.298020、AP+0.298814、Recall+0.061474，远端FP6.4958%→0.4171%。相对Pilot B开发+0.002127、MB1+0.028946，保留固定Barrier为候选研究主线，代价为OO/Turner/C.elegans下降。先读docs/Pilot_C100_final_report.md及work log45/46。服务inactive/退出0，GPU0未用，网站200；STOP，不自动延长、调参或新训练。 独立[draft PR#18](https://github.com/Polaris1010/EM-DINOv3-Research/pull/18)已提交推送，未合并；初始结果commit `e97cfc5209825546ca25df4c5c54847089b0a61a`，远程head已核验。

> 2026-10-28研究建议修正（DESIGN_ONLY）：用户仅询问下一步，明确不启动新训练。建议复用C0/C1各自45k full状态，以原30k anchor和固定Barrier配置配对续训到100k，完成固定下游后裁决长期效果；名义全流程约26h。45k负结果保持，停止当前配置的建议收紧为长期效果待100k检验。先读docs/Pilot_C_100k_Recommendation.md及work log44。所有现有任务仍停止，未授权100k；不重启旧45k/18h控制器，不自动训练。

> Pilot C-2已完成并停止（2026-10-28）：C0/C1各原30k恢复15k到45k，两head40k、两Linear500步、各38dev/七组MB1齐备；45000组输入/RNG/日程相同，无重放。C1−C0开发Dice−0.003790、MB1+0.003175，但肌肉Dice−0.075061、Recall−0.090534；相对中点两均值均低。当前Barrier不作为MICCAI主线继续。服务inactive/退出0，GPU0未用、网站200、历史SHA不变；先读docs/Pilot_C_Report.md和work log43。STOP，不延长、调参或追加Loss。 独立[draft PR#17](https://github.com/Polaris1010/EM-DINOv3-Research/pull/17)已提交推送，未合并；初始结果commit `2abead707acd37885b611de842e7942ec729d353`，远程head已核验。

> Pilot C-1已完成并停止：完整D2/30k状态可恢复，6项GPU测试、C0/C1各20步通过；60组输入/RNG/日程相等，C0首20步loss与历史完全相同。固定λ1，Barrier激活中位60.4%、局部梯度比中位6.94%；C0 767ms，C1生产估计861ms（实测含诊断1263ms）。完整15k配对+下游估计约9.2h，仅预算，未启动。先读docs/Pilot_C_Preflight_and_Budget.md和work log42。GPU0未用、网站200、服务inactive/退出0；本轮STOP，不自动15k或Primus。 独立[draft PR#16](https://github.com/Polaris1010/EM-DINOv3-Research/pull/16)已创建并推送，未合并；初始结果commit `0c99770a091be0a85b9ab2ec185feb6c43056468`，PR远程head已核验。

> Pilot B已完成并停止（2026-10-27）：固定D2/30k与100k的0.5 EMA中点；Linear23 500步和无条件Primus40k完成。38dev来源等权Dice 0.838631，MB1七组 0.505823；肌肉Dice 0.2157→0.4682，Recall未下降，HeLa等收益部分回退。先读docs/Pilot_B_Trajectory_Lerp05_Report.md及work log41。服务inactive/退出0，网站200，GPU0未用，历史SHA不变；停止，不启动Pilot C、SSL或新融合。 独立[draft PR15](https://github.com/Polaris1010/EM-DINOv3-Research/pull/15)已推送，未合并；初始结果commit `370b6bd48023034b11b705dc49c6f2f3e506c162`，远程PR head已核对。

> Pilot A2已完成并停止（2026-10-27）：同像素肌肉AUROC0.9429→0.9207、AP0.5043→0.3228；train-only偏置使100k Dice0.2906→0.3666、远端FP9.72%→1.93%，但Recall0.8019→0.3986，四控制来源Dice均下降。关闭全局偏置主修复路线；两Probe分别训练，Encoder共模原因仍INCONCLUSIVE。先读docs/Pilot_A2_Readout_Calibration_Report.md及work log40。GPU3运行50.2秒、服务inactive/退出0、网站200、历史输入SHA不变；无新模型训练、不自动后续实验。 独立draft PR14已创建并推送结果，未合并。

> 2026-10-27后续仅方案：已制定docs/Readout_Calibration_Minimal_Design.md，D2/30k与100k第23层同像素排序及MV3-train-only全局偏置对照。状态PLANNED/DESIGN_ONLY，用户尚未授权执行；不估计偏置、不启动评分或新训练。先读该方案及work log39。Pilot A仍已完成，既有模型、指标和registry不变。

> Pilot A已完成并停止（2026-10-27香港核实）：五encoder、四层特征及20个train-only轻量probe完成；633个锁定视野。肌肉D2/30k→100k Linear23 Dice0.5090→0.2906、远端FP0.31%→9.72%，23层有效秩33.3074→21.8166；稳定语义可分性下降和周期纹理捷径仍INCONCLUSIVE。先读docs/Feature_Mechanism_Pilot_Report.md、reviewed_verdicts.json及work log38。服务inactive/退出0、网站200、历史SHA不变；不自动Pilot B/C、SSL、Primus或其他实验。独立draft PR13，未合并。

> Coverage-NR已完成并停止（2026-10-22香港23:11核实）：SSL100k、head40k、38dev和2595张MB1均齐，GPU评分完成；开发Dice0.831821、MB1 Dice0.440027，低于D2-100k同seed0.841379/0.475754。CEM图像ID覆盖100%。服务inactive/MainPID0/退出0、网站200。先读docs/Coverage_NR_final_report.md和work_log36；本轮结束，不自动重启或新实验。

> 2026-10-22香港15:42：GPU评分等价性通过并部署。独立GPU校验误触旧controller未知PID保护，原任务94450停止；现已从94000 full状态恢复至94080，GPU1/2/3均100%、网站200。新服务coverage-nr-p0-gpu-resume-20261022.service；只接续原预算，不重复新初始化。详情work_logs/20261022_35_gpu_first_execution.md。

> 用户执行偏好（2026-10-22）：后续训练、推理和HD95/NSD等可加速评分优先使用物理GPU1/2/3，GPU0禁用。显存不足用主存配合完整轴GPU分块，不默认回退纯CPU。CPU用于I/O、调度、表面准备、汇总与必要验证。沿用精确算法，不为速度改阈值/尺度/精度或重新训练。

> Coverage-NR进度核实：2026-10-22香港15:23为93620/100000（93.62%），SSL仍运行；100k导出、40k head、开发集和MB1均未产生，暂无与D2-100k的新指标对比。本次只读核查，无训练启动或协议修改。详见docs/work_logs/20261022_34_coverage_nr_progress.md。

> 2026-10-21 23:06香港时间进度复核：Coverage-NR 33420/100000（33.42%），10k/30k EMA导出存在，最新full33000；GPU1/2/3工作、网站200。SSL已运行6.94h，按平均速度约余13.82h。head/dev/MB1尚未开始。按已消费ID位置重放CEM覆盖775442/1425433（54.40%）。本轮另安装用户级anti-defensive-writing；见work_logs/20261021_28_skill_install_progress.md。既有训练服务继续。

> 2026-10-21 16:11香港时间实测：Coverage-NR正式服务active，P0新初始化从0到60/100000；loss有限，GPU1/2/3利用率100%，各约11GiB，GPU0未用于本轮，网站200。code6129ac2，draft PR11。先读docs/Coverage_NR_execution.md与本轮run_index；已有服务继续，不重复启动。首seed40k及开发/MB1由同一有界服务顺序完成后停止。

> 2026-10-21 新授权：用户要求“继续，开始训练”。执行单组Coverage-NR从P0连续100k，继承既有MV+CEM池，之后首seed冻结Primus40k/38dev/固定MB1，完成停止。先读docs/Coverage_NR_execution.md及experiments/Coverage-NR/run_index.json。旧设计-only停止记录为历史，不重复启动服务，不增加R组/IsoOrg/QuantEM/D3，GPU0禁用。当前CPU计划通过，GPU smoke正在验证。

> 2026-10-21 metadata增量审查完成：见docs/Data_Expansion_Metadata_Review_20261021.md。当前P0设计/采样器/数据池不变；15个公开候选入口（8优先、7条件性），不是15个已证实独立acquisition。IsoOrg扣除历史MV与最终CEM同名来源后剩518子体积/18名称候选。没有图像下载、GT读取或训练，完成STOP。

> 2026-10-20 最新决定：D2-Coverage 改为从官方 P0 开始的完整100k方案，见 docs/D2_Coverage_P0_Design.md。取消30k恢复路线；旧设计和模拟保留为历史。全覆盖不能严格保持历史sqrt分布；需要因果比较时另建议同配额的P0有放回对照，未自动安排两组训练。当前仅设计更新，未启动训练，完成后STOP。

> 2026-10-20 D2-Coverage设计/CPU模拟完成，训练未授权未启动。先读docs/D2_Coverage_Experiment_Design.md：A67.05%、B81.46%、C100%；推荐同30k分支的全覆盖＋最小TV配额补偿C，69236步覆盖完，后70k集合TV最低15.7352%。完整checkpoint已只读核验，原D1/D2不变。当前codex/d2-coverage-design，仅提交设计/模拟/测试，完成STOP，不自动训练。

> 2026-10-20肌肉复核完成：见docs/D12_Glycolytic_muscle_review.md及最新work log。四组既有预测全302层计数复核、11张派生图完成；D2新增FP主要在背景细密纹理内部，未发现输入/评分错配证据。100k唯一ID覆盖D1 MV97.63%，D2 MV92.75%/CEM67.05%；保留来源/集合均100%。仅CPU分析，无训练/新推理/阈值或尺度搜索，完成STOP。

> D1/D2 100k正式完成并停止（2026-10-20现场复核）：两组SSL100k、新head各40k、38dev/12来源与七组MB1全量完成。30k→100k：D1开发0.817542→0.818930、MB1 0.488654→0.478851；D2开发0.829407→0.841379、MB1 0.461614→0.475754。100k D2开发优于D1，但MB1仍略低；逐来源变化见D12_100k_final_report.md。服务inactive，历史资产不变，网站200；旧failed/running通知为历史，不重启、不自动新实验。

> D1/D2终止状态 complete，UTC 20261020T045817Z。 两组累计100k SSL、首seed40k Primus、76条开发指标及5190条固定MB1诊断均完成；开发Dice D1=0.818930，D2=0.841379，D2−D1=+0.022448。先读docs/D12_100k_final_report.md。旧资产前后快照一致，网站200，GPU0未用，不自动更长SSL、D3或更多seed。 MB1七组Dice D1=0.478851，D2=0.475754。内部与跨数据集结论分开，不能仅凭内部均值提升声称广泛泛化。

> 100k任务已修复并恢复（2026-10-20，UTC快照2026-10-19T16:20:38.169570+00:00）：两组SSL均100000完成；新head D1=200/40000、D2=200/40000，初始化及同step采样hash一致，encoder冻结，100k SHA匹配。原故障为部署遗漏监督协议/cache映射，已补齐同一锁定输入；新服务d12-100k-readout-resume1-20261020.service，code33a0ff7，GPU1/2训练、GPU0未用、网站200。开发/MB1尚无100k结果。下方failed是保留历史，不重训SSL、不重复启动、不自动新实验。

> D1/D2终止状态 failed，UTC 20261019T091333Z。 保留实际产物，查看STATUS与服务器阶段日志；未完成部分不得声称完成，不自动改参/重启。

> D1/D2 100k已实际启动（UTC 2026-10-18T03:16:37.643181+00:00）：D1已完整恢复并到30020/100000，D2等待顺序接续；GPU1/2/3工作、GPU0未用、网站200。运行code f51a9e8，draft PR9，服务d12-100k-continuation-20261018.service。后续固定100k新head首seed40k、全部38dev和七组MB1诊断，结束提交推送停止。新窗口只核实既有服务/collector，不重复启动。下方30k停止通知为已保留历史，本轮用户授权已覆盖。

> D1/D2新授权（2026-10-18）：用户要求两组共同继续100k。各自完整30k状态续训70k，独立根ssl_diversity_d12_100k_20261018；固定100k新Primus首seed40k和38dev/七组MB1诊断，旧30k不改。先读docs/D12_100k_execution.md及experiments/D12-100k/run_index.json。本段覆盖下方旧不自动100k通知；不自动D3、更长SSL或额外seed。当前仅准备，未把计划写成已启动。

> D12完成复核（2026-10-18）：D1/D2各30k SSL和同seed40k Primus、全部开发/MB1诊断完成并停止。开发Dice0.817542→0.829407，但MB1七组Dice0.488654→0.461614；不能仅凭内部提升声称跨域改善。先读D12_final_report.md末尾完成复核与最新work log；不自动100k/D3或更多seed。

> D1/D2终止状态 complete，UTC 20261017T204218Z。 两组30k SSL、首seed40k Primus、76条开发指标及5190条固定MB1诊断均完成；开发Dice D1=0.817542，D2=0.829407，D2−D1=+0.011864。先读docs/D12_final_report.md。旧资产前后快照一致，网站200，GPU0未用，不自动100k、D3或更多seed。

> D1已从2711步完整状态接续（UTC 2026-10-17T06:33:19.240669+00:00核实至2720/30000）：心跳原子发布修复与Linux并发测试通过，训练器/配置/预算不改，GPU1/2/3工作、GPU0未用、网站200；D2与下游评测尚未开始。旧失败记录保留于7b29435，本次接续代码545ebba；先读docs/work_logs/20261017_15_d12_interruption_resume.md及experiments/D12/run_index.json。下方failed/停止是已保留历史，不代表本次恢复状态。

> D1/D2终止状态 failed，UTC 20261017T032244Z。 保留实际产物，查看STATUS与服务器阶段日志；未完成部分不得声称完成，不自动改参/重启。

> D1/D2最新授权（2026-10-17）：执行来源均衡MitoVerse vs50:50 CEM的匹配30k SSL，均从自然DINOv3开始；采用MitoNet排除benchmark及关联图像的原则，新共享留出池。两组smoke通过，正式事实见experiments/D12/run_index.json、服务器STATUS和docs/D12_execution.md。之后首seed固定MV3 Primus40k/开发/已暴露MB1诊断；不修改历史、不用GPU0、不自动100k/D3。下方旧停止通知由本次明确授权更新。

> MB1已完成并停止（核验UTC 2026-10-15T08:01:15.067219+00:00）：七组、八run、20760条二维指标和九张固定对照图全部完成。G-P0 Dice 0.457938；G-B100 Dice 0.502665；Official-MitoNet-v1 Dice 0.712154；MitoNet-MV Dice 0.240936。先读docs/MB1_summary.md、docs/MB1_overlap_status.md、experiments/MB-1/MB1_local_verification.json和最新work log。五组B100 SSL来源与抽样原图重叠，Lucchi++精确区域未确认，TEM未发现名称匹配；不能称严格未见图像外测。服务inactive/MainPID0，网站200，旧资产不变。任务结束，不启动新实验。

> MB1最新授权（2026-10-15）：只做EMPIAR-10982全部七组benchmark，复用MV3六个head及MV4/MV5两个MitoNet，TTA OFF。MV3/4/5结果保持；先读docs/MB1_execution.md、experiments/MB-1/run_index.json和最新work log。已有五组来源进入B100 SSL池，不能称严格未见图像外测；Lucchi++来源关联但精确区域未确认，TEM未发现名称重叠而非证明无暴露。旧停止通知由本次明确授权更新；结束MB1后仍停止。

> MV5完成状态已现场复核：40k和38区/12来源全部完成，服务inactive/MainPID0/exit0。结果已推送0933540；最终报告docs/MV5_final_report.md，最新本轮记录docs/work_logs/20261015_12_mv5_completion_recheck.md。自动收取器已退出，禁止按下方历史running通知恢复任务；本次仅修复PR交付回执与元数据，未启动新实验。

> MV5已完成：MitoNet-MV 40k、38区域/12来源开发指标完成；先读docs/MV5_final_report.md。任务结束，不启动新实验。 UTC 20261015T035253Z

> MV5 最新授权（2026-10-15）：用户决定结束MV4B实例分组排查，开始同156训练区域/38开发区域的MitoNet监督适配。MV4B机制在已测四区域正式关闭，不延伸其余34。MV4C缺失原TIFF不阻塞本轮。先读docs/MV5_execution.md、experiments/MV-5/run_index.json及最新work log；只接续本轮有界执行器，不重复启动，不训练DINOv3，不启动新排查或SSL。下方旧“停止/等待”是历史，已被本次明确授权更新。

> MV4C可执行核查已结束并停止：Guay作者原图→Zarr全像素复现一致；两来源Zarr→实际cache/GT/既有概率坐标链一致。Müller转换前TIFF未取得，原图链明确UNVERIFIED，不能声称整项原图核验完成。当前分支codex/mv4c-input-provenance，draft PR5（base为MV4B分支，未合并）；先读docs/MV4C_input_provenance_report.md和experiments/MV-4C/audit_index.json。仅待真实high_c1原TIFF路径；不自动阈值/尺度搜索、反相推理或训练。

> MV4B已完成：四区域Semantic/Panoptic差异不明显，不扩展38。Guay Dice0.014253、Müller high_c1约2.35e-7未恢复；raw0.5掩码与旧Panoptic四区域逐像素相等。旧MV4/MV3全部保留，不能声称已完成38区域Semantic Only或公平独立测试超越MitoNet。当前分支codex/mv4b-semantic-panoptic，draft PR4（base为MV4分支，未合并）；先读docs/MV4B_final_report.md、experiments/MV-4B/run_index.json及work_logs/20261014_09_mv4b_semantic_panoptic.md。研究服务已停止，不自动启动下一实验。

> MV4已完成并停止（2026-10-14香港时间23:15:04核验）。同38开发区域/12来源，来源等权Dice：P0 0.813741、B100 0.826449、官方MitoNet 0.610209；B100在8/12来源更高。当前分支codex/mv4-official-mitonet，独立draft PR3，base为未合并MV3分支。先读docs/MV4_final_report.md、experiments/MV-4/run_index.json和work_logs/20261014_08_mv4_official_mitonet.md。服务inactive/MainPID0，指标266行核验PASS；不重启MV3/MV4或旧watcher，不自动启动新实验。

# 当前交付：MV4已完成

完整报告：[MV4_final_report.md](MV4_final_report.md)。G-P0/G-B100复用全部三个decoder seeds，没有重训/重推；MitoNet使用固定官方权重、默认整图二维推理、TTA OFF，实例输出保留服务器。Han整幅GT继续如实披露，Turner nm指标保留spacing限制；这是监督数据和训练暴露不同的完整系统开发对照。

实际运行代码db2ae4e；完整指标experiments/MV-4/MV4_per_region_metrics.csv，来源差距MV4_source_deltas.csv，最终校验MV4_results_verification.json。服务器根/实例路径见run_index.json和最终报告。代码、指标、本轮Markdown及交接随PR3提交推送；PR未合并，不假设main已包含本轮。

下一项仅为决策：优先考虑跨来源稳健性，关注Guay/Müller仍存在的B100相对P0退化，同时保留MitoNet在Casser/Mekuc/Turner/Wei的优势作为实际改进目标。用户后续明确授权前停止所有实验；不新增审计、诊断、预训练或延长head训练。

## 以下保留MV3及更早阶段的历史状态/计划

# 当前任务完成后停止

MV3 v2六组40k、228条指标与18个checkpoint完成；先读MV3_final_report.md、最终核验与work_logs/20261014_07_mv3_completion.md。当前服务已inactive，不恢复训练或评分。

下一项科学决策为官方MitoNet同图像/有效GT对照，后续用户明确要求后执行。此轮没有启动MitoNet、Gram/HRA、TopoMask、LAST、LT或新预训练。

## 以下为完成前计划，保留历史

> 最新MV3终止状态 20261014T091140Z：complete。所有6组完成，先读MV3_final_report.md；不重启，不自动启动新实验。

> 最新核查：2026-10-14 15:00香港时间，v2首seed两组40k及38区域×12来源完整开发评测已完成，Dice P0 0.8130 / B100 0.8273（+0.0143）。P0全部三个seed、B100前两个seed均已40k训练完成；追加seed的开发推理在运行，B100第三seed等待既定调度。尚无三seed最终报告。

> 当前协议为 **MV-Generalist-v2**。用户明确指定“取消 mask 限制，整幅 GT 参与 loss”；仅 Han24 跳过 annotation mask，整幅 GT 参与采样、loss 和开发指标。其他来源、划分、冻结 encoder、原版 Primus、匹配 seeds、共同 40k 预算与 TTA=OFF 保持原配置。旧 v1 已停止并完整保留，不能混入 v2 对照。

# 下一项已确定的行动：完成已授权MV-3

2026-10-14用户已明确改为MV-3多来源监督训练，以下旧MV2结果收取计划被取代。不要按旧段落增加验证、推理、Linear Probe、Gram/HRA或新审计。

当前分支 `codex/mv3-generalist-training`，服务器 `/opt/data/3_117/tanghr/codex/mv_generalist_v2_han_fullgt_20261014`，service `mv3-generalist-v2-han-fullgt-20261014.service`。先读取现有STATUS、progress和最新evidence，禁止重复启动。只继续G-P0/G-B100、3匹配decoder seeds、各40000 optimizer steps的共同锁定协议。原版Primus，[5,11,17,23]，encoder冻结，TTA=OFF，GPU0禁用。

v2两组实际smoke已PASS，相同decoder初始化和采样SHA，Han整幅crop参与loss；首seed两组已各13200正式steps（10k checkpoint已保存），详细状态以experiments/MV-3/v2/run_index.json及最新v2快照为准。2×40k完成后做38区域/12来源全开发验证及既有物理指标，再补20261015/20261016；有界执行器仅执行这6个run。

MV-Generalist-v2：156 train/38 dev，12来源；部分探索性dev已转train，不再作为新验证；official test/已映射validation保留。Han整幅GT，不使用mask；有效区与每来源体积清单见 `experiments/MV-3/v2/MV3_source_coverage.csv`。v2全部194个区域复用现有图像/GT缓存只读；只有本轮元数据写入v2目录。不能把旧4-volume约束恢复成前置门槛。

结束时收取MV3_final_report.md、18个milestone checkpoint SHA、逐区域/来源指标与配对变化，回答监督扩展收益、B100多来源优势、Guay/Müller变化、主要短板和优先改进方向。比较旧预测仅作同新heldout区域的描述性参照，不重新运行旧9个head，不改历史记录。保持共同40k预算，不为某个encoder单独追加训练。

每轮将代码、状态和Markdown commit/push，实际报告完成前状态保持VERIFIED_RUNNING/PLANNED。最终报告后停止，不自动启动官方MitoNet或新预训练。详细协议见MV3_PROTOCOL，交接见NEW_WINDOW_HANDOFF。

## 以下保留为上一轮历史计划

**初始化PR已合并并交付main；下一项科学行动是收取已在运行的MV-2无TTA扩展结果，回答B100k对P0的收益是否跨更多来源成立。** 本轮仅完成代码和记录交付，不新增科学实验，也不重新启动MV-2。

现有服务器任务：`/opt/data/3_117/tanghr/codex/mv_dev_v2_notta_20261013`；service为 `mv-dev-v2-notta-20261013.service`。2026-10-13本轮快照为运行中，最终状态仍待读取。旧 `mv_dev_v2_expansion_20261013` 四翻转结果已废弃，保留历史，不参与正式汇总。

输入：`experiments/registry.yaml`登记的MV-1九个head（20000 steps）、实际MV-2 manifest188个volume/ROI与reserved test排除项、原版Primus、固定encoder、原normalization与原sliding函数。TTA=OFF，不训练，不增加模块，不改预处理或checkpoint选择。

验收：

1. 只读确认现有job最终状态；任务未完时保持VERIFIED_RUNNING，不把partial输出写成完整结果，也不默认启动失败重试。
2. 保存新的带时间戳status/result/manifest/hash快照，证明九个head和历史输入未变化。核对3×3模型/seed组及每组188个条目的完成、复用、跳过/失败情况；不静默缩小队列。
3. 每来源汇总Dice/Precision/Recall/HD95 nm/NSD@16nm，分开官方validation与自定义dev、每seed波动与独立volume/acquisition数量。unknown acquisition明确标UNVERIFIED。
4. 报告B100k−P0的逐volume和来源差异，B150k仅作已有参考。列出缺少官方native mapping或不可用区域，不把锁定test加入。结论限定transductive development。
5. 第二PR只做上述结果收取、覆盖核对和科学解释。通过已有CPU registry checks，不改模型。随后给出继续/转向的单项科学决策。

GitHub授权已通过用户确认的浏览器流程恢复，private remote已建立，首PR状态见 `docs/GITHUB.md`。未来严格B_clean与acquisition审计不阻塞这项开发问题。
