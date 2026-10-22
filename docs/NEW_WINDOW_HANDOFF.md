> 2026-10-22香港15:42：GPU评分等价性通过并部署。独立GPU校验误触旧controller未知PID保护，原任务94450停止；现已从94000 full状态恢复至94080，GPU1/2/3均100%、网站200。新服务coverage-nr-p0-gpu-resume-20261022.service；只接续原预算，不重复新初始化。详情work_logs/20261022_35_gpu_first_execution.md。

> 用户执行偏好（2026-10-22）：后续训练、推理和HD95/NSD等可加速评分优先使用物理GPU1/2/3，GPU0禁用。显存不足用主存配合完整轴GPU分块，不默认回退纯CPU。CPU用于I/O、调度、表面准备、汇总与必要验证。沿用精确算法，不为速度改阈值/尺度/精度或重新训练。

> Coverage-NR进度核实：2026-10-22香港15:23为93620/100000（93.62%），SSL仍运行；100k导出、40k head、开发集和MB1均未产生，暂无与D2-100k的新指标对比。本次只读核查，无训练启动或协议修改。详见docs/work_logs/20261022_34_coverage_nr_progress.md。

> 2026-10-21 23:06香港时间进度复核：Coverage-NR 33420/100000（33.42%），10k/30k EMA导出存在，最新full33000；GPU1/2/3工作、网站200。SSL已运行6.94h，按平均速度约余13.82h。head/dev/MB1尚未开始。按已消费ID位置重放CEM覆盖775442/1425433（54.40%）。本轮另安装用户级anti-defensive-writing；见work_logs/20261021_28_skill_install_progress.md。既有训练服务继续。

> 2026-10-21 16:11香港时间实测：Coverage-NR正式服务active，P0新初始化从0到60/100000；loss有限，GPU1/2/3利用率100%，各约11GiB，GPU0未用于本轮，网站200。code6129ac2，draft PR11。先读docs/Coverage_NR_execution.md与本轮run_index；已有服务继续，不重复启动。首seed40k及开发/MB1由同一有界服务顺序完成后停止。

> 2026-10-21 新授权：用户要求“继续，开始训练”。执行单组Coverage-NR从P0连续100k，继承既有MV+CEM池，之后首seed冻结Primus40k/38dev/固定MB1，完成停止。先读docs/Coverage_NR_execution.md及experiments/Coverage-NR/run_index.json。旧设计-only停止记录为历史，不重复启动服务，不增加R组/IsoOrg/QuantEM/D3，GPU0禁用。当前CPU计划通过，GPU smoke正在验证。

> 2026-10-21 metadata增量审查完成：见docs/Data_Expansion_Metadata_Review_20261021.md。当前P0设计/采样器/数据池不变；15个公开候选入口（8优先、7条件性），不是15个已证实独立acquisition。IsoOrg扣除历史MV与最终CEM同名来源后剩518子体积/18名称候选。没有图像下载、GT读取或训练，完成STOP。

> 2026-10-20 OrgMIM答疑：已核实官方5组928体积的组均匀→组内均匀有放回采样，不保证全覆盖。论文全文/组构建方式未核实；详见docs/OrgMIM_IsoOrg_sampling_note.md。P0设计不变，未启动训练。

> 2026-10-20 最新决定：D2-Coverage 改为从官方 P0 开始的完整100k方案，见 docs/D2_Coverage_P0_Design.md。取消30k恢复路线；旧设计和模拟保留为历史。全覆盖不能严格保持历史sqrt分布；需要因果比较时另建议同配额的P0有放回对照，未自动安排两组训练。当前仅设计更新，未启动训练，完成后STOP。

> 2026-10-20 D2-Coverage设计/CPU模拟完成，训练未授权未启动。先读docs/D2_Coverage_Experiment_Design.md：A67.05%、B81.46%、C100%；推荐同30k分支的全覆盖＋最小TV配额补偿C，69236步覆盖完，后70k集合TV最低15.7352%。完整checkpoint已只读核验，原D1/D2不变。当前codex/d2-coverage-design，仅提交设计/模拟/测试，完成STOP，不自动训练。

> 2026-10-20肌肉复核完成：见docs/D12_Glycolytic_muscle_review.md及最新work log。四组既有预测全302层计数复核、11张派生图完成；D2新增FP主要在背景细密纹理内部，未发现输入/评分错配证据。100k唯一ID覆盖D1 MV97.63%，D2 MV92.75%/CEM67.05%；保留来源/集合均100%。仅CPU分析，无训练/新推理/阈值或尺度搜索，完成STOP。

> 2026-10-20建议轮：用户只要求研究建议、数据范围和耗时，禁止新训练。建议暂缓D3，先一次有界肌肉假阳性定位（尚未执行/授权）；详见work_logs/20261020_21_d12_research_decision_data_runtime.md。两组共享153体积/34143合法Z，D2另用1425433图全可采样池；非原始全集、非保证逐图遍历。SSL累计D1约22.15h、D2约20.72h，各100k head约51分钟。已完成实验继续STOP。

> D1/D2 100k正式完成并停止（2026-10-20现场复核）：两组SSL100k、新head各40k、38dev/12来源与七组MB1全量完成。30k→100k：D1开发0.817542→0.818930、MB1 0.488654→0.478851；D2开发0.829407→0.841379、MB1 0.461614→0.475754。100k D2开发优于D1，但MB1仍略低；逐来源变化见D12_100k_final_report.md。服务inactive，历史资产不变，网站200；旧failed/running通知为历史，不重启、不自动新实验。

> D1/D2终止状态 complete，UTC 20261020T045817Z。 两组累计100k SSL、首seed40k Primus、76条开发指标及5190条固定MB1诊断均完成；开发Dice D1=0.818930，D2=0.841379，D2−D1=+0.022448。先读docs/D12_100k_final_report.md。旧资产前后快照一致，网站200，GPU0未用，不自动更长SSL、D3或更多seed。 MB1七组Dice D1=0.478851，D2=0.475754。内部与跨数据集结论分开，不能仅凭内部均值提升声称广泛泛化。

> 100k任务已修复并恢复（2026-10-20，UTC快照2026-10-19T16:20:38.169570+00:00）：两组SSL均100000完成；新head D1=200/40000、D2=200/40000，初始化及同step采样hash一致，encoder冻结，100k SHA匹配。原故障为部署遗漏监督协议/cache映射，已补齐同一锁定输入；新服务d12-100k-readout-resume1-20261020.service，code33a0ff7，GPU1/2训练、GPU0未用、网站200。开发/MB1尚无100k结果。下方failed是保留历史，不重训SSL、不重复启动、不自动新实验。

> D1/D2终止状态 failed，UTC 20261019T091333Z。 保留实际产物，查看STATUS与服务器阶段日志；未完成部分不得声称完成，不自动改参/重启。

> D1/D2 100k已实际启动（UTC 2026-10-18T03:16:37.643181+00:00）：D1已完整恢复并到30020/100000，D2等待顺序接续；GPU1/2/3工作、GPU0未用、网站200。运行code f51a9e8，draft PR9，服务d12-100k-continuation-20261018.service。后续固定100k新head首seed40k、全部38dev和七组MB1诊断，结束提交推送停止。新窗口只核实既有服务/collector，不重复启动。下方30k停止通知为已保留历史，本轮用户授权已覆盖。

> D1/D2新授权（2026-10-18）：用户要求两组共同继续100k。各自完整30k状态续训70k，独立根ssl_diversity_d12_100k_20261018；固定100k新Primus首seed40k和38dev/七组MB1诊断，旧30k不改。先读docs/D12_100k_execution.md及experiments/D12-100k/run_index.json。本段覆盖下方旧不自动100k通知；不自动D3、更长SSL或额外seed。当前仅准备，未把计划写成已启动。

> D12完成复核（2026-10-18）：D1/D2各30k SSL和同seed40k Primus、全部开发/MB1诊断完成并停止。开发Dice0.817542→0.829407，但MB1七组Dice0.488654→0.461614；不能仅凭内部提升声称跨域改善。先读D12_final_report.md末尾完成复核与最新work log；不自动100k/D3或更多seed。

> D1/D2终止状态 complete，UTC 20261017T204218Z。 两组30k SSL、首seed40k Primus、76条开发指标及5190条固定MB1诊断均完成；开发Dice D1=0.817542，D2=0.829407，D2−D1=+0.011864。先读docs/D12_final_report.md。旧资产前后快照一致，网站200，GPU0未用，不自动100k、D3或更多seed。

> D12最新进度（2026-10-17香港17:17）：D1 15050/30000，10k导出已存在；D2/head/开发评测尚未开始，无新指标。service active，继续既有任务，禁止重复启动。实际快照见evidence/D12/progress_20261017T091727Z.json与最新work log。

> D1已从2711步完整状态接续（UTC 2026-10-17T06:33:19.240669+00:00核实至2720/30000）：心跳原子发布修复与Linux并发测试通过，训练器/配置/预算不改，GPU1/2/3工作、GPU0未用、网站200；D2与下游评测尚未开始。旧失败记录保留于7b29435，本次接续代码545ebba；先读docs/work_logs/20261017_15_d12_interruption_resume.md及experiments/D12/run_index.json。下方failed/停止是已保留历史，不代表本次恢复状态。

> D1/D2终止状态 failed，UTC 20261017T032244Z。 保留实际产物，查看STATUS与服务器阶段日志；未完成部分不得声称完成，不自动改参/重启。

> D1/D2最新授权（2026-10-17）：执行来源均衡MitoVerse vs50:50 CEM的匹配30k SSL，均从自然DINOv3开始；采用MitoNet排除benchmark及关联图像的原则，新共享留出池。两组smoke通过，正式事实见experiments/D12/run_index.json、服务器STATUS和docs/D12_execution.md。之后首seed固定MV3 Primus40k/开发/已暴露MB1诊断；不修改历史、不用GPU0、不自动100k/D3。下方旧停止通知由本次明确授权更新。

> MB1已完成并停止（核验UTC 2026-10-15T08:01:15.067219+00:00）：七组、八run、20760条二维指标和九张固定对照图全部完成。G-P0 Dice 0.457938；G-B100 Dice 0.502665；Official-MitoNet-v1 Dice 0.712154；MitoNet-MV Dice 0.240936。先读docs/MB1_summary.md、docs/MB1_overlap_status.md、experiments/MB-1/MB1_local_verification.json和最新work log。五组B100 SSL来源与抽样原图重叠，Lucchi++精确区域未确认，TEM未发现名称匹配；不能称严格未见图像外测。服务inactive/MainPID0，网站200，旧资产不变。任务结束，不启动新实验。

> MB1最新授权（2026-10-15）：只做EMPIAR-10982全部七组benchmark，复用MV3六个head及MV4/MV5两个MitoNet，TTA OFF。MV3/4/5结果保持；先读docs/MB1_execution.md、experiments/MB-1/run_index.json和最新work log。已有五组来源进入B100 SSL池，不能称严格未见图像外测；Lucchi++来源关联但精确区域未确认，TEM未发现名称重叠而非证明无暴露。旧停止通知由本次明确授权更新；结束MB1后仍停止。

> MV5完成状态已现场复核：40k和38区/12来源全部完成，服务inactive/MainPID0/exit0。结果已推送0933540；最终报告docs/MV5_final_report.md，最新本轮记录docs/work_logs/20261015_12_mv5_completion_recheck.md。自动收取器已退出，禁止按下方历史running通知恢复任务；本次仅修复PR交付回执与元数据，未启动新实验。

> MV5已完成：MitoNet-MV 40k、38区域/12来源开发指标完成；先读docs/MV5_final_report.md。任务结束，不启动新实验。 UTC 20261015T035253Z

> MV5实际运行：service mv5-mitonet-adaptation-20261015.service、root见run_index；训练GPU3，一个seed40k，后续38dev自动评估。代码2cbc7c7已推送、draft PR6；本地被动收取器scripts/watch_mv5_results.py。新窗口只读查STATUS/已有collector，不重复启动。

> MV5 最新授权（2026-10-15）：用户决定结束MV4B实例分组排查，开始同156训练区域/38开发区域的MitoNet监督适配。MV4B机制在已测四区域正式关闭，不延伸其余34。MV4C缺失原TIFF不阻塞本轮。先读docs/MV5_execution.md、experiments/MV-5/run_index.json及最新work log；只接续本轮有界执行器，不重复启动，不训练DINOv3，不启动新排查或SSL。下方旧“停止/等待”是历史，已被本次明确授权更新。

> MV4C可执行核查已结束并停止：Guay作者原图→Zarr全像素复现一致；两来源Zarr→实际cache/GT/既有概率坐标链一致。Müller转换前TIFF未取得，原图链明确UNVERIFIED，不能声称整项原图核验完成。当前分支codex/mv4c-input-provenance，draft PR5（base为MV4B分支，未合并）；先读docs/MV4C_input_provenance_report.md和experiments/MV-4C/audit_index.json。仅待真实high_c1原TIFF路径；不自动阈值/尺度搜索、反相推理或训练。

> MV4B已完成：四区域Semantic/Panoptic差异不明显，不扩展38。Guay Dice0.014253、Müller high_c1约2.35e-7未恢复；raw0.5掩码与旧Panoptic四区域逐像素相等。旧MV4/MV3全部保留，不能声称已完成38区域Semantic Only或公平独立测试超越MitoNet。当前分支codex/mv4b-semantic-panoptic，draft PR4（base为MV4分支，未合并）；先读docs/MV4B_final_report.md、experiments/MV-4B/run_index.json及work_logs/20261014_09_mv4b_semantic_panoptic.md。研究服务已停止，不自动启动下一实验。

> MV4已完成并停止（2026-10-14香港时间23:15:04核验）。同38开发区域/12来源，来源等权Dice：P0 0.813741、B100 0.826449、官方MitoNet 0.610209；B100在8/12来源更高。当前分支codex/mv4-official-mitonet，独立draft PR3，base为未合并MV3分支。先读docs/MV4_final_report.md、experiments/MV-4/run_index.json和work_logs/20261014_08_mv4_official_mitonet.md。服务inactive/MainPID0，指标266行核验PASS；不重启MV3/MV4或旧watcher，不自动启动新实验。

# 当前交付：MV4已完成

完整报告：[MV4_final_report.md](MV4_final_report.md)。G-P0/G-B100复用全部三个decoder seeds，没有重训/重推；MitoNet使用固定官方权重、默认整图二维推理、TTA OFF，实例输出保留服务器。Han整幅GT继续如实披露，Turner nm指标保留spacing限制；这是监督数据和训练暴露不同的完整系统开发对照。

实际运行代码db2ae4e；完整指标experiments/MV-4/MV4_per_region_metrics.csv，来源差距MV4_source_deltas.csv，最终校验MV4_results_verification.json。服务器根/实例路径见run_index.json和最终报告。代码、指标、本轮Markdown及交接随PR3提交推送；PR未合并，不假设main已包含本轮。

下一项仅为决策：优先考虑跨来源稳健性，关注Guay/Müller仍存在的B100相对P0退化，同时保留MitoNet在Casser/Mekuc/Turner/Wei的优势作为实际改进目标。用户后续明确授权前停止所有实验；不新增审计、诊断、预训练或延长head训练。

## 以下保留MV3及更早阶段的历史状态/计划

# MV3已完成：新窗口先读最终报告

当前MV-Generalist-v2已完成六组×40000 steps、228条全开发指标、18个checkpoint。服务器service inactive、MainPID=0，网站200；GPU0未使用，没有自动新实验。先读[MV3_final_report.md](MV3_final_report.md)、experiments/MV-3/v2的指标与最终核验、最新work_logs记录。

三seed来源等权Dice：P0 0.813741±0.000827，B100 0.826449±0.000832；配对Δ=+0.012707±0.001361（decoder seed样本SD）。三个seed差值均正；Guay/Müller持续略低，Haberl基本持平。数据协议和监督预算改变的旧预测对照另表保存，不混成同一排行榜。

本轮已经结束，不恢复MV3或旧watcher，不自动运行MitoNet/SSL。下一项研究决策是同图像/有效GT的官方MitoNet对照，须后续用户任务再执行。当前分支codex/mv3-generalist-training、draft PR2，未经自动合并。

## 以下为完成前的交接快照

> 最新MV3终止状态 20261014T091140Z：complete。所有6组完成，先读MV3_final_report.md；不重启，不自动启动新实验。

> 最新核查：2026-10-14 15:00香港时间，v2首seed两组40k及38区域×12来源完整开发评测已完成，Dice P0 0.8130 / B100 0.8273（+0.0143）。P0全部三个seed、B100前两个seed均已40k训练完成；追加seed的开发推理在运行，B100第三seed等待既定调度。尚无三seed最终报告。

# 新窗口交接

当前协议为 **MV-Generalist-v2**。用户明确指定“取消 mask 限制，整幅 GT 参与 loss”；仅 Han24 跳过 annotation mask，整幅 GT 参与采样、loss 和开发指标。其他来源、划分、冻结 encoder、原版 Primus、匹配 seeds、共同 40k 预算与 TTA=OFF 保持原配置。旧 v1 已停止并完整保留，不能混入 v2 对照。

分支 `codex/mv3-generalist-training`，draft PR #2。先读最新工作记录、MV3_PROTOCOL和 `experiments/MV-3/v2/run_index.json`，核验新service实际状态；不重复启动。当前服务器根目录 `/opt/data/3_117/tanghr/codex/mv_generalist_v2_han_fullgt_20261014`，service `mv3-generalist-v2-han-fullgt-20261014.service`。v1根目录保持只读历史，v1 service已inactive，不恢复。

旧v1首seed两组各40k并完成全开发推理，69/76开发指标时停止，其他seeds尚未启动。版本更改来自用户指令，不依据Dice，也不重新全面审计。仅Han改变有效区使用；v2双方从头训练，六组共同40k预算、首seed全来源指标后再补另外两seed。

本地被动收取器源码 `scripts/watch_mv3_results.py`，只等待v2终止，默认v2根目录。核验 `artifacts/mv3_control_20261014/collector_pid.json` 的server_root与实际PID命令；旧PID12992已停止，停止回执见collector_stopped_for_han_fullgt.json。原collector PID156036因SSH断线已退出；当前collector PID194460已重新启动且支持有界重连，server_root为v2；有48小时上限，完成时收取/push报告及指标，不启动实验；工作区不干净则留在collected_v2_results。服务器有界执行器独立于本地窗口。

Han三份原图/GT/mask并排照片在本地 `artifacts/mv3_control_20261014/Han24_review`。旧Han24_mask_review记录的是v1证据判断，其黄色图为原始mask非零可视化；在v2中不再用它定义有效区。

MV2已完成1692条核心指标，不追加旧9个head验证。当前156train/38dev区域、12来源，官方test/val保留。GPU0禁用，网站/环境/历史不改。每轮代码和MD提交推送，最终报告后停止。

以下内容为上一轮交接，旧MV2运行中记录不再代表当前状态。

当前研究仓库与网站分开。首PR保存服务器只读盘点、26条实验记录、115份精选来源快照；本轮另补664份本地历史源码/配置/记录，其中76份Markdown。精选实际运行代码位于 `src/`，本地历史代码位于 `archive/research/`，历史Markdown位于 `docs/history/`。每份历史文件的来源、字节数和SHA位于 `evidence/history_file_index.json`；归档本身不证明实验已运行。

首PR交付进度与实际合并回执以 [GITHUB](GITHUB.md) 和 `evidence/github_delivery.json` 为准。旧 `evidence/github_publication.json` 是首次发布的历史快照，不是实时状态。

服务器 `root@192.168.203.97`，研究根目录 `/opt/data/3_117/tanghr`。具体环境、模型和路径见 [SERVER_PATHS](SERVER_PATHS.md)、[ENVIRONMENTS](ENVIRONMENTS.md) 和registry。密码不入库。

P0、B100k、B150k及MV-1九个head已核验；B150k是100k+50k continuation。原版Primus已与锁定upstream按AST比对。无标签预训练暴露包括开发/测试图像，因此结论限定transductive development。保留Stage2A/Stage2A2的失败及负面结果。

最近服务器快照来自2026-10-13 19:55:21香港时间：已有 `mv-dev-v2-notta-20261013.service` 正在运行，partial inferred507/scored251。这不是当前实时状态，也不能写成完成。本次补交代码和记录没有连接服务器或新增实验。

按 [NEXT_ACTION](NEXT_ACTION.md) 只读收取现有MV-2无TTA结果，先确认是否完成，再核对3模型×3seeds及每组188个条目的覆盖。保持encoder、九个head、原版Primus、TTA=OFF及全部推理协议。旧四翻转目录已废弃，保留档案，不用于正式汇总。

`src/` 和 `archive/research/` 的脚本含原服务器绝对路径、依赖关系及历史watcher。归档未改变代码，也未赋予启动权限；不要批量执行。可直接运行的本仓库CPU校验见README。

每轮记录在 `docs/work_logs/`。结束前补充实际操作、验证、commit/PR、未完事项；事实变化时更新PROJECT_STATE、registry及NEXT_ACTION。先核验当前Git状态和远程，再继续工作。
