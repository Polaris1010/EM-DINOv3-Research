> 2026-10-22 00:57香港实测：D1/D2三decoder-seed任务active，首批D1-30k/D2-30k/D1-100k的seed20261015均100/40000，采样hash一致，GPU1/2/3各约14–15GiB总占用；其余5个head排队。原Coverage-NR继续到42340，网站200，GPU0未用。服务d12-three-seeds-20261022.service；先读docs/D12_three_seed_execution.md。只有head重复，没有新SSL；完成固定38dev及七组MB1后输出均值/样本SD和配对差值，停止。

> 2026-10-22 新授权：补齐D1/D2的30k、100k四encoder各三个decoder seeds，复用20261014、新增20261015/16共8个40k head；固定156train/38dev及七组MB1诊断，非三折或三SSL seed。先读docs/D12_three_seed_execution.md与experiments/D12-three-seeds/run_index.json。Coverage-NR既有SSL继续，GPU0禁用。当前部署准备，尚未声称训练已启动。

> 2026-10-21 23:06香港时间进度复核：Coverage-NR 33420/100000（33.42%），10k/30k EMA导出存在，最新full33000；GPU1/2/3工作、网站200。SSL已运行6.94h，按平均速度约余13.82h。head/dev/MB1尚未开始。按已消费ID位置重放CEM覆盖775442/1425433（54.40%）。本轮另安装用户级anti-defensive-writing；见work_logs/20261021_28_skill_install_progress.md。既有训练服务继续。

> 2026-10-21 16:11香港时间实测：Coverage-NR正式服务active，P0新初始化从0到60/100000；loss有限，GPU1/2/3利用率100%，各约11GiB，GPU0未用于本轮，网站200。code6129ac2，draft PR11。先读docs/Coverage_NR_execution.md与本轮run_index；已有服务继续，不重复启动。首seed40k及开发/MB1由同一有界服务顺序完成后停止。

> 2026-10-21 新授权：用户要求“继续，开始训练”。执行单组Coverage-NR从P0连续100k，继承既有MV+CEM池，之后首seed冻结Primus40k/38dev/固定MB1，完成停止。先读docs/Coverage_NR_execution.md及experiments/Coverage-NR/run_index.json。旧设计-only停止记录为历史，不重复启动服务，不增加R组/IsoOrg/QuantEM/D3，GPU0禁用。当前CPU计划通过，GPU smoke正在验证。

> 2026-10-21 metadata增量审查完成：见docs/Data_Expansion_Metadata_Review_20261021.md。当前P0设计/采样器/数据池不变；15个公开候选入口（8优先、7条件性），不是15个已证实独立acquisition。IsoOrg扣除历史MV与最终CEM同名来源后剩518子体积/18名称候选。没有图像下载、GT读取或训练，完成STOP。

> 2026-10-20 最新决定：D2-Coverage 改为从官方 P0 开始的完整100k方案，见 docs/D2_Coverage_P0_Design.md。取消30k恢复路线；旧设计和模拟保留为历史。全覆盖不能严格保持历史sqrt分布；需要因果比较时另建议同配额的P0有放回对照，未自动安排两组训练。当前仅设计更新，未启动训练，完成后STOP。

> 2026-10-20 D2-Coverage设计/CPU模拟完成，训练未授权未启动。先读docs/D2_Coverage_Experiment_Design.md：A67.05%、B81.46%、C100%；推荐同30k分支的全覆盖＋最小TV配额补偿C，69236步覆盖完，后70k集合TV最低15.7352%。完整checkpoint已只读核验，原D1/D2不变。当前codex/d2-coverage-design，仅提交设计/模拟/测试，完成STOP，不自动训练。

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

> MV5 最新授权（2026-10-15）：用户决定结束MV4B实例分组排查，开始同156训练区域/38开发区域的MitoNet监督适配。MV4B机制在已测四区域正式关闭，不延伸其余34。MV4C缺失原TIFF不阻塞本轮。先读docs/MV5_execution.md、experiments/MV-5/run_index.json及最新work log；只接续本轮有界执行器，不重复启动，不训练DINOv3，不启动新排查或SSL。下方旧“停止/等待”是历史，已被本次明确授权更新。

> MV4C可执行核查已结束并停止：Guay作者原图→Zarr全像素复现一致；两来源Zarr→实际cache/GT/既有概率坐标链一致。Müller转换前TIFF未取得，原图链明确UNVERIFIED，不能声称整项原图核验完成。当前分支codex/mv4c-input-provenance，draft PR5（base为MV4B分支，未合并）；先读docs/MV4C_input_provenance_report.md和experiments/MV-4C/audit_index.json。仅待真实high_c1原TIFF路径；不自动阈值/尺度搜索、反相推理或训练。

> MV4B已完成：四区域Semantic/Panoptic差异不明显，不扩展38。Guay Dice0.014253、Müller high_c1约2.35e-7未恢复；raw0.5掩码与旧Panoptic四区域逐像素相等。旧MV4/MV3全部保留，不能声称已完成38区域Semantic Only或公平独立测试超越MitoNet。当前分支codex/mv4b-semantic-panoptic，draft PR4（base为MV4分支，未合并）；先读docs/MV4B_final_report.md、experiments/MV-4B/run_index.json及work_logs/20261014_09_mv4b_semantic_panoptic.md。研究服务已停止，不自动启动下一实验。

> MV4已完成并停止（2026-10-14香港时间23:15:04核验）。同38开发区域/12来源，来源等权Dice：P0 0.813741、B100 0.826449、官方MitoNet 0.610209；B100在8/12来源更高。当前分支codex/mv4-official-mitonet，独立draft PR3，base为未合并MV3分支。先读docs/MV4_final_report.md、experiments/MV-4/run_index.json和work_logs/20261014_08_mv4_official_mitonet.md。服务inactive/MainPID0，指标266行核验PASS；不重启MV3/MV4或旧watcher，不自动启动新实验。

# 当前交付：MV4已完成

完整报告：[MV4_final_report.md](MV4_final_report.md)。G-P0/G-B100复用全部三个decoder seeds，没有重训/重推；MitoNet使用固定官方权重、默认整图二维推理、TTA OFF，实例输出保留服务器。Han整幅GT继续如实披露，Turner nm指标保留spacing限制；这是监督数据和训练暴露不同的完整系统开发对照。

实际运行代码db2ae4e；完整指标experiments/MV-4/MV4_per_region_metrics.csv，来源差距MV4_source_deltas.csv，最终校验MV4_results_verification.json。服务器根/实例路径见run_index.json和最终报告。代码、指标、本轮Markdown及交接随PR3提交推送；PR未合并，不假设main已包含本轮。

下一项仅为决策：优先考虑跨来源稳健性，关注Guay/Müller仍存在的B100相对P0退化，同时保留MitoNet在Casser/Mekuc/Turner/Wei的优势作为实际改进目标。用户后续明确授权前停止所有实验；不新增审计、诊断、预训练或延长head训练。

## 以下保留MV3及更早阶段的历史状态/计划

# 当前研究状态：MV3 v2已完成

2026-10-14香港时间17:10:18实际核验六组各40k、各38开发区域/12来源，共228条指标和18个checkpoint；服务器inactive/MainPID0，网站200、GPU0未使用，没有启动后续实验。最终报告与全部指标已收取并通过覆盖、混淆矩阵算术、SHA关联和三个seed配对核验。

P0 Dice 0.813741±0.000827，B100 Dice 0.826449±0.000832；配对Δ+0.012707±0.001361。监督扩展的描述性同区域收益P0+0.4670、B100+0.4359，包含来源覆盖和20k→40k预算等改变，不声称单变量因果。B100主要收益在OpenOrganelle/Han/Turner，Guay/Müller仍低于P0，Haberl基本持平。当前短板仍为Haberl/Han/Guay/OpenOrganelle；下一项决策是MitoNet同图像GT对照，本轮停止。

当前结果见docs/MV3_final_report.md、experiments/MV-3/v2/run_index.json和MV3_final_results_verification.json。MV1/MV2及v1历史记录保留。

## 以下为完成前与盘点历史状态

> 最新核查：2026-10-14 15:00香港时间，v2首seed两组40k及38区域×12来源完整开发评测已完成，Dice P0 0.8130 / B100 0.8273（+0.0143）。P0全部三个seed、B100前两个seed均已40k训练完成；追加seed的开发推理在运行，B100第三seed等待既定调度。尚无三seed最终报告。

# 当前研究状态

最新终止状态 20261014T091140Z：MV3 complete；实际产物见experiments/MV-3/v2，完整完成时报告为docs/MV3_final_report.md。

## 最新阶段：2026-10-14 MV-3

当前协议为 **MV-Generalist-v2**。用户明确指定“取消 mask 限制，整幅 GT 参与 loss”；仅 Han24 跳过 annotation mask，整幅 GT 参与采样、loss 和开发指标。其他来源、划分、冻结 encoder、原版 Primus、匹配 seeds、共同 40k 预算与 TTA=OFF 保持原配置。旧 v1 已停止并完整保留，不能混入 v2 对照。

v1首seed双方各40k与完整开发推理已完成；69/76指标时因用户改变Han规则而停止，自有service inactive。旧结果保留，v2真实smoke双方PASS，已从相同初始化重启；当前13200steps/组的dated快照见v2登记与evidence，不代表40k完成。

用户已明确授权进入MitoVerse多来源监督训练，取代初始化轮的只读范围。MV-2现已完成188区域×9组=1692条核心指标，12来源等权Dice：P0 0.3492、B100k 0.4067、B150k 0.4102；实证完成状态见 `evidence/MV-3/server_initial_status_20261014.json`。旧注册记录与结果保留原快照，不重写历史。

本轮唯一任务MV-3：G-P0/G-B100 frozen encoder + original Primus，统一40k预算、3匹配seeds，TTA=OFF。当前锁定MV-Generalist-v2：156 train区域和38 dev区域，覆盖12来源；194个区域输入准备完成。Han按用户确认忽略mask、整幅GT参与loss/采样/指标，正式test排除；详情见MV3_PROTOCOL。

实际smoke/训练/开发验证状态以 `experiments/MV-3/v2/run_index.json` 和最新带时间戳服务器快照为准，不能根据PLANNED字段声称已完成。旧9个head不再扩展验证。不启动200k SSL、B150k训练或新增模块。研究代码和每轮Markdown提交到 `codex/mv3-generalist-training`。

以下2026-10-13盘点为保留的历史状态，MV-2曾running与main曾只有scaffold均已被后续实际交付取代。

本轮事实快照：2026-10-13 19:33–19:36（Asia/Hong_Kong；精确 UTC 见 `evidence/collection_receipt.json`）。SSH 对服务器进行只读盘点，CPU 读取 checkpoint 元数据，无新训练/推理、环境安装、网站变更或进程停止。运行中的任务状态会变化，此文件不是实时监控。

## 状态定义

| 状态 | 含义 |
| --- | --- |
| VERIFIED_COMPLETED | 已找到实际完成记录，且关联配置/指标/资产支持完成结论 |
| VERIFIED_FAILED | 有实际失败诊断或协议废弃记录；不等于训练崩溃 |
| PAUSED | 用户暂停，有历史状态记录 |
| PLANNED | 未来行动，未执行且不自动授权启动 |
| UNVERIFIED | 缺少当前任务所需的核验，不能填猜测值 |
| VERIFIED_RUNNING | 本轮已找到活动服务、进程和部分输出；完成状态仍未核验 |

## 模型与实验

| 资产/实验 | 状态 | 本轮证据与含义 |
| --- | --- | --- |
| P0 官方自然图像 ViT-L/16 | VERIFIED_COMPLETED | 文件存在，CPU 安全读取 state dict；SHA 使用服务器既有登记 |
| B_full 100k | VERIFIED_COMPLETED | `result.json`=complete/100000，checkpoint sidecar、teacher/student metadata 对应；旧 rank-log 审计为本地历史快照 |
| B_full 150k | VERIFIED_COMPLETED | 从100k完整状态额外续训50k，固定1e-6；不是从零的150k调度；完整状态和teacher/student均存在 |
| CMCC/CMCT 原版 Primus 六组 | VERIFIED_COMPLETED | `results.json`、逐volume表与原版源码校验记录；四翻转TTA，GT==1；不是MitoVerse推理协议 |
| MV-1 九个原版 Primus head | VERIFIED_COMPLETED | 3 encoders × 3 seeds，4 train/4 dev，20000 steps；九个head安全读取且元数据身份一致，均44576514参数；TTA=OFF |
| MV-2 无TTA扩展 | VERIFIED_RUNNING | 活动service与三组inference进程、scoring进程及部分输出；188个volume/ROI，12来源，尚无完整结果声明 |
| MV-2旧四翻转版本 | VERIFIED_FAILED（协议废弃） | `superseded_tta_status.json`明确因用户纠正而废弃；旧结果保留，排除出正式MV-2 |
| Stage2A topology calibration | VERIFIED_FAILED | gate=FAIL，AUTO_LAUNCH_T1=false，T1=NOT_STARTED；424张label-free切片的诊断失败不能写成成功 |
| Stage2A2-P | PAUSED | 本轮读取服务器历史STATUS中的用户暂停状态 |
| Stage2A2-E | VERIFIED_COMPLETED | 旧冻结probe基础设施完成记录；与后来原版Primus/MV-1 head分别登记 |
| Stage2A2-S | VERIFIED_COMPLETED（负面pilot） | 量化pilot及424×4真实标签诊断完成；不授权长训，最终human gate尚未核验 |
| 旧31-volume B/L/T/LT | UNVERIFIED（完成细节） | 历史根目录在已发现目录列表中；保留本地控制代码与服务器历史，不恢复watcher，不算全量正式结果 |
| Gram/HRA/LAST/Topology在新B_full中 | OFF / NOT_RUN | B_full配置和实际trainer/result不包含这些正式阶段；旧P1不得混入 |
| B_clean | PLANNED | 图像隔离的未来方案，不阻塞当前开发评测 |

全部26条具体记录（含每seed和六个CMCC/CMCT结果）见 `experiments/registry.yaml`。九个MV-1 head与CMCC/CMCT head不能混用。

## 当前科学证据

原版Primus历史Dice：

| Dataset | P0 | B100k | B150k |
| --- | ---: | ---: | ---: |
| CMCC | 0.784606 | 0.889716 | 0.878806 |
| CMCT | 0.954885 | 0.960962 | 0.960518 |

MV-1三个decoder seeds的来源均值（本轮复制真实结果表）：

| 来源 | dev volumes | P0 Dice | B100k Dice | B150k Dice |
| --- | ---: | ---: | ---: | ---: |
| guay21 | 1 | 0.756738 | 0.745874 | 0.746829 |
| jiang25 | 1 | 0.867301 | 0.883872 | 0.882137 |
| muller21 | 2 | 0.893847 | 0.883981 | 0.884007 |

B100k并非在首批所有来源都改善；目前只说明小规模开发表现，不能外推全MitoVerse。标准差与每volume指标保存在CSV，seeds数量与生物采集体数量分别理解。

B100k/B150k暴露了完整MitoVerse无标签池，是transductive SSL development assets；不能宣称已暴露图像是完全未见图像泛化。MitoVerse发布者proofreading声明和既有数值QC，不是独立人工重标注完整性认证。

## 本轮发现的记录冲突

1. DINOv3 Git根目录为 `/opt/data/3_117/tanghr/DINOv3/dinov3`；父目录不是Git仓库。HEAD确为91ffb680…，另有未跟踪文件。
2. MedDINOv3共享仓库HEAD是bd6a650…，存在历史修改；不等于本项目原版decoder的锁定上游a2460c2…。实验实际独立使用 `mv_generalist_stage1_20261013/src/official_primus.py`，SHA b5398dc…，与上游源码AST相同，仅末尾LF差异。
3. 原版CMCC/CMCT目录的 `model_registry.json` 仍描述历史频域门控factory；已完成 `results.json`、独立源码和审计优先，不把旧factory字段套到新结果。原件保留。
4. Stage2A2 STATUS仍写B150k ready_after_resume_smoke；后续实际B150k result/checkpoint证实完成。该旧STATUS不是新的训练状态。
5. 用户交接说明把MV-2执行状态留待查询；本轮已证实它在运行。下一项行动是收取它的结果，不重复启动。
6. 废弃四翻转目录的旧STATUS仍显示running；本轮service已inactive/MainPID=0，superseded标记优先。这是未更新的旧文件，不是第二个正在运行的MV-2。

最后一次只读查询：2026-10-13 19:55:21（香港时间），正式无TTA service仍active，inferred=507、scored=251（部分结果计数，不是完成体积数），网站HTTP200。精确快照见 `evidence/server_status_20261013T115521.json`。

B150k目录保留的 `REPORT_100k_150k_zh.md` 是历史modified/gated Primus下游比较，其数值与后来的原版Primus表不同，不能作为原版结果引用。原版六组以 `experiments/CMCC_CMCT/results.json` 为准。

大checkpoint没有重新计算完整SHA；使用既有metadata/latest_checkpoint/input_hashes记录，并另行读取CPU安全metadata。两份full optimizer checkpoint在weights_only=True下不接受其中numpy状态，本轮没有切换到不受限反序列化；teacher/student和九个head安全读取通过，当前评测不依赖full optimizer加载。

## 仓库与未完成事项

本地独立仓库：`E:\mira_website_demo\EM-DINOv3-Research`。主分支main仅为初始scaffold；首PR分支为用户指定的 `chore/bootstrap-research-registry`。网站工作区既有修改未改动。

GitHub远程：[Polaris1010/EM-DINOv3-Research](https://github.com/Polaris1010/EM-DINOv3-Research)，已核验private。用户确认账户与浏览器授权后，本轮恢复GitHub认证并对owner repositories完整分页查重，未发现同名或相关研究仓库，随后创建独立private远程。首PR的实际链接与交付状态见 `docs/GITHUB.md`，科学快照时间不因远程交付而改变。
