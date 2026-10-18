> D1/D2新授权（2026-10-18）：用户要求两组共同继续100k。各自完整30k状态续训70k，独立根ssl_diversity_d12_100k_20261018；固定100k新Primus首seed40k和38dev/七组MB1诊断，旧30k不改。先读docs/D12_100k_execution.md及experiments/D12-100k/run_index.json。本段覆盖下方旧不自动100k通知；不自动D3、更长SSL或额外seed。当前仅准备，未把计划写成已启动。

> D12完成复核（2026-10-18）：D1/D2各30k SSL和同seed40k Primus、全部开发/MB1诊断完成并停止。开发Dice0.817542→0.829407，但MB1七组Dice0.488654→0.461614；不能仅凭内部提升声称跨域改善。先读D12_final_report.md末尾完成复核与最新work log；不自动100k/D3或更多seed。

> D1/D2终止状态 complete，UTC 20261017T204218Z。 两组30k SSL、首seed40k Primus、76条开发指标及5190条固定MB1诊断均完成；开发Dice D1=0.817542，D2=0.829407，D2−D1=+0.011864。先读docs/D12_final_report.md。旧资产前后快照一致，网站200，GPU0未用，不自动100k、D3或更多seed。

> D1已从2711步完整状态接续（UTC 2026-10-17T06:33:19.240669+00:00核实至2720/30000）：心跳原子发布修复与Linux并发测试通过，训练器/配置/预算不改，GPU1/2/3工作、GPU0未用、网站200；D2与下游评测尚未开始。旧失败记录保留于7b29435，本次接续代码545ebba；先读docs/work_logs/20261017_15_d12_interruption_resume.md及experiments/D12/run_index.json。下方failed/停止是已保留历史，不代表本次恢复状态。

> D1/D2终止状态 failed，UTC 20261017T032244Z。 保留实际产物，查看STATUS与服务器阶段日志；未完成部分不得声称完成，不自动改参/重启。

> D1/D2最新授权（2026-10-17）：执行来源均衡MitoVerse vs50:50 CEM的匹配30k SSL，均从自然DINOv3开始；采用MitoNet排除benchmark及关联图像的原则，新共享留出池。两组smoke通过，正式事实见experiments/D12/run_index.json、服务器STATUS和docs/D12_execution.md。之后首seed固定MV3 Primus40k/开发/已暴露MB1诊断；不修改历史、不用GPU0、不自动100k/D3。下方旧停止通知由本次明确授权更新。

> MB1已完成并停止（核验UTC 2026-10-15T08:01:15.067219+00:00）：七组、八run、20760条二维指标和九张固定对照图全部完成。G-P0 Dice 0.457938；G-B100 Dice 0.502665；Official-MitoNet-v1 Dice 0.712154；MitoNet-MV Dice 0.240936。先读docs/MB1_summary.md、docs/MB1_overlap_status.md、experiments/MB-1/MB1_local_verification.json和最新work log。五组B100 SSL来源与抽样原图重叠，Lucchi++精确区域未确认，TEM未发现名称匹配；不能称严格未见图像外测。服务inactive/MainPID0，网站200，旧资产不变。任务结束，不启动新实验。

> MB1最新授权（2026-10-15）：只做EMPIAR-10982全部七组benchmark，复用MV3六个head及MV4/MV5两个MitoNet，TTA OFF。MV3/4/5结果保持；先读docs/MB1_execution.md、experiments/MB-1/run_index.json和最新work log。已有五组来源进入B100 SSL池，不能称严格未见图像外测；Lucchi++来源关联但精确区域未确认，TEM未发现名称重叠而非证明无暴露。旧停止通知由本次明确授权更新；结束MB1后仍停止。

> MV5已完成：MitoNet-MV 40k、38区域/12来源开发指标完成；先读docs/MV5_final_report.md。任务结束，不启动新实验。 UTC 20261015T035253Z

> MV5 最新授权（2026-10-15）：用户决定结束MV4B实例分组排查，开始同156训练区域/38开发区域的MitoNet监督适配。MV4B机制在已测四区域正式关闭，不延伸其余34。MV4C缺失原TIFF不阻塞本轮。先读docs/MV5_execution.md、experiments/MV-5/run_index.json及最新work log；只接续本轮有界执行器，不重复启动，不训练DINOv3，不启动新排查或SSL。下方旧“停止/等待”是历史，已被本次明确授权更新。

> MV4C可执行核查已结束并停止：Guay作者原图→Zarr全像素复现一致；两来源Zarr→实际cache/GT/既有概率坐标链一致。Müller转换前TIFF未取得，原图链明确UNVERIFIED，不能声称整项原图核验完成。当前分支codex/mv4c-input-provenance，draft PR5（base为MV4B分支，未合并）；先读docs/MV4C_input_provenance_report.md和experiments/MV-4C/audit_index.json。仅待真实high_c1原TIFF路径；不自动阈值/尺度搜索、反相推理或训练。

> MV4B已完成：四区域Semantic/Panoptic差异不明显，不扩展38。Guay Dice0.014253、Müller high_c1约2.35e-7未恢复；raw0.5掩码与旧Panoptic四区域逐像素相等。旧MV4/MV3全部保留，不能声称已完成38区域Semantic Only或公平独立测试超越MitoNet。当前分支codex/mv4b-semantic-panoptic，draft PR4（base为MV4分支，未合并）；先读docs/MV4B_final_report.md、experiments/MV-4B/run_index.json及work_logs/20261014_09_mv4b_semantic_panoptic.md。研究服务已停止，不自动启动下一实验。

> MV4已完成并停止（2026-10-14香港时间23:15:04核验）。同38开发区域/12来源，来源等权Dice：P0 0.813741、B100 0.826449、官方MitoNet 0.610209；B100在8/12来源更高。当前分支codex/mv4-official-mitonet，独立draft PR3，base为未合并MV3分支。先读docs/MV4_final_report.md、experiments/MV-4/run_index.json和work_logs/20261014_08_mv4_official_mitonet.md。服务inactive/MainPID0，指标266行核验PASS；不重启MV3/MV4或旧watcher，不自动启动新实验。

> 2026-10-14：MV3 v2六组40k与228条开发指标已完成，服务inactive。该轮授权任务已结束；先读最终报告，禁止恢复运行或自动MitoNet/新SSL。后续用户明确任务可提供新的执行授权。

# Research takeover instructions

每个新窗口先读 `START_HERE.md`、`docs/NEW_WINDOW_HANDOFF.md`、最新 `docs/work_logs/` 记录，再读 `docs/PROJECT_STATE.md`、`experiments/registry.yaml`、`docs/NEXT_ACTION.md` 和最新已合并 PR。不能声称尚未发生的提交、推送或合并已经完成。

每轮完成用户授权的代码、配置、报告修改后，同步新增工作记录、更新交接和记录索引；事实变化时更新状态和registry。完成适用CPU检查后，将本轮代码和Markdown一并commit并push，汇报实际commit、分支、PR状态。用户已要求持续保留每次md记录和下一个窗口所需文件；不得只在聊天里总结。

## Evidence and claims

可信顺序：实际 resolved config、checkpoint metadata、源码 commit/SHA 和运行日志；已合并 PR 与实验注册表；研究报告；聊天计划。保留冲突原件，解释采用哪一项记录，不能把旧计划写成完成事实。状态使用 VERIFIED_COMPLETED、VERIFIED_FAILED、PAUSED、PLANNED、UNVERIFIED；已有任务正在运行时另用 VERIFIED_RUNNING，不能把它写成完成。缺失字段填 UNVERIFIED，禁止猜测路径或默认值。

研究目标：开发跨来源通用 2D EM 线粒体分割模型，研究 DINOv3 可迁移自监督表征，在公平协议下与官方 MitoNet 比较。优先复用既有模型和结果。

## Scope and resources

当前协议为 **MV-Generalist-v2**。用户明确指定“取消 mask 限制，整幅 GT 参与 loss”；仅 Han24 跳过 annotation mask，整幅 GT 参与采样、loss 和开发指标。其他来源、划分、冻结 encoder、原版 Primus、匹配 seeds、共同 40k 预算与 TTA=OFF 保持原配置。旧 v1 已停止并完整保留，不能混入 v2 对照。

2026-10-14用户已明确授权MV-3多来源监督训练，取代前一轮“仅只读初始化”的范围限制。仅训练G-P0与G-B100的全程冻结encoder+原版Primus，在锁定MV-Generalist-v2上共享40k预算、10k/20k/40k保存点、固定40k选模和3个匹配decoder seeds；首seed全来源开发指标完成后再补另两seeds。允许仅为这6个run使用本轮有界执行器；不恢复旧watcher，不启动其他模型或模块。

禁止修改网站、环境包或MV-1/MV-2历史checkpoint/结果/注册记录，禁止自动新预训练或MitoNet。物理GPU0永远禁用；只能使用1/2/3，先核验既有进程和网站。不停止未知进程。MV-2已完成，不再追加旧9个head的推理或分析任务。结束时输出MV3_final_report.md并停止。

MV-Generalist-v2是当前监督协议：正式test与已映射官方validation保留；允许将自建探索性dev重新分配到train，当前train/dev空间互斥。沿用已有source registry、proofreading声明和有效区规则；用户明确取消Han mask限制，Han整幅GT参与采样/loss/指标；mask文件不读取；不发起全面数据审计。单acquisition来源可使用不重叠空间区域，必须记录开发评测独立性限制。

扩大 MV-1 验证范围时，只改变验证覆盖。保持 encoder、九个 head、原版 Primus、TTA=OFF、normalization、sliding window、threshold、sampling、checkpoint-selection rule。CMCC/CMCT 的历史四翻转 TTA 不迁移到 MitoVerse。禁止增加 FFT gates、LAST、Gram、HRA、TopoMask 或新模块。

## Research decisions

问题分 BLOCKING、NON_BLOCKING、FUTURE。权重不能安全加载或图像/GT 尺寸不匹配，阻塞相关输入；acquisition 独立性未证实，不阻塞当前 transductive 开发验证，只限制严格泛化结论。B_clean 属于未来独立方案。不无限增加 NO_GO 或把诊断失败改写成成功。

B100k/B150k 是 transductive SSL development assets。无标签暴露不是监督标签泄露；其暴露图像结果不能声称为完全未见图像泛化。三个 decoder seeds 不增加独立生物 acquisition 数量。proofreading 声明与数值 QC 不等于独立重标注完整性认证。

## Repository and PR discipline

只提交必要源码、配置、小型表格、元数据、环境记录和脱敏证据。禁止提交原始图像/Zarr、checkpoint、大量预测、凭据、私钥、敏感 .env、网站配置、运行缓存或大日志。服务器路径只表示资产位置，不保证其他人可访问。远程必须 private；确认账户和创建方式，先复用同目标仓库。

`src/` 是原样历史源码快照，不因新位置就假定可启动。使用前读 `src/README.md`。不重构成熟框架。

每个 PR 说明上一版本实际状态、本次变量、固定配置、实际执行、指标变化和下一项科学决策。PR 不自动合并。更新事实时保存新的带时间戳快照和来源索引，不覆盖已有证据。用 `python scripts/update_registry.py` 从证据生成注册表，随后运行 `python scripts/update_registry.py --check`、`python scripts/validate_repository.py`、`python -m unittest discover -s tests -v`。这些检查只用 CPU，不运行实验。
