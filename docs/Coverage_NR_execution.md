> Coverage-NR已完成并停止（2026-10-22香港23:11核实）：SSL100k、head40k、38dev和2595张MB1均齐，GPU评分完成；开发Dice0.831821、MB1 Dice0.440027，低于D2-100k同seed0.841379/0.475754。CEM图像ID覆盖100%。服务inactive/MainPID0/退出0、网站200。先读docs/Coverage_NR_final_report.md和work_log36；本轮结束，不自动重启或新实验。

> 2026-10-22香港15:42：GPU评分等价性通过并部署。独立GPU校验误触旧controller未知PID保护，原任务94450停止；现已从94000 full状态恢复至94080，GPU1/2/3均100%、网站200。新服务coverage-nr-p0-gpu-resume-20261022.service；只接续原预算，不重复新初始化。详情work_logs/20261022_35_gpu_first_execution.md。

# Coverage-NR：P0连续100k执行记录

2026-10-21用户明确授权“继续，开始训练”，覆盖前轮设计-only停止状态。本轮只运行一组Coverage-NR，保持当前MitoVerse+CEM池；不启动Coverage-R，不加入IsoOrg/QuantEM或新loss。

从官方P0重新初始化student、EMA、optimizer和loss centers，连续100k LR/warmup5000，global48/seed0，DINO+iBOT+0.1KoLeo；10k/30k/100k导出。之后固定100k EMA、原版Primus seed20261014/40k、156train38dev、TTA OFF，以及七组已暴露MB1诊断，完成停止。

CPU逐一验证全部480万domain位置；2398767 CEM槽位、2401233 MV槽位。每集合n_i+最大余数((M-N)*sqrt(n_i)/sum(sqrt(n)))配额，集合列表全程打乱，集合内按轮次无放回。实际计划覆盖1425433个ID（100%），每图最多51次。3rank/2worker顺序重排及不同step恢复的432个完整位置记录一致，损坏状态拒绝。计划SHA bf56bad49c74eb484f0eee125370f31c386379c56b8ea877a8467fc4685d500e；此为计划覆盖，不冒充已训练覆盖。

复用src/d12_100k的优化循环、图像adapter、官方增强、readout与benchmark；新代码在src/coverage_nr，原文件不变。训练器唯一算法变量为image-ID计划；另采用此前设计已披露的连续LR。历史D2仅系统参考，不能将结果差值全部归因于coverage。

服务器新目录/opt/data/3_117/tanghr/codex/coverage_nr_p0_20261021；环境沿用现有两Python，无包升级。网站进程保留，GPU0禁用，UUID映射只暴露1/2/3。正式服务Restart=no，单进程锁、心跳和网站保护继承既有控制器；失败停自身进程，不自动改参或追加实验。

短smoke包括10步新初始化和接续到12步，验证loader/finite loss/三rank参数同步/full checkpoint/RNG及sampler恢复。以smoke.json和STATUS实际收据为准。正式启动后使用服务与progress记录，不把计划写成完成。

预计SSL约21小时（三GPU），head约51分钟，开发/MB1另计。因保留10k/30k/100k三个里程碑full状态及两份最新恢复状态，峰值SSL checkpoint需预留约45GB，连同head/预测建议65GB以上；实际数据盘空间充分。训练期间已有ID计划/权重不覆盖。

新窗口：先读本记录和experiments/Coverage-NR/run_index.json，查看既有服务/STATUS；不要重复启动服务，不重建计划。完整恢复时只允许本run latest full checkpoint和相同plan/config/code，禁止用30k历史D2续训替代。结果生成后收取小型指标/报告并更新本轮registry、Markdown和PR。

## 正式启动事实

2026-10-21 16:11香港时间：已从P0新状态运行到60/100000（实测快照，并非实时值），服务active，GPU1/2/3各100%利用率，网站200。实际运行code6129ac2、PR11 draft。收据evidence/Coverage-NR/launch_snapshot.json；配置experiments/Coverage-NR/resolved_config.yaml。被动收取器在本机存活时会等待本轮终态、收取小型报告/指标并提交推送；若本机断线，可由下个窗口从服务器同一目录收取，服务器训练不依赖本地收取器。
