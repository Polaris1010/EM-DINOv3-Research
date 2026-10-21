# Coverage-NR：P0连续100k执行记录

2026-10-21用户明确授权“继续，开始训练”，覆盖前轮设计-only停止状态。本轮只运行一组Coverage-NR，保持当前MitoVerse+CEM池；不启动Coverage-R，不加入IsoOrg/QuantEM或新loss。

从官方P0重新初始化student、EMA、optimizer和loss centers，连续100k LR/warmup5000，global48/seed0，DINO+iBOT+0.1KoLeo；10k/30k/100k导出。之后固定100k EMA、原版Primus seed20261014/40k、156train38dev、TTA OFF，以及七组已暴露MB1诊断，完成停止。

CPU逐一验证全部480万domain位置；2398767 CEM槽位、2401233 MV槽位。每集合n_i+最大余数((M-N)*sqrt(n_i)/sum(sqrt(n)))配额，集合列表全程打乱，集合内按轮次无放回。实际计划覆盖1425433个ID（100%），每图最多51次。3rank/2worker顺序重排及不同step恢复的432个完整位置记录一致，损坏状态拒绝。计划SHA bf56bad49c74eb484f0eee125370f31c386379c56b8ea877a8467fc4685d500e；此为计划覆盖，不冒充已训练覆盖。

复用src/d12_100k的优化循环、图像adapter、官方增强、readout与benchmark；新代码在src/coverage_nr，原文件不变。训练器唯一算法变量为image-ID计划；另采用此前设计已披露的连续LR。历史D2仅系统参考，不能将结果差值全部归因于coverage。

服务器新目录/opt/data/3_117/tanghr/codex/coverage_nr_p0_20261021；环境沿用现有两Python，无包升级。网站进程保留，GPU0禁用，UUID映射只暴露1/2/3。正式服务Restart=no，单进程锁、心跳和网站保护继承既有控制器；失败停自身进程，不自动改参或追加实验。

短smoke包括10步新初始化和接续到12步，验证loader/finite loss/三rank参数同步/full checkpoint/RNG及sampler恢复。以smoke.json和STATUS实际收据为准。正式启动后使用服务与progress记录，不把计划写成完成。

预计SSL约21小时（三GPU），head约51分钟，开发/MB1另计。因保留10k/30k/100k三个里程碑full状态及两份最新恢复状态，峰值SSL checkpoint需预留约45GB，连同head/预测建议65GB以上；实际数据盘空间充分。训练期间已有ID计划/权重不覆盖。

新窗口：先读本记录和experiments/Coverage-NR/run_index.json，查看既有服务/STATUS；不要重复启动服务，不重建计划。完整恢复时只允许本run latest full checkpoint和相同plan/config/code，禁止用30k历史D2续训替代。结果生成后收取小型指标/报告并更新本轮registry、Markdown和PR。
