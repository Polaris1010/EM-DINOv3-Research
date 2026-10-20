# D1/D2共同延长至100k

用户2026-10-18明确授权“继续100k，看看效果如何”，覆盖旧30k任务结束时不自动延长的限制。D1和D2各从自己已完成30k的完整状态继续70k，累计100k；不是重跑B100，也不只延长D2。

严格恢复student、EMA teacher、AdamW、DINO/iBOT状态、每rank RNG和绝对样本索引，父full-state SHA、数据池、sampler、官方源码、global batch48、seed0和格式适配验证一致。全程物理GPU1/2/3，GPU0禁用，网站和既有进程保持。

原30k余弦已走到min_lr约1e-6。续训采用共同100k余弦日程：step>=30000时lr=min+(peak-min)*(1+cos(pi*(step-5000)/95000))/2，peak=2.598076211353316e-5，min=1e-6。恢复点LR约2.195e-5，存在明确跳变，不假装等价于第0步开始的100k训练。两组日程完全一致；不额外重置优化器、不增加新loss。

固定导出50k和100k，50k只留档，不据dev/MB1选模型。两组固定100k EMA encoder各重新训练首seed20261014的原版冻结Primus40k，继承MV-Generalist-v2的156train/38dev、Han整幅GT、原评估器及TTA OFF；不复用旧encoder的head。然后完成38区域/12来源开发和既有MB1七组诊断，比较D2−D1及各自100k−30k，不看中途成绩改配置。

MB1已参与开发，不能称盲测；未知CEM别名/采集体关系仍UNVERIFIED。无标签暴露不等于监督泄漏。历史B100的池和日程不同，不把D1−B100认作纯采样因果效应。

独立服务器根为/opt/data/3_117/tanghr/codex/ssl_diversity_d12_100k_20261018；旧30k、MV1至MV5、MB1均保持。输出D12_100k_final_report.md、逐区域/来源/MB1指标、配对变化、100k_vs30k表、checkpoint SHA、实际曲线/抽样计数及历史前后快照，提交推送后停止。不自动D3、更长SSL、更多seed或超参搜索。
