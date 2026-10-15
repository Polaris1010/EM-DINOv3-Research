> MV5完成状态已现场复核：40k和38区/12来源全部完成，服务inactive/MainPID0/exit0。结果已推送0933540；最终报告docs/MV5_final_report.md，最新本轮记录docs/work_logs/20261015_12_mv5_completion_recheck.md。自动收取器已退出，禁止按下方历史running通知恢复任务；本次仅修复PR交付回执与元数据，未启动新实验。

# MV-5 执行协议与交接

用户2026-10-15明确授权MitoNet在MV3相同156训练区域上监督适配，并在相同38开发区域/12来源比较。MV4B实例分组丢前景假设在已测四区域正式关闭，不扩展34区域；旧MV4指标原样保留。MV4C的Müller原始TIFF缺口不阻塞新任务。

配置锁定于configs/evaluation/MV-5_protocol.json；一个seed20261014，40k真实optimizer updates，10k/20k/40k保存，固定40k评测。MitoNet全模型微调，官方PanopticLoss/目标/训练forward及AdamW/OneCycleLR；batch4、source-aware原生256采样和双翻转。没有使用napari全部默认增强或batch16，不能称逐项默认复现。与冻结DINOv3的batch1、优化器及曝光次数不同，不能声称等计算/等优化预算消融。

多ID标签保持原ID关系；只有二值crop时做2D四连通伪实例，不当作真实独立实例GT认证。GT前景逐crop与原MV3缓存逐像素一致。人工补边忽略全部四项loss，真实像素不裁减；Han仍忽略mask整幅GT。原test/val和所有训练/开发ROI边界不变。无阈值或尺度搜索，推理TTA OFF。

已通过十二来源短smoke：55158486参数训练，四项loss与梯度有限，checkpoint回读推理逐元素一致。CPU验证无padding时与官方loss完全相同，改变padding预测/PointRend点不影响loss且梯度为零。smoke不计入正式40k；正式从发布权重重新加载，未以任何开发Dice选择配置。

服务器独立根：`/opt/data/3_117/tanghr/codex/mv5_mitonet_adaptation_20261015`。训练只用物理GPU3，评估仅1/2/3；GPU0禁用，网站与未知进程保留。现有M4隔离venv只读复用，不安装依赖。`STATUS.json`是实际状态；完成前不得声称有MV5指标。服务名mv5-mitonet-adaptation-20261015.service，最多48小时，不重启重训。训练完成自动执行38区推理、统一物理指标和MV5_final_report.md，随后停止。

本地只读收取器将只收本轮小型指标/配置/哈希/Markdown；原图、checkpoint和大量预测留服务器。若窗口结束或收取器故障，新窗口先读此文件和run_index，再只读检查服务；不能盲目重启。所有结果保留四系统分开与逐区域配对，不把MV4B四区验证扩张为38区结论。
