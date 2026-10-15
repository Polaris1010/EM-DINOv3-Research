# MV5同监督区域适配启动记录

用户明确关闭MV4B实例分组丢前景机制，转向相同156监督区域的MitoNet适配。本轮接续代码分支codex/mv5-mitonet-adaptation（父MV4C，历史保留），不继续灰度/尺度/SemanticOnly检查，不改DINOv3。

已核对官方发布TorchScript保留训练forward，使用PanopticLoss的CE/中心MSE/偏移L1/PointRend损失；官方fine-tune支持finetune_layer=all。固定一个seed20261014、40k真实更新，10k/20k/40k保存与固定40k开发评测，source-aware原生crop。官方训练接口与loss/target源码按字节SHA保存；当前上游main的commit未单独核验，明确UNVERIFIED，不猜测。

首次smoke在正式训练前发现小于256的训练区域，未训练即停；为人工padding加入全loss忽略并重跑短smoke。十二来源每源一步、共12步通过，55158486参数可训练、checkpoint推理回读一致。CPU实际验证补边输出/点扰动不改变loss且无梯度；无padding时逐值等同官方loss。四个CPU loader workers预取目标，GPU3训练，避免串行读取造成GPU等待；不增加训练区域或配置搜索。

实际开跑、实时步数、commit及PR回执将在同文件追加；在实际40k与38区指标完成前，不填写完成结果。每轮代码、协议与本Markdown一并提交推送。

## 实际正式开跑

执行代码commit `2cbc7c773ca94271ce22b0d0e4820b745fde38b7` 已推送；服务器部署逐文件SHA与该版一致，正式service于2026-10-15 02:58:38 UTC启动，MainPID3878，训练worker3886，物理GPU3。采集快照已完成 2600/40000真实更新，尚无开发指标；十二来源均参与采样。网站200，GPU0保留原进程。Github插件私有访问404，本机已授权Polaris1010认证核验private后创建draft PR6，base为未合并MV4C分支，不自动合并。

收取器 `scripts/watch_mv5_results.py` 仅被动读取本轮服务、终止后收取结果并提交推送；48小时硬截止，凭据只在内存。若分支切换或有无关改动，只保存scratch结果，不混入用户修改。训练/推理/新实验启动不在收取器能力内。

实际被动收取器已连接，PID 31336（控制台确认MV5_PASSIVE training），48h上限。新增4项只用fake SSH/临时目录的收取边界回归检查；共25项CPU测试通过，覆盖认证不重试、重连截止不延长、脏工作区不写入以及发布失败不重复提交。

## 首个里程碑

实际快照 10600/40000更新；10k脚本checkpoint与optimizer/scaler resume均已保存，权重SHA `de809ca4076907ca8ab6b54aef0530a288d5029890d7729890aef682d0995d97`，元数据已收取，权重不入Git。固定40k选模；本轮未进行10k开发Dice选型，没有开发指标。服务active、GPU3训练、网站200，GPU0未使用。被动收取器已连接并等待最终终止；正式结束后收取304行四系统记录与最终报告，提交推送后停止。

## 实际终止收取 20261015T035253Z

MV5已完成：MitoNet-MV 40k、38区域/12来源开发指标完成；先读docs/MV5_final_report.md。任务结束，不启动新实验。 原40条registry保持；实际checkpoint SHA、指标、配置与本轮Markdown同步提交推送至draft PR6，不自动合并。自动收取未启动任何实验。
