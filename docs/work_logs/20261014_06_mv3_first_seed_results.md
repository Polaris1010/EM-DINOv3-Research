# MV3首seed完整开发结果与当前进度

用户请求核查40k完成情况，并展示已有开发指标。本轮只读读取现有v2服务器产物，没有修改监督协议、训练配置、数据划分或GPU安排，没有启动新科学任务。

香港时间2026-10-14 14:57:24（UTC06:57:24）快照：service active；seed20261014的G-P0/G-B100均完成40k训练与38区域/12来源全开发评测，共76/76指标。seed20261015两组40k训练完成，开发评测待执行；P0 seed20261016已到40k checkpoint后的监测收尾，尚无training_complete回执；B100 seed20261016等待既定调度。

首seed12来源等权Dice：P0 0.8130427361，B100 0.8273065136，配对差值+0.0142637775。B100在10/12来源高于P0，Guay -0.0140939327、Müller -0.0068326782；Haberl、Han和OpenOrganelle为当前较低来源。仅一个decoder seed，不当作三seed最终结论；MV2与当前监督划分/预算/Han有效区不同，不直接把两阶段总均值相减作为严格扩展收益。

原样收取MV3_first_seed_report.md、first_seed_summary、76条原始CSV、首seed训练/推理/验证回执及10k/20k/40k checkpoint sidecar。额外导出24条来源汇总和38条逐区域P0/B100配对CSV；逐来源及宏平均与原始76条记录交叉校验至1e-12。源数据、真实checkpoint、预测和凭据不入Git。

本地被动collector因WinError10054 SSH连接重置而退出；服务器service和任务实际继续正常。修复仅收取器：重连仅处理传输错误，每次30秒退避，原48小时截止不随重连延长；认证错误不重复尝试，Git发布错误不混入SSH重试。3项模拟断线/截止/认证测试通过，总13项CPU测试通过。已重新启动collector PID194460，只读取现有v2任务，不改变训练/网站/环境。

新窗口读首seed报告、当前run_index和最新快照；后续两seed仍需完整开发指标，最终MV3_final_report.md尚未生成，不自动启动MitoNet或SSL。旧v1及原26条历史/6条v1 registry快照保留。本轮代码、结果与Markdown一并提交推送现有draft PR2。

最新15:00:13香港时间快照（UTC07:00:13）：P0 seed14/15/16全部训练完成40000且encoder_unchanged=True；B100 seed14/15训练完成40000；B100 seed16未开始。seed15两组与P0 seed16正在全开发推理，当前仍只有首seed76条完整开发指标。最新状态与run_index已更新；没有手动重复启动任务。
