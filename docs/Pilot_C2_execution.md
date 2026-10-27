# Pilot C-2：固定45k正式配对

2026-10-28用户明确批准两组从D2/30k完整状态各续训15000步。唯一差异是C1加入已接受的Selective One-Sided Gram Barrier；lambda1、margin0.05、bottom50%、Chebyshev>2及layer23锁定。关闭C-1完整梯度诊断，只保留正常单次backward的参与标志和数值健康记录。

当前为启动准备，不能据此声称训练已经运行。正式状态须由新目录的STATUS、实际步数、源码commit和checkpoint哈希确认。已有C-1/Pilot B及其他PR不修改。

两SSL完成后，各自运行原版冻结Primus40k/seed20261014、Linear23 500步、全部38开发区域和固定MB1七组。原下游算法仅调整实验输出目录导入，核心Barrier与C-1字节一致。源代码与资源调度54项CPU测试已通过（3项既有Windows平台skip）；生产模式GPU集成测试由有界执行器在正式更新前运行。

最终比较D2/30k、D2/100k、Trajectory-Lerp05、C0、C1；只根据固定结果给出净收益和研究决策。Barrier激活不代表遗忘；45k同预算收益不自动证明阻止100k长期退化。完成报告、指标、代码和Markdown提交到独立draft PR后停止，不自动延长或添加方法。
