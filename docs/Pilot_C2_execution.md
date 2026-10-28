> 已完成并停止：两15k SSL、匹配40k heads/Linear及全量评价齐备。最终结果与决策见[Pilot_C_Report.md](Pilot_C_Report.md)。以下保留本轮按时间记录。

# Pilot C-2：固定45k正式配对

2026-10-28用户明确批准两组从D2/30k完整状态各续训15000步。唯一差异是C1加入已接受的Selective One-Sided Gram Barrier；lambda1、margin0.05、bottom50%、Chebyshev>2及layer23锁定。关闭C-1完整梯度诊断，只保留正常单次backward的参与标志和数值健康记录。

启动前记录：当时仅为准备，不能据此声称训练已经运行。正式状态须由新目录的STATUS、实际步数、源码commit和checkpoint哈希确认。已有C-1/Pilot B及其他PR不修改。

两SSL完成后，各自运行原版冻结Primus40k/seed20261014、Linear23 500步、全部38开发区域和固定MB1七组。原下游算法仅调整实验输出目录导入，核心Barrier与C-1字节一致。源代码与资源调度54项CPU测试已通过（3项既有Windows平台skip）；生产模式GPU集成测试由有界执行器在正式更新前运行。

最终比较D2/30k、D2/100k、Trajectory-Lerp05、C0、C1；只根据固定结果给出净收益和研究决策。Barrier激活不代表遗忘；45k同预算收益不自动证明阻止100k长期退化。完成报告、指标、代码和Markdown提交到独立draft PR后停止，不自动延长或添加方法。

实际启动核实：2026-10-28 00:42:07 HK，生产GPU集成测试通过，C0正式到30160，本轮160/15000，loss有限、三卡执行、网站200。源码commit `b3c101680eaa451b74963d66433ba70bd6077adc`；C1及下游尚未完成，指标和最终PR均不作完成声明。


2026-10-28 03:57 HK 训练里程碑：C0全部15,000次新更新已完成（global45000），实际11697.10s。EMA SHA256 `54a20de4c373bec12af858c810b4edf7eb337aaff62dd3631ff3a00b8a69e021`。C1已从原30k完整状态恢复，初始student、teacher和optimizer step30000严格加载，rank0 RNG恢复hash与C0相同。生产模式前20步全部三rank输入/RNG/采样链与C-1一致、Barrier upstream=1且同一次backward，未启用额外VJP/Student诊断forward；与C-1诊断运行相比，loss最大绝对差5.6267e-5，不能写成数值逐位相等。方法参数和预算均未改变；下游尚未开始，暂无C2开发/MB1指标。


2026-10-28 07:30 HK：C0和C1均完成原定15,000次新SSL更新，global45000。实际训练耗时C0=11697.1s，C1=12582.63s。45,000组rank-step全部输入、RNG、采样链、LR/WD相等，无重放；C1每步单次Student forward且Barrier upstream=1，无额外VJP/诊断forward，anchor无梯度。两个匹配Linear23均500步完成，两Primus各6800/40000（同step采样hash相等）。开发/MB1尚未开始，未形成最终科学结论。


最终完成：C0/C1均完成global45000与各15k新更新，匹配Primus40k/seed20261014、Linear500、38dev和七组MB1全量完成。337份小型元数据/指标SHA核对；均值独立重算一致。C1相对C0开发Dice−0.003790、NSD−0.003432、MB1+0.003175；肌肉Dice−0.075061/AP−0.051623/Recall−0.090534，远端FP降低伴随召回损失。相对Pilot B开发−0.008421、MB1−0.018905。Turner退化0.064686为开发均值的主要负项，Han改善0.012336；MB1六组改善、一组肌肉明显下降。当前固定Barrier配置不继续作为MICCAI主线；45k结果不证明阻止100k遗忘。服务inactive/退出0，网站200，历史原30k full/EMA SHA不变；无后续训练。完整报告[Pilot_C_Report.md](Pilot_C_Report.md)，所有代码/指标/交接记录待独立draft PR交付后停止。
