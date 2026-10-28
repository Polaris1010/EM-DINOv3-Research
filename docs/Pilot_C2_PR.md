固定D2/30k完整状态各续训15k至45k，仅C1加入锁定λ1/m0.05/bottom50%/Chebyshev>2/layer23 Barrier。45,000组rank-step输入、RNG与LR/WD完全配对，生产模式Barrier在同次backward参与全部更新；无额外诊断forward/VJP。两组各完成原版冻结Primus40k/seed20261014、Linear23及38dev/七组MB1。

C1−C0：开发Dice−0.003790、NSD−0.003432，MB1 Dice+0.003175；肌肉Dice−0.075061、AP−0.051623、Recall−0.090534。远端FP降低伴随Recall损失。相对Pilot B中点开发/MB1均下降，当前配置不继续作为MICCAI主线。

包含独立代码、完整小型指标、SHA及配对/停止证据、最终报告与本轮Markdown交接。337份收集文件SHA核验，来源/组均值独立重算一致。C0/45k尚未复现100k严重肌肉Dice/FP退化；masked/DropPath student与unmasked eval anchor的差异限制机制解释。一个decoder seed、MB1为已暴露诊断，不宣称独立泛化或已阻止长期遗忘。

服务inactive/退出0，网站200、GPU0未用，历史资产及已有PR保持。独立draft，基于C-1分支，不自动合并或继续训练。CPU检查见evidence/Pilot-C2/cpu_checks.json。
