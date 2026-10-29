固定C0/C1各自45k完整checkpoint续训55k至100k，C1保持原30k anchor及λ1/margin0.05/bottom50%/Chebyshev>2/Layer23。165000组rank-step输入、RNG和历史LR/WD一致，无重放；Barrier每次参与同次backward。两模型均完成冻结原版Primus40k/seed20261014、Linear500、38开发区域及七组MB-1。

C1−C0：开发Dice−0.001007，MB-1 Dice+0.049892；肌肉Dice+0.298020、AP+0.298814、Recall+0.061474，远端FP6.4958%→0.4171%。相对Pilot B中点开发+0.002127、MB-1+0.028946。OO/Turner/C.elegans仍下降；肌肉贡献MB-1增益85.3%。100k结果支持保留固定Barrier候选主线，修正45k阶段的过早淘汰建议。

含真实源码/锁定协议、七系统完整指标、checkpoint SHA、终点图、最终报告与本轮Markdown交接。418份小型输出SHA验证、六历史权重重新核验、均值独立重算和67条历史registry保持。单decoder seed、MB-1为已暴露诊断；masked Student与unmasked Anchor限制机制解释。

流水线26.12h，服务inactive/退出0，网站200、GPU0未用；全部预算完成后停止。独立draft，base为codex/pilot-c-100k-recommendation，不修改已有PR、不自动合并或新训练。CPU检查见evidence/Pilot-C100/cpu_checks.json。
