# 固定读出决策点诊断：本轮仅方案

用户要求复用现有特征/预测，提出区分表征退化与读出校准问题的最小实验；禁止用MB-1 GT调阈值，暂不启动新训练。

读取Pilot A正式报告、run_index、特征/读出及offline实现。发现现有AUROC来自纯前景patch与远端背景，Dice来自全部有效像素，需先统一单位；Pilot A的Primus缓存是二值mask，不能冒充可校准概率。主比较收束为D2/30k、100k第23层，原320随机诊断视野不变，既有原型只作同patch口径辅助。

方案：同native像素AUROC/AP与原始Dice；只在原312个MV3 train视野像素上按来源均衡、保留自然类别比例确定两个全局偏置；固定后一次性在五来源评分。变换置于上采样之后以保持分数排序；不使用MB1/MV-dev标签确定偏置，不启用单来源阈值或温度搜索。原始/校准两组都给30k与100k，不把30k恶化当100k恢复；报告Brier与Recall代价。

负校准结果不能自动证明encoder退化。稳定排序但全局偏置无效仍INCONCLUSIVE；linear/prototype同口径排序均退化才增加表征侧证据，范围限于既有train-only读出。以一次分析为边界，不追加GT驱动调参。

交付docs/Readout_Calibration_Minimal_Design.md。无服务器连接、无GPU任务、无标量参数估计或评分、无新训练，未修改历史结果、模型或实验注册表。Pilot A保持VERIFIED_COMPLETED；本方案PLANNED/DESIGN_ONLY。沿用独立分支codex/feature-mechanism-pilot-a和draft PR13记录后续设计，不自动合并。

文档收尾检查：registry --check一致（60条，未增改实验记录）；validate_repository PASS（2161份文本）；unittest 51项、3项既有平台跳过，结果OK；git diff --check通过。本轮仅文档与交接变更，沿用PR13提交方案，不自动执行。
