# MV-Dev-v1 split/exposure audit

- 数据发布固定 revision d0af9a6fd11faaf16ce0822233b7bbf72553ce7d；代码 revision df884f6bf353b3cd6b4b6a5ee3620daabd8a79be。
- Guay vol0=train / vol1=dev / vol2=reserved test；Jiang vol1=train / vol2=dev / vol0=reserved test，沿用官方 overlay。
- BetaSeg high_c1/low_c1=train；high_c3/low_c2=dev，全部来自官方 TRAIN 集；high_c2/high_c4/low_c3仍为 reserved test。
- 8 stores 与 source image-file identity 在监督 train/dev 间不重复，坐标是完整已存 ZYX volume；未推断 parent crop offset。
- 未取得物理 acquisition 独立性的证据，acquisition_id保留null；禁止 unseen acquisition 结论。已知重叠 crop / split 不完整来源未纳入。
- B100/B150 全部8volumes在正式SSL全量manifest中：ssl_exposed=true仅表示采样池暴露，不声称每像素都被抽到。P0在本项目未进行 EM SSL；不能证实其自然预训练集合不存在 EM。
- 历史424样本 GT crop diagnostic覆盖全部212volume，开发曝光记录不能清零；reserved test不意味着原来untouched。
- 本次训练/预算/选模/误差图未读reserved test labels，全部36评测行的supervised_role=dev。
- 官方 split overlays、SSL encoder checkpoints、历史 FAIL/结果及网站 Primus 源码哈希保护；本批仅新建cache、head与输出。
- 严格 B_clean 的 Stage0 NO_GO 保留；本开发基准不解锁严格训练/最终测试。
- 官方Primus代码孤立导入，未修改函数或共用网站模块；无FFT、频域gate、LAST。
- 协议单seed通过训练及完整评测后才扩到3seeds；每个seed同head初始化与同采样序列，所有模型固定同steps。
- 体积SD（ddof1）与seedSD分开报告；不足样本不填0，空预测保留Dice0、NSD0、HD95inf并显式标记。
