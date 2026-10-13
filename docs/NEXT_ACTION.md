# 下一项已确定的行动

**初始化PR完成并确认后，收取已在运行的MV-2无TTA扩展结果，回答B100k对P0的收益是否跨更多来源成立。** 本轮不新增科学实验，也不重新启动MV-2。

现有服务器任务：`/opt/data/3_117/tanghr/codex/mv_dev_v2_notta_20261013`；service为 `mv-dev-v2-notta-20261013.service`。2026-10-13本轮快照为运行中，最终状态仍待读取。旧 `mv_dev_v2_expansion_20261013` 四翻转结果已废弃，保留历史，不参与正式汇总。

输入：`experiments/registry.yaml`登记的MV-1九个head（20000 steps）、实际MV-2 manifest188个volume/ROI与reserved test排除项、原版Primus、固定encoder、原normalization与原sliding函数。TTA=OFF，不训练，不增加模块，不改预处理或checkpoint选择。

验收：

1. 只读确认现有job最终状态；任务未完时保持VERIFIED_RUNNING，不把partial输出写成完整结果，也不默认启动失败重试。
2. 保存新的带时间戳status/result/manifest/hash快照，证明九个head和历史输入未变化。核对3×3模型/seed组及每组188个条目的完成、复用、跳过/失败情况；不静默缩小队列。
3. 每来源汇总Dice/Precision/Recall/HD95 nm/NSD@16nm，分开官方validation与自定义dev、每seed波动与独立volume/acquisition数量。unknown acquisition明确标UNVERIFIED。
4. 报告B100k−P0的逐volume和来源差异，B150k仅作已有参考。列出缺少官方native mapping或不可用区域，不把锁定test加入。结论限定transductive development。
5. 第二PR只做上述结果收取、覆盖核对和科学解释。通过已有CPU registry checks，不改模型。随后给出继续/转向的单项科学决策。

当前GitHub授权问题只阻塞首PR的远程交付，不需要重训或重新盘点已核验资产。未来严格B_clean与acquisition审计不阻塞这项开发问题。
