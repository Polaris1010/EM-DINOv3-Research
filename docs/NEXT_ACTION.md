# 下一项已确定的行动：完成已授权MV-3

2026-10-14用户已明确改为MV-3多来源监督训练，以下旧MV2结果收取计划被取代。不要按旧段落增加验证、推理、Linear Probe、Gram/HRA或新审计。

当前分支 `codex/mv3-generalist-training`，服务器 `/opt/data/3_117/tanghr/codex/mv_generalist_v1_20261014`，service `mv3-generalist-20261014.service`。先读取现有STATUS、progress和最新evidence，禁止重复启动。只继续G-P0/G-B100、3匹配decoder seeds、各40000 optimizer steps的共同锁定协议。原版Primus，[5,11,17,23]，encoder冻结，TTA=OFF，GPU0禁用。

两组首seed smoke已PASS，真实12来源loader、ignore loss/梯度、encoder不变、checkpoint往返与推理通过；decoder初始化及采样SHA相同。首seed20261014已实际开始正式训练。2×40k完成后做38区域/12来源全开发验证及既有物理指标，再补20261015/20261016；有界执行器仅执行这6个run。

MV-Generalist-v1：156 train/38 dev，12来源；部分探索性dev已转train，不再作为新验证；official test/已映射validation保留。Han非零mask有效；有效区与每来源体积清单见 `experiments/MV-3/MV3_source_coverage.csv`。193个区域复用既有缓存只读，1个区域补新缓存。不能把旧4-volume约束恢复成前置门槛。

结束时收取MV3_final_report.md、18个milestone checkpoint SHA、逐区域/来源指标与配对变化，回答监督扩展收益、B100多来源优势、Guay/Müller变化、主要短板和优先改进方向。比较旧预测仅作同新heldout区域的描述性参照，不重新运行旧9个head，不改历史记录。保持共同40k预算，不为某个encoder单独追加训练。

每轮将代码、状态和Markdown commit/push，实际报告完成前状态保持VERIFIED_RUNNING/PLANNED。最终报告后停止，不自动启动官方MitoNet或新预训练。详细协议见MV3_PROTOCOL，交接见NEW_WINDOW_HANDOFF。

## 以下保留为上一轮历史计划

**初始化PR已合并并交付main；下一项科学行动是收取已在运行的MV-2无TTA扩展结果，回答B100k对P0的收益是否跨更多来源成立。** 本轮仅完成代码和记录交付，不新增科学实验，也不重新启动MV-2。

现有服务器任务：`/opt/data/3_117/tanghr/codex/mv_dev_v2_notta_20261013`；service为 `mv-dev-v2-notta-20261013.service`。2026-10-13本轮快照为运行中，最终状态仍待读取。旧 `mv_dev_v2_expansion_20261013` 四翻转结果已废弃，保留历史，不参与正式汇总。

输入：`experiments/registry.yaml`登记的MV-1九个head（20000 steps）、实际MV-2 manifest188个volume/ROI与reserved test排除项、原版Primus、固定encoder、原normalization与原sliding函数。TTA=OFF，不训练，不增加模块，不改预处理或checkpoint选择。

验收：

1. 只读确认现有job最终状态；任务未完时保持VERIFIED_RUNNING，不把partial输出写成完整结果，也不默认启动失败重试。
2. 保存新的带时间戳status/result/manifest/hash快照，证明九个head和历史输入未变化。核对3×3模型/seed组及每组188个条目的完成、复用、跳过/失败情况；不静默缩小队列。
3. 每来源汇总Dice/Precision/Recall/HD95 nm/NSD@16nm，分开官方validation与自定义dev、每seed波动与独立volume/acquisition数量。unknown acquisition明确标UNVERIFIED。
4. 报告B100k−P0的逐volume和来源差异，B150k仅作已有参考。列出缺少官方native mapping或不可用区域，不把锁定test加入。结论限定transductive development。
5. 第二PR只做上述结果收取、覆盖核对和科学解释。通过已有CPU registry checks，不改模型。随后给出继续/转向的单项科学决策。

GitHub授权已通过用户确认的浏览器流程恢复，private remote已建立，首PR状态见 `docs/GITHUB.md`。未来严格B_clean与acquisition审计不阻塞这项开发问题。
