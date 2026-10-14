# 继续完成MV3与三seed最终交付

用户要求按原方案跑完剩余seeds及开发评测，汇总三seed，输出MV3_final_report.md，指标/代码/本轮Markdown一并commit/push；完成后停止，不自动启动新实验。

本轮继承MV-Generalist-v2：仅Han整幅GT、不用annotation mask；其余来源、划分、初始化、采样、冻结encoder、原版Primus、三个seeds与共同40k预算保持。沿用现有server service，不重复启动。

开始时首seed两组40k及76条全开发指标已完成，P0另外两seed与B100第二seed40k完成、开发推理中；B100第三seed等待原调度。最新实时状态仍为additional_matched_seeds_train_and_infer，网站200，GPU0禁用。

新增verify_mv3_results.py仅核验已收取结果的覆盖与算术、checkpoint元数据关联和匹配初始化/采样；不读原图、不加载权重、不运行实验。首seed76条记录、6份checkpoint sidecar、12来源均值核验通过。最终追加三seed逐seedDice、来源配对差值及decoder-seed SD，不能把decoder seeds当作独立acquisition。

实际最终状态、228条指标/18份checkpoint回执、五项研究回答、提交推送及停止状态将在完成后追加。当前不把剩余队列或最终报告写成已经完成。

## GPU空闲与评分调度修复

用户指出GPU空闲及内存约20GiB。16:50:17香港时间只读核查：六组40k及全开发推理已完成，149条指标已写出；评分worker PID3742491 CPU132%、RSS约19GiB，GPU3仅386MiB上下文，正在Müller high_c1的CPU表面预处理。当前无模型训练，GPU EDT是短阶段，显存用量不能当作训练是否继续的证据。此前串行新进程评分加15秒轮询存在执行等待，前一条已明确核实，不以GPU占满作为目标。

仅修正现有评分调度：最多三个worker，GPU1/2/3各一个；保守预留每任务45×valid_voxels+4GiB，并保留16GiB全局余量；1秒轮询。evaluator、训练、loader、数据协议、模型与已保存结果不改。新恢复路径必须存在全部六组training_complete和inference_complete，先核验匹配，再仅运行缺失评分与最终报告，避免重训或重复推理。

新增4项模拟调度测试证明每卡至多一个、最大三进程、内存上限生效及失败停止排队；总17项CPU测试通过。实际部署及并发进程回执将在运行后追加。

评分执行修复commit `f410b772f15e8fc29e3136bb6ba7c377d23beb35`已推送并部署。恢复前确认六组40k与全开发推理完成，保留18个checkpoint及189个已完成指标；仅更改run.py和新增scoring_queue.py，其余六份训练/loader/evaluator/report/manifest/protocol SHA不变。新service MainPID3748795 active。17:04:57香港时间回执：三个评分worker实际绑定物理GPU1/2/3，指标222/228；189份已有指标SHA和18个checkpoint文件size/mtime均一致。没有重新训练或推理；最后缺失指标继续执行。

## 实际最终完成与停止

服务器17:10:18香港时间：六组各40k、228条指标、18个checkpoint及最终报告全部完成；service inactive/MainPID0/Resultsuccess，网站200、GPU0未用、无后续实验。被动collector已完成收取并推送原始最终结果commit `6cb5983f31cb8c7a7b6a073df6a26f5d7964c402`，没有重新启动任务。

228条记录覆盖和混淆矩阵算术、逐来源/三seed平均、18份checkpoint sidecar关联、三个seed初始化和sampling/source_draws匹配核验PASS。追加逐seed汇总与逐来源配对SD，P0 0.813741±0.000827、B100 0.826449±0.000832、Δ+0.012707±0.001361，三个seed方向均正。

五项回答：同区域描述性扩展收益P0+0.4670、B100+0.4359，不能单独归因增加来源；B100仍有平均多来源收益，Haberl基本持平；Guay和Müller没有改善，且新协议B100仍低于P0；主要短板Haberl/Han/Guay/OpenOrganelle；当前优先监督适配，输入尺度未被操纵，下一步决策是官方MitoNet同图像GT对照。本轮停止，不自动执行。

初次回执推送遇TLS unexpected EOF，使用同一授权private分支重试成功；后续原始终端结果也已推送。当前最终增强报告、指标核验、代码和全部Markdown交接将一并完成最后提交。

最终报告补充的逐seed/配对SD与核验文件已同步回服务器本轮目录，原始自动报告另存evidence/MV3_final_report_server_original.md；本地与服务器SHA一致。同步仅写报告/小表/回执，服务仍inactive，没有启动科学任务。
