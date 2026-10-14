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
