# Coverage-NR完成、同seed对照与停止

用户询问当前进度，沿用之前授权的100k/40k/开发及MB1固定任务。本轮只读服务器状态、收取已完成结果并核对表格；没有启动训练、重新推理或距离评分。

## 实际完成

香港2026-10-22 23:11（UTC15:11:08）现场查询确认：Coverage-NR SSL100000、seed20261014原版Primus40000、38开发区域/12来源、2595张MB1/7组全部完成。最终服务coverage-nr-p0-gpu-resume-20261022.service inactive、MainPID0、退出0；网站HTTP200，GPU0未用，历史输入保护检查通过。本轮结束停止。

保留CEM1425433个图像ID全部消费，覆盖100%；CEM2398767次、MitoVerse2401233次抽样。历史D2唯一访问955756（67.05%）。这是已完成100k消费后的覆盖结果。

## 结果与解释

| Dice | Coverage-NR seed20261014 | D2-100k同seed | NR−D2 |
|---|---:|---:|---:|
| 开发12来源等权 | 0.831821 | 0.841379 | −0.009558 |
| MB1七组等权 | 0.440027 | 0.475754 | −0.035726 |

NR与D2使用相同head初始化SHA、sampling hash和GT计数。开发OpenOrganelle Dice下降0.111681，是主要总体差距来源；MB1肌肉改善0.215677→0.393368，其余六组均下降。开发HD95改善534.23→505.28nm，但NSD略降；MB1 Precision更高、Recall/前景比例更低，完整指标见报告及逐组差值表。

本轮没有得到总体Dice收益，不据此继续自动增加覆盖率实验。NR从P0连续100k，而D2是分段30k+70k LR且collection分布不同，差值不能仅归于覆盖率。NR仅首decoder seed；D2三seed均值0.839914/0.463563作为背景，主对照仍按同seed进行。MB1为已暴露开发诊断。

## GPU执行与时间

38条开发指标均为exact GPU EDT，Wei使用完整轴主存GPU分块；MB1记录4818次GPU EDT调用。有/无ignore的数值等价验证已在上一轮通过。CPU在本轮只进行小表格聚合和仓库检查。

上一轮GPU合成校验误触未知PID保护的错误和恢复记录保留：94450停止，从94000 full状态恢复，450步重放，模型最终仍100k。中断前84238.7秒+接续4437.1秒，记录段合计约24.63小时（包括重放，恢复加载等另计）；head49.61分钟、开发推理32.91分钟，评分/MB1另计。

## 资产和交付

完整中文报告`docs/Coverage_NR_final_report.md`；服务器原报告保留在`experiments/Coverage-NR/Coverage_NR_final_report.md`。该目录包含原始38区域、12来源、2595逐图、7组结果与MB1 JSON，checkpoint路径/SHA以及D2同seed引用表、完整差值表、失败计数。

`scripts/finalize_coverage_nr_report.py`从现有TP/FP/FN复核Dice/IoU，逐区域/图像核对GT计数，重建开发来源等权和MB1组等权聚合，验证匹配初始化/监督采样及GPU记录。PASS收据在`evidence/Coverage-NR/final_report_validation.json`。停止/网站/权重索引证据为`final_status_20261022T151108Z.json`。

run_index、registry和新窗口入口同步VERIFIED_COMPLETED。私人仓库codex/coverage-nr-p0-training、draft PR11；代码、指标、完整报告与本轮Markdown一并提交推送，不自动合并。实际最终commit以Git与PR head为准。

下一窗口先读最终报告；不恢复已停止服务，不自动追加seed、D3、长程训练、阈值或尺度搜索。

## 实际仓库检查

独立最终表格重算PASS；registry生成59条并check一致；repository validator PASS；51项unittest成功（3项已有POSIX skip）；git diff --check通过。全部原指标和D2引用表保持，未触发新GPU作业。
