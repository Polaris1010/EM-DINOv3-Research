# Coverage-NR进度核查：93.62k，尚无下游结果

用户要求核实100k与开发/MB1结果，并与D2-100k比较，先不启动新训练。本轮仅一次只读SSH现场查询；原SSH已断开，重新连接后读状态，未改服务器文件、服务或训练配置。

香港2026-10-22 15:23:51（UTC07:23:51）：SSL进度93620/100000；服务active、MainPID3930759、退出状态0，GPU1/2/3利用率90%/94%/89%，各约10.8–11.3GiB。当前loss12.8125、grad_norm2.6757有限。已用23.23小时，近期0.696秒/step；剩余6380步约74分钟纯迭代，连同checkpoint等估计75–100分钟。这仅是SSL剩余估算，不包括后续40k head和完整评测。

最后完整恢复checkpoint为step93000，SHA256 0a53675504477000eb81cbf76a15ba977ab058ee89e08ddb11192c79009bd363。100k导出目录尚无资产，head的progress/training_complete/inference_complete不存在；downstream只有锁定输入协议，benchmark无结果，故没有可报告Coverage-NR指标。网站HTTP200，GPU0未用于本轮。

D2-100k首decoder seed20261014匹配参考为开发Dice0.841379、MB1 Dice0.475754。三seed总体参考为0.839914±0.001345与0.463563±0.010832；将来NR首seed结果应先对同seed比较，再列三seed分布作背景，不能把NR单seed当作三seed均值。本次不计算未完成模型差值。

Coverage-NR从P0连续100k LR，与历史D2分段30k+70k LR不同，历史D2只能作为系统参考，差值不能全部归于覆盖率。没有新训练启动、阈值/尺度搜索或新实验。

证据：evidence/Coverage-NR/progress_check_20261022T072351Z.json。run_index更新实际93620及时间；文档顶部更新本轮进度，历史记录保留。交付codex/coverage-nr-p0-training、draft PR11，不合并。另一个窗口SARR工作树不动，本轮独立worktree记录。

下一窗口只核实既有进程与结果。100k未完成，开发与MB1仍待实际产出。

CPU检查实际结果：registry生成59条并check通过，repository validator PASS；51项unittest成功（3项已有POSIX skip）；git diff --check通过。本轮只有进度证据/文档/注册表更新，GPU未用于检查。
