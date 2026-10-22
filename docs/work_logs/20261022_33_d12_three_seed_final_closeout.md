# D1/D2三decoder seed完整评分与交付收尾

用户要求核实最后8个Wei，完成后汇总三个seed完整指标与结论，把代码、报告和本轮Markdown提交推送后停止。本轮复用已完成预测，只读取服务器终态、收取指标并独立CPU重算；没有训练、重新推理或参数搜索。

## 实际完成与证据

- 新增8个head均40k，开发预测与MB1均完成；最后8个Wei的HD95/NSD全部齐全，新增距离评分304/304。
- 合并历史seed20261014与新增20261015/16：12个run、456开发区域、144来源×seed、84 benchmark×seed、31,140条逐图指标（12份无损CSV分片）。
- UTC 2026-10-22T03:00:10服务器快照：最终评分服务inactive、MainPID0、退出0，任务无残留进程；网站200、GPU0未用于本轮。旧服务非零退出属于运维切换历史，由成功接续服务取代。
- `evidence/D12-three-seeds/final_verification_20261022.json`包含8个Wei的完整结果/评分GPU/源码SHA、277与296条既有指标文件SHA不变、停止状态及NR监督释放证据。
- 独立已授权Coverage-NR沿原任务继续，快照step72540；本轮没有重复启动或增加实验。

## 完整指标与研究结论

完整报告：`docs/D12_three_seed_report.md`。保留自动生成原报告及原始指标，新增IoU、空前景/缺spacing计数、逐来源配对和实际耗时表。

| encoder | MV3 Dice均值±样本SD | MB1 Dice均值±样本SD |
|---|---:|---:|
| D1-30k | 0.817498±0.001640 | 0.472930±0.013660 |
| D2-30k | 0.829209±0.000874 | 0.446823±0.015937 |
| D1-100k | 0.818676±0.000324 | 0.476961±0.002890 |
| D2-100k | 0.839914±0.001345 | 0.463563±0.010832 |

100k时D2−D1开发Dice+0.021238、MB1 Dice−0.013398，三个seed方向一致。D2从30k到100k开发Dice+0.010705、MB1 Dice+0.016740也三个seed一致，但开发HD95+48.73nm且三个seed均恶化。肌肉D2 Dice均值0.443746→0.196840，三个seed均下降。D1续训MB1平均略升+0.004030，仅2/3 seed正向，修正此前单seed退化判断。

MB1总体HD95因空GT或空预测的单边前景失败为∞，保留失败；TEM100物理spacing缺失保持NA。D2的MB1 NSD/Recall更高，但Precision更低、前景更多，避免把Dice差异写成所有指标一致变化。

本轮衡量固定划分上的decoder随机性，SSL未分别重复预训练，独立acquisition数未增加；MB1是已暴露开发诊断。

## 代码、验证和Git交付

新增`python scripts/finalize_d12_seed_report.py`：从原始TP/FP/FN重算Dice/IoU，检查12个run相同GT集合、初始化及采样hash、checkpoint索引，重建两级等权均值/样本SD并逐项对照自动汇总。输出`evidence/D12-three-seeds/final_report_validation.json`。这是CPU聚合，不读取图像或checkpoint。

交付沿用`codex/d12-three-seed-validation`、私人仓库draft PR12，不自动合并。自动收取已推送c2555f1；本轮补充完整叙述报告、独立复核脚本/证据、派生CSV、最终run_index/registry、交接及本记录。最终提交以Git日志和PR head为准。

适用CPU检查：独立重算脚本、registry生成与check、repository validator、unittest、git diff --check。实际执行结果见本记录后续验证段。保留其他窗口的未跟踪文件，不纳入本轮提交。

## 下一窗口

此三seed任务已完成，停止。入口为完整报告、final_verification与run_index；不恢复旧评分服务，不自动D3或新增诊断/训练。Coverage-NR按其独立既有授权继续。

## 实际验证结果

- 独立聚合复核PASS：456区域、31,140逐图、84逐组、12checkpoint行，2595图像/38区域GT集合一致。
- registry生成67条，check匹配证据；与cb5c871逐项比较，前59条历史记录全部不变。
- repository validator PASS：115 provenance、664 historical文件，未使用GPU。
- unittest共57项，成功，3项既有POSIX环境skip。
- PR正文的Windows换行已统一LF；git diff --check最终通过。
