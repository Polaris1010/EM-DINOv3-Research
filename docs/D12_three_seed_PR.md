D1/D2的30k与100k对比原先各只有一个decoder seed。本PR复用四个已有encoder和seed20261014，补齐seed20261015/16共8个40k冻结Primus head，在固定156train/38dev及七组MB1诊断上汇总三个seed的完整指标、样本SD和同seed配对差值。

所有训练、推理和评分均已完成：最后8个Wei完成，新增距离评分304/304；合计12个run、456条开发区域和31,140条MB1逐图指标。100k时D2−D1开发Dice为+0.021238±0.001069，MB1为−0.013398±0.008921，三个seed方向一致。D2续训改善Dice/NSD，但开发HD95三个seed均恶化；肌肉退化也三个seed复现。D1续训MB1平均略升，修正先前单seed解读。

继承MV-Generalist-v2、原版Primus、冻结ViT-L/16、固定40k选模、TTA OFF和既有预处理。MB1物理指标只涵盖6个可靠spacing组，保留空前景导致的HD95∞与TEM100物理NA。三个decoder seeds衡量固定划分下的随机性；MB1仍作为已暴露开发诊断。

提交包含训练/推理及exact GPU评分代码、所有小型指标表、checkpoint路径/SHA索引、完整报告、独立CPU重算脚本、服务器停止证据、交接与每轮Markdown。Wei采用主存保存距离场、连续块传输及完整空间轴CUDA EDT；完整距离场与MONAI一致性验证通过，切换前既有指标SHA保持不变。最终服务inactive/MainPID0/退出0，GPU0未用、网站200；独立已授权Coverage-NR沿既有任务继续。

验证：独立重算456条区域/31,140条逐图指标并逐项匹配自动汇总；registry生成与check、repository validator、CPU unittest和git diff --check。完整结果见docs/D12_three_seed_report.md及docs/work_logs/20261022_33_d12_three_seed_final_closeout.md。本轮结束，不自动启动新实验；PR保持draft，不自动合并。
