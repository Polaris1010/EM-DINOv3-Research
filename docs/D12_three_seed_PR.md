D1/D2的30k与100k比较此前各只有一个decoder seed，无法判断head随机性是否影响研究决策。本轮复用四个现有encoder与seed20261014结果，各补两个匹配seed的固定40k冻结Primus，在相同38区域和七组MB1诊断上输出三seed均值、样本标准差及同seed配对差值。

原监督划分、模型、训练预算、推理和历史结果保留。新增有界并行服务在GPU1/2/3剩余显存运行，并临时接管Coverage-NR运行监督，使其SSL ranks继续训练。完成固定范围后停止，无新SSL训练。

验证：CPU unittest 55项通过（3项已有POSIX skip）；实际运行状态和进度见本轮run_index、execution文档与工作记录。
