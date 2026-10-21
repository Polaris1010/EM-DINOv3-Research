# D1/D2 三seed执行记录

用户授权补齐四个encoder的固定划分三decoder-seed重复。读取真实checkpoint登记和seed20261014完成记录，当前每组确实只有单seed。新增两个seed各40k，共8次head，旧seed复用。独立根、原版trainer/evaluator、40%进程显存限制与Coverage-NR监督接管方案已编码。CPU55项通过（3项已有skip）；正式启动及实测进度在本记录后追加。
