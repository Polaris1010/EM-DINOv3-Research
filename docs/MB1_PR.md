EMPIAR-10982全部七组官方benchmark的固定推理与评分已完成：G-P0/G-B100各3个MV3 decoder seeds，官方MitoNet与MV5 40k MitoNet-MV各1个模型，共8run、20760条二维指标。G-P0 Dice 0.457938；G-B100 Dice 0.502665；Official-MitoNet-v1 Dice 0.712154；MitoNet-MV Dice 0.240936；B100−P0 +0.044727。

复用已有checkpoint及MV3/MV4原推理路径，无训练、checkpoint写入、阈值/尺度/TTA搜索。提供逐图/逐组/配对seed指标、六volume全体素IoU、TEM100分布、固定9图的manifest、原资产/源码/模型SHA及本轮工作记录和下一窗口交接。论文3D参考成绩与本轮2D实测独立呈现。

五组有B100 SSL来源与抽样图像重叠，Lucchi++来源关联但精确区域未确认，TEM未发现名称匹配不证明无暴露；不能称严格未见图像测试。TEM缺可靠物理标定，nm指标留空。独立核对全部精确计数与七组等权平均，历史输入前后不变，网站200，GPU0未用，服务已停止。

验证：scripts/verify_mb1_results.py、registry --check、validate_repository.py、31项CPU测试及git diff --check全部通过；独立复核8run使用相同GT像素、全部精确混淆计数和七组等权汇总，下载文件SHA与服务器终止快照一致。此draft以未合并codex/mv5-mitonet-adaptation为base，不自动合并。
