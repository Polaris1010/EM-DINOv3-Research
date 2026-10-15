在MitoNet作者公开的EMPIAR-10982全部七组benchmark上，固定评测已有G-P0/G-B100各三个decoder seeds、官方MitoNet和MV5 MitoNet-MV，共八个run。仅推理、评分和结果分析；不训练、不创建/覆盖checkpoint、不调整阈值/尺度/TTA。

官方212个原始TIFF已在服务器找到，全部文件长度/字节SHA和10项模型资产SHA验证；106个图像条目共2595原生平面，预期20760条逐图记录。DINOv3原样调用MV3 sliding256/stride128；两个MitoNet使用同MV4 Engine2d/native-scale1/confidence0.5。保留原始Panoptic输出，DINOv3不计算实例AP/PQ/F1。

主表按平面/图像、seed、七组benchmark依次等权；另保留六体数据全体素IoU与TEM100图分布。无可靠TEM物理标定，不伪造nm指标。文献Table1成绩与本轮实测分开。五个Conrad23来源已在B100 SSL池且三抽样原图平面相同；Lucchi++来源相关但确切区域未确认，TEM未发现名称匹配不等同无暴露，因此不是严格未见图像测试。

最终产物为MB1_benchmark_manifest.json、MB1_protocol.json、逐图/逐组指标、MB1_summary.md、MB1_overlap_status.md及固定代表性对照图。当前执行事实见experiments/MB-1/run_index.json；未完成时不声称总分。四系统无GT原生XY smoke、registry、repository与31项CPU检查已通过，旧41条registry不变。本轮Markdown和下一窗口交接持续随代码提交推送，完成后停止。

此draft以未合并的codex/mv5-mitonet-adaptation为base，保留MV3/MV4/MV5历史，不自动合并。
