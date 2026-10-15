> 本协议已全量执行完成；最终结果见[MB1_summary.md](MB1_summary.md)，实际终止证据见evidence/MB-1/terminal_snapshot.json。以下为执行配置记录。

# MB-1 固定官方benchmark推理与交接

用户明确授权EMPIAR-10982全部七组：C.elegans、Fly brain、Glycolytic muscle、HeLa、Lucchi++、Salivary gland、TEM-100。复用G-P0/G-B100的MV3各三个40k decoder seeds、MV4官方MitoNet和MV5一个40k适配权重，共八个run，无训练、checkpoint创建或覆盖，无GT调参/尺度搜索。

服务器已有完整官方原始资产：/opt/data/3_117/tanghr/MitoNet/dataset/EMPIAR-10982/data。212/212文件长度与官方HTTPS HEAD一致，另逐文件计算SHA；106个图像条目（六体积、100TEM）共2595平面。10项checkpoint/encoder资产SHA已复算通过，四系统在c_elegans、Lucchi++、salivary和首TEM原生完整XY的单平面smoke通过，不评分选型。新根/opt/data/3_117/tanghr/codex/mb1_official_benchmark_20261015，物理GPU0禁止，只允许1/2/3，网站不修改。

DINOv3直接调用既有MV3的mv_probe.build/sliding，原版Primus[5,11,17,23]、frozen、crop256/stride128、tile_batch16、mean-softmax>0.5、原uint8归一化与native-grid；没有重写滑窗。两个MitoNet原样复用MV4 Engine2d/Preprocessor AST与相同参数，scale1/confidence0.5/TTA OFF，逐层panoptic保留。图像与GT不重采样，完整原始TIFF只读。

主指标按每幅二维图（体数据每Z平面）计算Dice/IoU/P/R、HD95 nm、NSD@16nm、前景比例。GT>0；原生全部像素，包括空GT平面，empty规则固定。物理距离用既有MONAI/MV2接口的二维spacing。六体数据额外保存全体素混淆计数得到volumetric semantic IoU；这是二维逐层推理，不是原论文3D实例流程。每数据集先平均平面/图像，再平均同模型seeds，最后七组等权；不按2595平面整体池化。另保留每seed及每组pooled混淆。TEM100图均无可靠独立物理标定：附带CSV在公开目录及既有目录未找到，nm指标留空；只对有标定的来源计算物理宏平均并披露数量。

重叠仅作有限来源/资产核对：五个Conrad23来源形状相同、首/中/末三平面与B100池对应Zarr逐像素相等；不宣称全体积哈希匹配。Lucchi++相关source形状不同，确切区域仍UNCONFIRMED，不做空间/尺度搜索。候选六volume均未进入MV3/MV5监督train。TEM名称在212-volume registry未发现；不等同于证明没进入其他上游训练池。仍全部七组评测，禁止称严格外部未见图像测试。

最终输出MB1_benchmark_manifest.json、MB1_protocol.json、MB1_per_image_metrics.csv、MB1_per_dataset_metrics.csv、MB1_summary.md、MB1_overlap_status.md及固定代表性对照图。论文Table1指标存独立参考文件，不混入本轮实测。原始图像/预测/权重留服务器；本地收取代码、指标、证据和Markdown，提交推送draft PR后停止。服务与实际状态以run_index、服务器STATUS及新快照为准，不能依据本执行计划声称已完成。
