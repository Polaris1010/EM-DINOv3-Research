# MB-1：Official MitoNet Benchmark Evaluation

EMPIAR-10982全部七组、八个固定run完成：G-P0/G-B100各三个MV3 decoder seeds，官方MitoNet与MV5 MitoNet-MV各一个固定模型。无监督训练、SSL、checkpoint创建/覆盖、阈值或尺度搜索；TTA OFF。

## 七组等权二维语义结果

每个体数据按全部原生Z平面计算二维指标后等层平均；TEM按全部100图平均；随后同模型等seed平均、七benchmark等权。不是全体2595层池化，也不是三seed集成预测。Dice/IoU空GT且空预测为1，undefined P/R留空并单列计数；HD95有漏检可为inf，不隐藏。

| 系统 | Dice | IoU | Precision | Recall | HD95 nm | NSD@16nm | 前景比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| G-P0 | 0.4579 | 0.3448 | 0.5953 | 0.4138 | inf | 0.2157 | 0.0447 |
| G-B100 | 0.5027 | 0.3998 | 0.6722 | 0.4424 | inf | 0.2785 | 0.0436 |
| Official-MitoNet-v1 | 0.7122 | 0.6155 | 0.8465 | 0.6518 | inf | 0.4584 | 0.0537 |
| MitoNet-MV | 0.2409 | 0.1632 | 0.5001 | 0.1852 | inf | 0.0959 | 0.0249 |

**物理指标覆盖限制：** TEM100图无可靠独立物理标定，HD95 nm和NSD@16nm全部留空；因此上述物理宏平均只覆盖六个有可靠XY标定的benchmark，不冒充七组物理指标。未虚构Z spacing或3D nm距离。

## 每benchmark二维Dice（seed均值）

| Benchmark | G-P0 | G-B100 | Official MitoNet | MitoNet-MV | B100−P0 |
|---|---:|---:|---:|---:|---:|
| c_elegans | 0.1869 | 0.1213 | 0.6108 | 0.0455 | -0.0656 |
| fly_brain | 0.5998 | 0.8392 | 0.8991 | 0.4611 | 0.2395 |
| glycolytic_muscle | 0.7741 | 0.5854 | 0.9050 | 0.0151 | -0.1887 |
| hela_cell | 0.3293 | 0.5124 | 0.7337 | 0.2858 | 0.1831 |
| lucchi_pp | 0.6918 | 0.8429 | 0.8903 | 0.4416 | 0.1511 |
| salivary_gland | 0.0151 | 0.0047 | 0.1278 | 0.0284 | -0.0103 |
| TEM-100 | 0.6087 | 0.6126 | 0.8183 | 0.4091 | 0.0040 |

## 全体素volumetric semantic IoU：本轮实测

以下是逐层二维推理后，对完整体数据全体素混淆计数计算的IoU，不是二维IoU的平均，也不等同作者完整3D实例流程。TEM此表另标为全图像像素池化，不用它代替TEM逐图均值。

| Benchmark | G-P0 | G-B100 | Official MitoNet | MitoNet-MV |
|---|---:|---:|---:|---:|
| c_elegans | 0.1062 | 0.0655 | 0.4515 | 0.0285 |
| fly_brain | 0.4475 | 0.7414 | 0.8305 | 0.3192 |
| glycolytic_muscle | 0.6392 | 0.4316 | 0.8305 | 0.0081 |
| hela_cell | 0.2349 | 0.4037 | 0.6257 | 0.2096 |
| lucchi_pp | 0.5364 | 0.7344 | 0.8067 | 0.2864 |
| salivary_gland | 0.0042 | 0.0030 | 0.0597 | 0.0038 |
| TEM-100 (pooled pixels, not volume) | 0.4885 | 0.4942 | 0.7693 | 0.3253 |

## 原论文公开参考，单独报告

[MitoNet原论文Table1](https://pmc.ncbi.nlm.nih.gov/articles/PMC9883049/)的volume语义IoU在3D统计，TEM是100图二维IoU均值。以下是文献数字，**不是MB1实测结果，也不是DINOv3实例指标**：

| 文献Benchmark | 文献语义IoU |
|---|---:|
| C.elegans | 0.604 |
| Fly brain | 0.902 |
| Glycolytic muscle | 0.840 |
| HeLa | 0.791 |
| Lucchi++ | 0.871 |
| Salivary gland | 0.103 |
| TEM-100图均值 | 0.717 |

本轮官方MitoNet采用固定MV4二维Engine2d而非作者的完整体数据推理/3D后处理；即便同用全体素IoU，复现差异不能仅归为模型权重或数据缺陷。未为贴近文献成绩改变参数。

## 最终问题的回答

1. G-B100二维Dice超过官方MitoNet的组：无。各组完整IoU/P/R/距离/前景比例见MB1_per_dataset_metrics.csv与model_means表。
2. B100相对P0七组等权Dice变化 +0.044727；改善组为 fly_brain, hela_cell, lucchi_pp, TEM-100。三个匹配seed的逐组差值保存在MB1_paired_seed_deltas.csv；不把seed数当生物学独立采集数。
3. MitoNet-MV相对官方MitoNet下降组：c_elegans, fly_brain, glycolytic_muscle, hela_cell, lucchi_pp, salivary_gland, TEM-100。这是当前MV5单seed适配模型的固定benchmark表现，不代表MitoNet微调能力上限。
4. 五个Conrad23来源确认B100池来源和抽样图像重叠；Lucchi++source关联、精确区域未确认；TEM未发现名称重叠但不证明无上游暴露。六个候选volume均未直接进入MV3/MV5监督train，细节见MB1_overlap_status.md。
5. B100相对P0的正向宏平均与逐组变化需要分别解释，不能把平均收益称为所有来源一致改善。本轮没有任何benchmark的B100二维Dice超过官方MitoNet，且多个来源仍严重失败，因此不支持已获得广泛通用优势的主张。来源/抽样原图SSL重叠进一步限制了严格未见图像泛化结论。当前结果是固定完整系统的表现，尚未分离自然DINOv3、Primus、多来源监督适配及EM持续预训练的因果贡献。

本轮到此停止。没有自动启动新训练、更多seed、独立诊断或参数搜索。

## 可复现配置和产物

DINOv3直接复用MV3 mv_probe.sliding/build与原版Primus[5,11,17,23]，native-grid、crop256/stride128、tile_batch16、既有归一化和mean-softmax>0.5。两个MitoNet原样复用MV4 Engine2d/Preprocessor，scale1/confidence0.5、NMS3、整图、TTA OFF；逐层Panoptic保留，不读取GT调输入或参数。

GT仅使用原官方实例TIFF>0形成语义真值，完整原始XY/Z，不用GT box，不删空GT图像。212个原始TIFF文件长度与官方HEAD一致，逐文件计算SHA留档；官方未提供用于比对的SHA，不宣称原始文件SHA与官方摘要校验通过。10项既有checkpoint/encoder资产SHA已与实验记录复核。GTsigned实例ID格式被正常读取，未重采样、反相或改变有效区域。

Manifest SHA `8a05d1a5c46933d91e5d20977bcb06c0b9ca6bee10abbcdc17a43b62ac0ea046`；protocol SHA `7869c030d26082f07581bb77bfaccfac0a86ab39445ce11cd520068ea852369e`。模型/run/seed/path/SHA对照在MB1_protocol.json；逐平面结果用短run_id与dataset_id避免重复，均可回查完整manifest。CSV浮点保留7位有效数字，完整未舍入JSON和预测在服务器。

MB1_per_image_metrics.csv：20760行二维指标；MB1_per_image_confusion.csv保留精确整数计数。MB1_per_dataset_metrics.csv为56行run×benchmark；MB1_per_dataset_model_means.csv为28行等seed均值。MB1_TEM_image_distributions.csv保留100图均值/SD/分位/缺失/inf计数，分位数用经验逆CDF、不插值。所有seeds和所有图像都进入统计。

代表图固定选每volume中间Z和TEM按文件名排序0/49/99图，共9张，四模型均在同图同GT上展示；DINOv3显示首seed，其余seed完整计入数字。MB1_figure_manifest.json保存文件名、SHA和选择规则。没有计算DINOv3实例AP/PQ/F1@75。

官方来源：[EMPIAR-10982](https://www.ebi.ac.uk/empiar/EMPIAR-10982/)、[torch-em CEM benchmark接口](https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/electron_microscopy/cem.py)。来源/无标签暴露边界见独立overlap报告。
