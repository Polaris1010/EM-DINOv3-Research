# MV-4B：Official Semantic vs Panoptic输出复核

本轮决策：**四区域的输出路径差异不明显，不扩展38区域。** 原始semantic probability在官方0.5阈值后的掩码，与既有Panoptic二值输出在四个完整区域中逐像素完全相等（XOR总计0）。Guay及Müller high_c1在Semantic Only下没有恢复；所以在这些固定输入/参数上，“Panoptic中心/分组丢弃语义前景导致接近零Dice”的解释不成立。

官方Semantic Only和raw threshold仅在Müller high_c1有极小差别：stuff_area64规则去掉329个GT背景像素，TP保持2，无语义GT前景丢失，Dice变化约4.53e-12。这个区域的Dice约2.35e-7，不能与历史Müller两个区域来源均值0.000168混淆。Casser/Wei两条输出完全一致。

此复核只覆盖4个指定区域；**没有获得38区域Semantic Only总成绩**，不能将历史Panoptic0.610209重新命名为完整Semantic Only结果，也不能用四来源平均0.472604替代12来源成绩。监督数据/SSL暴露/输入分布/2D尺度与独立测试限制仍然存在；此结果排除了一个具体解释，并没有证明“DINOv3预训练全面优于MitoNet”。

同官方MitoNet_v1权重、同uint8图像/GT/有效区、norms0.57571/0.12765、scale1、confidence0.5、tile0、TTA OFF。只改变官方semantic_only选项；原始概率在同一次官方forward捕获，无GT阈值/尺度调参。MV3/MV4结果与预测全部只读保留。

实际范围：4开发区域/4来源。按用户条件先完成四区域；未将这四来源均值冒充12来源结果。

## 四区域直接指标（固定0.5阈值）

| 区域 | 路径 | Dice | Precision | Recall | 前景比例 |
|---|---|---:|---:|---:|---:|
| casser20_kasthuri++_train__dev | raw_threshold | 0.940289 | 0.963017 | 0.918608 | 0.035219 |
| casser20_kasthuri++_train__dev | semantic | 0.940289 | 0.963017 | 0.918608 | 0.035219 |
| casser20_kasthuri++_train__dev | panoptic | 0.940289 | 0.963017 | 0.918608 | 0.035219 |
| guay21_vol1 | raw_threshold | 0.014253 | 0.038330 | 0.008754 | 0.006853 |
| guay21_vol1 | semantic | 0.014253 | 0.038330 | 0.008754 | 0.006853 |
| guay21_vol1 | panoptic | 0.014253 | 0.038330 | 0.008754 | 0.006853 |
| muller21_high_c1 | raw_threshold | 2.34698e-07 | 1.10094e-06 | 1.3135e-07 | 0.003117 |
| muller21_high_c1 | semantic | 2.34703e-07 | 1.10114e-06 | 1.3135e-07 | 0.003117 |
| muller21_high_c1 | panoptic | 2.34698e-07 | 1.10094e-06 | 1.3135e-07 | 0.003117 |
| wei20_mitoEM-R__dev | raw_threshold | 0.935874 | 0.955562 | 0.916980 | 0.059927 |
| wei20_mitoEM-R__dev | semantic | 0.935874 | 0.955562 | 0.916980 | 0.059927 |
| wei20_mitoEM-R__dev | panoptic | 0.935874 | 0.955562 | 0.916980 | 0.059927 |

## 原始概率、官方语义与原Panoptic差异

| 区域 | raw>=0.5 Dice | Semantic Only Dice | Panoptic Dice | Semantic−Panoptic | Semantic前景丢失比例 | 无中心平面 |
|---|---:|---:|---:|---:|---:|---:|
| guay21_vol1 | 0.014253 | 0.014253 | 0.014253 | 0.000000 | 0.000000 | 0 |
| muller21_high_c1 | 2.34698e-07 | 2.34703e-07 | 2.34698e-07 | 4.5307e-12 | 0.000000 | 23 |
| casser20_kasthuri++_train__dev | 0.940289 | 0.940289 | 0.940289 | 0.000000 | 0.000000 | 0 |
| wei20_mitoEM-R__dev | 0.935874 | 0.935874 | 0.935874 | 0.000000 | 0.000000 | 0 |

完整Precision/Recall/前景比例及有效前景丢失TP、丢失平面在MV4B_path_comparison.csv；每平面中心数量/保留比例在每区域JSON。官方Semantic Only把类1视为stuff，单平面总前景小于64时过滤；raw阈值掩码另存，全部平面已断言两者仅由该官方规则不同。

## 系统同图像指标

| 系统 | 来源等权Dice | Precision | Recall | HD95 nm | NSD@16nm | 前景比例 |
|---|---:|---:|---:|---:|---:|---:|
| G-P0 | 0.871240 | 0.895241 | 0.849341 | 310.484990 | 0.745350 | 0.037009 |
| G-B100 | 0.866543 | 0.893203 | 0.842703 | 331.123545 | 0.740152 | 0.036931 |
| MitoNet-Panoptic-v1 | 0.472604 | 0.489228 | 0.461086 | 685.327019 | 0.439663 | 0.026279 |
| MitoNet-Semantic-v1 | 0.472604 | 0.489228 | 0.461086 | NA | NA | 0.026279 |

G-B100在2/4来源高于官方Semantic Only。DINOv3来源均值使用原三个decoder seeds，Semantic/Panoptic分别为同一个官方固定模型的不同输出路径。

## 每来源Dice

| 来源 | P0 | B100 | Panoptic | Semantic Only | B100−Semantic |
|---|---:|---:|---:|---:|---:|
| casser20 | 0.931950 | 0.933052 | 0.940289 | 0.940289 | -0.007237 |
| guay21 | 0.751182 | 0.732921 | 0.014253 | 0.014253 | +0.718668 |
| muller21 | 0.888352 | 0.876927 | 0.000000 | 0.000000 | +0.876927 |
| wei20 | 0.913475 | 0.923271 | 0.935874 | 0.935874 | -0.012602 |

## 历史MV4敏感性分析（不删除主结果）

| 范围 | 来源数 | B100 | 既有Panoptic | 差值 |
|---|---:|---:|---:|---:|
| all12 | 12 | 0.826449 | 0.610209 | +0.216240 |
| without_Guay_Muller | 10 | 0.830101 | 0.730808 | +0.099293 |
| without_Guay_Muller_Han | 9 | 0.844457 | 0.783138 | +0.061319 |

用完整精度原指标复算，Guay/Müller合计贡献历史来源等权差值的61.74%。这是描述性敏感性分析，不改变12来源主指标或历史报告。

## 表述与结论边界

MV4的0.610209必须命名为“released MitoNet_v1逐层2D Panoptic前景union的语义Dice”。此轮Semantic Only才是直接语义输出主对照；不能把两个成绩混成一个MitoNet名次。MitoVerse监督适配后的DINOv3使用156区域标签，MitoNet未经本项目标签适配；官方训练重叠未核验，不能绝对称为无训练暴露的零样本。B100预训练含全部MitoVerse无标签图像；不是公平独立跨域测试或等监督预训练方法比较。匹配P0/B100的EM持续预训练收益仍约+0.012707。

原图与输入灰度、物理scale1、2D而非orthoplane3D的限制继续披露；本轮未做原始强度/极性转换或尺度优化，输入分布机制未被操纵。Han整幅GT按用户指令继承，GT外输出不直接证明生物学假阳性；Turner spacing UNVERIFIED继续限制nm指标。没有真实实例GT，未计算AP/PQ，不将panoptic语义覆盖差距称作实例分割优劣。

权重SHA256 `487d63dd1deb971a3e374785b350673ee5427e0ffcd44880bf9a54f6c2c4bf04`；MV4B protocol SHA256 `93f435e4fce0005e112fb2d5116114ffd8a4ab4a84b12494b97fd2e99bdb6281`。资产/源码/环境复用MV4已验证版本；实际部署commit与每文件SHA见evidence/deployment*.json。概率、raw threshold、official semantic留runs/Official-MitoNet-Semantic-v1/；旧实例/前景仍在MV4目录。

comparison_figures/*.png提供原图、GT、概率、raw>=0.5、Semantic Only、旧Panoptic、两种DINOv3首seed的八面板图；选择最大语义前景丢失平面仅用于解释，没有反馈推理。

本轮完成后停止，不自动运行灰度诊断、MitoNet fine-tuning、尺度搜索、3D或新的SSL。

## 实际执行与下一项决策

2026-10-14香港时间23:53:39四区域全Z完成；4/4推理和路径比较，pilot service inactive/MainPID0/Resultsuccess。原模型概率/原阈值掩码/官方语义掩码共12份大型数组留服务器；物理GPU0禁用、网站200。MV3/MV4保护hash/stat前后相同，既有Panoptic混淆矩阵与历史逐项一致。actual pilot code commit8a8b757b300a7dc485d5298925b9b50b38042bfb；报告脚本为事后CPU解释，不参与推理或参数选择。未启动38区域扩展或新科学实验。

继续保留当前完整系统成绩，但修正研究顺序：不再仅由0.8264对0.6102直接确定新SSL方向。本轮退休实例分组丢失这一特定解释；输入强度语义、尺度适配或实际域差异仍未鉴别。后续如另行授权，才开展一个针对候选原因的限定检查；本轮停止，不启动灰度审计、尺度搜索、MitoNet微调或预训练。

四张八面板图入口：E:/mira_website_demo/artifacts/mv4b_control_20261014/comparison_figures/MV4B_gallery.html。全部指标、每平面中心/前景丢失记录、概率资产形状/路径与实际源码hash保存本轮小型证据。旧MV4_final_report.md和所有旧注册记录未改写，本报告为独立补充。
