# MV-4B：Official Semantic vs Panoptic输出复核

同官方MitoNet_v1权重、同uint8图像/GT/有效区、norms0.57571/0.12765、scale1、confidence0.5、tile0、TTA OFF。只改变官方semantic_only选项；原始概率在同一次官方forward捕获，无GT阈值/尺度调参。MV3/MV4结果与预测全部只读保留。

实际范围：4开发区域/4来源。按用户条件先完成四区域；未将这四来源均值冒充12来源结果。

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
