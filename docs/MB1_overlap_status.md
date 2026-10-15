# MB1来源与训练暴露状态

所有七组照常评测。以下是有限来源/资产核对，不做全面数据审计，不据此筛掉数据或改变推理。

| Benchmark | B100k无标签池与资产线索 | MV3/MV5监督暴露 |
|---|---|---|
| c_elegans | CONFIRMED_SOURCE_AND_SAMPLED_IMAGE_OVERLAP | 候选volume不在156 train中；同来源/采集关系仍须谨慎解释 |
| fly_brain | CONFIRMED_SOURCE_AND_SAMPLED_IMAGE_OVERLAP | 候选volume不在156 train中；同来源/采集关系仍须谨慎解释 |
| glycolytic_muscle | CONFIRMED_SOURCE_AND_SAMPLED_IMAGE_OVERLAP | 候选volume不在156 train中；同来源/采集关系仍须谨慎解释 |
| hela_cell | CONFIRMED_SOURCE_AND_SAMPLED_IMAGE_OVERLAP | 候选volume不在156 train中；同来源/采集关系仍须谨慎解释 |
| lucchi_pp | CONFIRMED_SOURCE_ASSOCIATION_EXACT_REGION_UNCONFIRMED | 候选volume不在156 train中；同来源/采集关系仍须谨慎解释 |
| salivary_gland | CONFIRMED_SOURCE_AND_SAMPLED_IMAGE_OVERLAP | 候选volume不在156 train中；同来源/采集关系仍须谨慎解释 |
| TEM-100 | NO_OVERLAP_FOUND_IN_AVAILABLE_REGISTRY | 未发现TEM benchmark名称匹配；精确图像暴露仍UNVERIFIED |

五个Conrad23来源已列入B100的212-volume SSL清单，形状相同，首/中/末三平面与官方原图逐像素相等：确认来源与被抽样图像重叠；没有声称全体积逐像素/哈希比较。Lucchi++与casser20_lucchi++有来源关联，但原生形状[165,768,1024]对[82,768,512]不同，精确坐标暴露未确认；未做重采样、尺度或空间搜索。

TEM-100在现有212-volume来源/文件名清单中未发现匹配，属于NO_OVERLAP_FOUND而非已证实无图像暴露。没有穷举比对CEM1.5M、CEM-MitoLab或自然图像上游训练池，官方MitoNet的精确上游图像交叉关系仍UNVERIFIED。

G-P0没有本项目EM持续预训练，但其decoder使用MV3监督数据。G-B100具有上述无标签暴露。MitoNet-MV继承官方MitoNet再用同156区域监督适配。无标签暴露不等同监督标签泄漏，但本轮不能称严格独立、完全未见图像的外部泛化测试；三个decoder seeds不增加独立生物采集数量。

检查证据：evidence/overlap_source_check.json、已锁定SSL manifest与MV-Generalist-v2训练条目。所有官方GT仅用于固定评分/展示，不进入训练或参数选择。
