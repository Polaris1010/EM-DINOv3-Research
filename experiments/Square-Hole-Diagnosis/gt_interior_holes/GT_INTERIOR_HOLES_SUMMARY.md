# MB-1 去边界内部漏洞评测

状态 **完整**。物理 GPU 1 复算三种 C1 二维推理并评分，CPU 计算 GT 形态与统计。实际总耗时 2433.06 秒。所有 C1 掩膜逐图复核与原全量评测的混淆计数一致。
MitoNet-v1 的六个体数据直接使用已有的官方三轴 Engine3d 共识掩膜。TEM-100 使用官方 Engine2d 复算。表中 MitoNet 是三维+二维混合参照。此轮未纳入可选四相位。
保留逐切片 GT 实例 ID。实例 recall <10% 单独计为整颗漏检。已检出实例逐个以 disk(3/5) 腐蚀。内部漏分率在组内按符合条件的内部核像素汇总。封闭孔以预测掩膜的 4 连通背景填孔判断，其他内部漏分计为开放掏空。
受影响实例分母采用全部已检出实例，腐蚀后无核的实例不计为受影响。CSV 另给仅在可评估实例中的受影响率以及无核跳过率。网格对齐统计的是面积至少 64 像素的内部漏分连通域的四条外接框边缘，距 16 像素格线不超过 1 像素。组间七组等权，分母为零的率记 NA。

| 配置 | 半径 | 视野 | 内部漏分 | 封闭 / 开放 | 受影响实例 | 整颗漏检 | 无核跳过 | 大漏洞网格对齐 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 Baseline | 3 | 2595 | 11.84% | 4.58% / 95.42% | 33.50% | 22.41% | 1.08% | 32.83% |
| C1 Baseline | 5 | 2595 | 10.56% | 7.43% / 92.57% | 23.51% | 22.41% | 7.47% | 31.81% |
| C1 Stride-120 | 3 | 2595 | 10.96% | 4.19% / 95.81% | 31.88% | 22.38% | 1.02% | 27.68% |
| C1 Stride-120 | 5 | 2595 | 9.63% | 6.74% / 93.26% | 22.32% | 22.38% | 7.32% | 28.63% |
| C1 Dual-Phase | 3 | 2595 | 10.48% | 3.95% / 96.05% | 31.34% | 23.66% | 0.87% | 24.34% |
| C1 Dual-Phase | 5 | 2595 | 8.93% | 6.07% / 93.93% | 21.71% | 23.66% | 6.94% | 23.55% |
| MitoNet-v1 | 3 | 2595 | 14.11% | 0.51% / 99.49% | 27.87% | 23.00% | 2.04% | 17.29% |
| MitoNet-v1 | 5 | 2595 | 13.90% | 0.91% / 99.09% | 21.10% | 23.00% | 10.13% | 16.10% |

## 七组 r=3

| 组 | C1 Baseline 漏分 / 受影响 | C1 Stride-120 | C1 Dual-Phase | MitoNet-v1 |
|---|---:|---:|---:|---:|
| TEM-100 | 8.89% / 37.11% | 8.64% / 33.50% | 7.67% / 31.71% | 5.89% / 26.72% |
| c_elegans | 14.19% / 30.29% | 14.19% / 30.29% | 11.85% / 27.54% | 21.71% / 30.03% |
| fly_brain | 3.52% / 18.71% | 3.52% / 18.71% | 3.57% / 18.91% | 0.50% / 3.31% |
| glycolytic_muscle | 1.37% / 5.48% | 2.16% / 8.31% | 1.16% / 5.12% | 0.89% / 2.80% |
| hela_cell | 8.16% / 27.77% | 8.16% / 27.77% | 9.86% / 33.55% | 3.09% / 13.33% |
| lucchi_pp | 2.04% / 26.44% | 1.61% / 22.62% | 1.84% / 21.59% | 2.25% / 21.88% |
| salivary_gland | 44.70% / 88.74% | 38.44% / 81.96% | 37.42% / 80.96% | 64.42% / 97.02% |

## 人工核对图

原图加四列预测。绿色为真阳性，黄色为检测到的实例边界带漏分，红色为其内部漏分，蓝色为背景假阳性。灰色单列整颗漏检实例。六个指定来源分别选 Baseline r=3 最高的两张视野，四个配置显示相同视野。
- [TEM-100 · TEM_TCzgvIZVHqNFtdJEPaXf z=0](visual_panels/TEM-100__TEM_TCzgvIZVHqNFtdJEPaXf__z0.png)
- [TEM-100 · TEM_bUsTjDOJHiHMPCNIWkuT z=0](visual_panels/TEM-100__TEM_bUsTjDOJHiHMPCNIWkuT__z0.png)
- [c_elegans · c_elegans z=251](visual_panels/c_elegans__c_elegans__z251.png)
- [c_elegans · c_elegans z=250](visual_panels/c_elegans__c_elegans__z250.png)
- [glycolytic_muscle · glycolytic_muscle z=53](visual_panels/glycolytic_muscle__glycolytic_muscle__z53.png)
- [glycolytic_muscle · glycolytic_muscle z=52](visual_panels/glycolytic_muscle__glycolytic_muscle__z52.png)
- [hela_cell · hela_cell z=134](visual_panels/hela_cell__hela_cell__z134.png)
- [hela_cell · hela_cell z=132](visual_panels/hela_cell__hela_cell__z132.png)
- [lucchi_pp · lucchi_pp z=115](visual_panels/lucchi_pp__lucchi_pp__z115.png)
- [lucchi_pp · lucchi_pp z=114](visual_panels/lucchi_pp__lucchi_pp__z114.png)
- [salivary_gland · salivary_gland z=561](visual_panels/salivary_gland__salivary_gland__z561.png)
- [salivary_gland · salivary_gland z=1014](visual_panels/salivary_gland__salivary_gland__z1014.png)

[汇总 CSV](summary_all.csv) · [逐组 CSV](summary_by_group.csv) · [逐切片 CSV](per_plane.csv)
