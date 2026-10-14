# MV-4：Official MitoNet vs DINOv3 Generalist

三个完整系统在MV-Generalist-v2的同38开发区域、12来源、同有效GT和物理坐标上比较。G-P0/G-B100直接复用40k、三个decoder seeds的既有预测/指标；本轮无训练、无重新推理DINOv3、无TTA修改。

来源等权Dice：P0 **0.813741**，B100 **0.826449**，官方MitoNet **0.610209**。B100−MitoNet **+0.216240**；B100在8/12来源更高、4/12更低。

## 来源等权完整指标汇总

| 系统 | Dice | Precision | Recall | HD95 nm | NSD@16nm | 预测前景比例 |
|---|---:|---:|---:|---:|---:|---:|
| G-P0 | 0.8137 | 0.8386 | 0.8113 | 529.2191 | 0.7059 | 0.0791 |
| G-B100 | 0.8264 | 0.8438 | 0.8351 | 559.4837 | 0.7255 | 0.0815 |
| Official-MitoNet-v1 | 0.6102 | 0.6818 | 0.6575 | 617.8221 | 0.5434 | 0.0812 |

B100−MitoNet Dice为+0.216240（21.6240个百分点）；本轮8来源更高、4来源更低。MitoNet在Casser/Mekuc/Turner/Wei的Dice领先B100，分别+0.007237/+0.010896/+0.040620/+0.012602。整体Dice排序不等于逐来源、逐指标一致排序，完整来源指标见下表。

## 实际推理与数据协议

官方MitoNet_v1来自Zenodo6861565，发布者MD5及文件字节数校验后另计算SHA256；完整权重和输出留在服务器。官方empanada-dl0.1.7与hash锁定Engine2d/Preprocessor原定义，通过AST原样编译提供无Qt入口；没有重写模型/后处理。官方二维批量工作流：native XY/Z、scale1、confidence0.5、center threshold0.1、NMS kernel3、label divisor10000、padding16、coarse boundaries、fill holes OFF、TTA OFF；采用官方默认整图路径tile_size=0，最大4096×4096原生图像smoke通过。初始tile1280尝试因官方Tiler兼容错误而失败，发生在正式预测/任何指标之前，旧协议与回执另存。

推理不读取GT/annotation mask值，不使用GT box。所有原生Z推理；有效区仅参与评测。Han按用户确认忽略mask、整幅GT。保留原始逐层panoptic ID，以及跨层不重复的逐层instance ID；这些ID不代表三维追踪。

Manifest SHA256 `787fa42d2d944844d75d9e70bb23bd095dd2375bbc1a9d2123a37292d7c2b259`；MV4 protocol SHA256 `66db27f82a731c2edb81a74c681b2cd00a747abecfcc3c0c6e9f9082033ebe4b`。实际资产哈希/来源详见MV4_summary.json的official_assets及MV4_protocol.json。

Dice/Precision/Recall来自有效像素混淆矩阵；HD95 nm和NSD@16nm直接调用MV3沿用的MV2物理距离实现。来源内等区域，来源间等权；DINOv3每个来源先平均三个seed，MitoNet是一个固定官方系统。undefined precision/recall保留NA，inf HD95不隐藏。

## 每来源Dice

| 来源 | G-P0 | G-B100 | Official MitoNet | B100−MitoNet |
|---|---:|---:|---:|---:|
| casser20 | 0.9319 | 0.9331 | 0.9403 | -0.0072 |
| guay21 | 0.7512 | 0.7329 | 0.0143 | +0.7187 |
| haberl18 | 0.5208 | 0.5209 | 0.1697 | +0.3512 |
| han24 | 0.6751 | 0.7009 | 0.2598 | +0.4411 |
| jiang25 | 0.8622 | 0.8735 | 0.8270 | +0.0465 |
| kunduri22 | 0.9460 | 0.9515 | 0.9219 | +0.0296 |
| mekuc20 | 0.9292 | 0.9351 | 0.9460 | -0.0109 |
| muller21 | 0.8917 | 0.8835 | 0.0002 | +0.8833 |
| openorganelle | 0.6511 | 0.7412 | 0.6255 | +0.1157 |
| turner22 | 0.7715 | 0.7959 | 0.8365 | -0.0406 |
| wei20 | 0.9135 | 0.9233 | 0.9359 | -0.0126 |
| xiao18 | 0.9206 | 0.9257 | 0.8455 | +0.0803 |

## 完整指标

| 来源 | 模型 | Dice | Precision | Recall | HD95 nm | NSD@16nm | 前景比例 |
|---|---|---:|---:|---:|---:|---:|---:|
| casser20 | G-P0 | 0.9319 | 0.9396 | 0.9244 | 34.3469 | 0.9085 | 0.0363 |
| casser20 | G-B100 | 0.9331 | 0.9408 | 0.9255 | 48.7736 | 0.9096 | 0.0363 |
| casser20 | Official-MitoNet-v1 | 0.9403 | 0.9630 | 0.9186 | 30.0000 | 0.9162 | 0.0352 |
| guay21 | G-P0 | 0.7512 | 0.8025 | 0.7066 | 461.4509 | 0.6289 | 0.0264 |
| guay21 | G-B100 | 0.7329 | 0.7933 | 0.6815 | 605.4419 | 0.6043 | 0.0258 |
| guay21 | Official-MitoNet-v1 | 0.0143 | 0.0383 | 0.0088 | 1605.7709 | 0.0102 | 0.0069 |
| haberl18 | G-P0 | 0.5208 | 0.7140 | 0.4103 | 147.0386 | 0.5910 | 0.0451 |
| haberl18 | G-B100 | 0.5209 | 0.7614 | 0.3961 | 167.8362 | 0.5959 | 0.0408 |
| haberl18 | Official-MitoNet-v1 | 0.1697 | 0.9589 | 0.0931 | 361.8232 | 0.1969 | 0.0076 |
| han24 | G-P0 | 0.6751 | 0.5573 | 0.8564 | 2889.3761 | 0.4227 | 0.0352 |
| han24 | G-B100 | 0.7009 | 0.5592 | 0.9387 | 2913.0223 | 0.5147 | 0.0384 |
| han24 | Official-MitoNet-v1 | 0.2598 | 0.1514 | 0.9157 | 2998.3254 | 0.1912 | 0.1384 |
| jiang25 | G-P0 | 0.8622 | 0.8402 | 0.8854 | 307.0399 | 0.7509 | 0.1210 |
| jiang25 | G-B100 | 0.8735 | 0.8482 | 0.9008 | 189.8826 | 0.7712 | 0.1220 |
| jiang25 | Official-MitoNet-v1 | 0.8270 | 0.7953 | 0.8612 | 113.6882 | 0.6996 | 0.1243 |
| kunduri22 | G-P0 | 0.9460 | 0.9515 | 0.9406 | 531.4461 | 0.7366 | 0.3006 |
| kunduri22 | G-B100 | 0.9515 | 0.9471 | 0.9560 | 501.6528 | 0.7618 | 0.3070 |
| kunduri22 | Official-MitoNet-v1 | 0.9219 | 0.9548 | 0.8913 | 337.8994 | 0.6990 | 0.2839 |
| mekuc20 | G-P0 | 0.9292 | 0.9057 | 0.9540 | 213.0034 | 0.9255 | 0.0580 |
| mekuc20 | G-B100 | 0.9351 | 0.9192 | 0.9515 | 127.8770 | 0.9408 | 0.0570 |
| mekuc20 | Official-MitoNet-v1 | 0.9460 | 0.9387 | 0.9533 | 136.9379 | 0.9533 | 0.0559 |
| muller21 | G-P0 | 0.8917 | 0.9019 | 0.8821 | 364.6147 | 0.7131 | 0.0215 |
| muller21 | G-B100 | 0.8835 | 0.8939 | 0.8740 | 459.0585 | 0.6840 | 0.0215 |
| muller21 | Official-MitoNet-v1 | 0.0002 | 0.0031 | 0.0001 | 1123.2256 | 0.0003 | 0.0018 |
| openorganelle | G-P0 | 0.6511 | 0.7442 | 0.6483 | 573.7231 | 0.4684 | 0.1332 |
| openorganelle | G-B100 | 0.7412 | 0.7937 | 0.7692 | 629.2191 | 0.5515 | 0.1467 |
| openorganelle | Official-MitoNet-v1 | 0.6255 | 0.8179 | 0.5420 | 402.3325 | 0.4972 | 0.1219 |
| turner22 | G-P0 | 0.7715 | 0.8699 | 0.6946 | 312.5966 | 0.6232 | 0.0513 |
| turner22 | G-B100 | 0.7959 | 0.8265 | 0.7711 | 599.5256 | 0.6286 | 0.0604 |
| turner22 | Official-MitoNet-v1 | 0.8365 | 0.8568 | 0.8176 | 175.0712 | 0.6756 | 0.0619 |
| wei20 | G-P0 | 0.9135 | 0.9298 | 0.8978 | 487.2783 | 0.7530 | 0.0603 |
| wei20 | G-B100 | 0.9233 | 0.9341 | 0.9127 | 447.9728 | 0.7894 | 0.0610 |
| wei20 | Official-MitoNet-v1 | 0.9359 | 0.9556 | 0.9170 | 78.7909 | 0.8322 | 0.0599 |
| xiao18 | G-P0 | 0.9206 | 0.9070 | 0.9347 | 28.7142 | 0.9489 | 0.0606 |
| xiao18 | G-B100 | 0.9257 | 0.9081 | 0.9442 | 23.5425 | 0.9540 | 0.0612 |
| xiao18 | Official-MitoNet-v1 | 0.8455 | 0.7482 | 0.9718 | 50.0000 | 0.8489 | 0.0764 |

## 典型错误与实例输出

MV4_failure_cases.csv及failure_cases/*.png保存每来源MitoNet最低Dice区域、该区域最大FP+FN平面，展示原图、有效GT、两种DINOv3首seed、MitoNet前景及实例颜色。此选择发生于完成推理后，只用于检查错误；不会反馈到阈值/推理。其他两个seed仍完整纳入所有数值。

MitoNet的raw panoptic、逐层instance ID及semantic foreground全部保留在runs/Official-MitoNet-v1/*_{panoptic,instances,prediction}.npy。当前没有将二值GT的连通分量冒充真实实例GT；未声称实例AP/PQ或三维追踪优劣。弱边界/小目标的机制性判断需结合实图，单个语义Dice无法证明。

### 已查看对照图的具体观察

- Guay：MitoNet预测少量目标，许多与GT位置不一致；来源Recall0.008754、Precision0.038330，不能仅描述为边界轻微误差。Müller：所示高分辨率区域大量GT目标未被检出，来源Recall0.000086。这里只记录实际输出，不将极低Dice解释成GT极性错误或修改输入/阈值。
- Haberl：三个系统都有漏检，MitoNetRecall0.093084，B100Recall0.396116；所示图中MitoNet只覆盖少量目标，仍是完整系统共同短板。
- Han：MitoNetRecall0.915742但Precision0.151399，预测前景比例0.138401；B100分别0.938732/0.559248/0.038414。对照图显示MitoNet覆盖许多GT以外的细胞内结构。由于本轮按用户要求整幅GT评分，而历史标注完整性存在限制，“GT外预测”不能直接判定为真实生物学假阳性；不得据此翻转mask或修改协议。
- Wei：MitoNetDice0.935874高于B1000.923271，HD95为78.79nm对447.97nm，NSD为0.832164对0.789448；此来源体现Dice和边界指标的同时优势。Kunduri示例中三者覆盖总体接近，MitoNet保留逐层实例分离结果，但没有真实实例GT指标证明实例能力排序。
- Turner的自动最坏平面图呈现全黑输入及大量GT，不能用这一张图概括整个来源的弱边界能力；原样保留并披露其局限，不启动额外数据审计或改动MV3缓存。Turner来源级MitoNetDice0.836497对B1000.795877仍按既定38区域协议报告。

12张合并图留在服务器failure_cases/，本机浏览入口为E:/mira_website_demo/artifacts/mv4_control_20261014/failure_cases/MV4_failure_case_gallery.html。图片SHA与位置记录在experiments/MV-4/MV4_figure_manifest.json；原图/预测/权重不入Git。

## 研究判断与边界

多数来源G-B100超过官方MitoNet；下一项优先决策是研究Guay/Müller跨来源稳健性。这里只提出决策，不启动SSL。

排除OpenOrganelle后，MV3的B100−P0来源等权增益为+0.005667，与用户指出的收益集中一致。MV4不重复MV3训练，也不将MV2/MV3不同监督数据协议混成排行榜。

官方MitoNet训练于CEM-MitoLab，其训练图像/标注与我们的MitoVerse156训练区域不同；具体重叠映射未核验，因此本轮是完整系统竞争力开发对照，不是等监督预算、等训练暴露或严格隔离测试。B100有无标签transductive exposure；三个decoder seeds不增加独立acquisition数量。Han整幅GT沿用用户确认；Turner物理spacing仍UNVERIFIED，nm表面指标须附此限制。

本轮完成后停止。未启动Gram/HRA/LAST/TopoCL、encoder预训练、head延长训练或新对照模型。

## 执行、哈希与停止状态

2026-10-14香港时间23:07:15启动最终协议；23:15:04核验推理38/38、评分38/38，service inactive/MainPID0/Result=success。网站200，GPU0未使用；MV3输入before/after保护回执逐项相同。实际运行代码commit `db2ae4e2007c85b9badee722fd331fad138d17c2`；后续提交只补本轮完成结果和交付记录，没有重新推理或训练。

官方MitoNet权重SHA256 `487d63dd1deb971a3e374785b350673ee5427e0ffcd44880bf9a54f6c2c4bf04`，221241200bytes，发布者MD5 `b56fce28d877a3c35b7ca970cec73bc9`已验证。六组既有DINOv3 head的逐checkpoint SHA原样保留在266行表的head_sha256列及MV3最终报告。

推理：独立Python3.11.9/Torch2.5.1、empanada-dl0.1.7、NumPy1.26.4/Numba0.59.1；评分与报告：原meddinov3环境、NumPy2.4.4，距离evaluator SHA `143c5403458ca3fad215880ac7afd14e4fd066fbee8342051a0cc49a083e4d54`。原始机器生成报告另存evidence/MV-4/MV4_final_report_generated.md，本报告追加指标概览、已查看图片的观察与执行说明，不改变数值。资产、部署、smoke和最终状态回执一并保存。

最终266条指标已通过CPU核验：38条MitoNet、228条复用DINOv3；38区域/12来源全覆盖，混淆矩阵与Dice算术一致，权重/协议SHA关联正确，原DINOv3指标及保护输入未变化。研究与评分服务已经停止；下一项跨来源稳健性研究仅为科学决策，尚未执行。
