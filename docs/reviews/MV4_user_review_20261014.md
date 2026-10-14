# MV-4 结果复核：成绩可能是真的，但“全面超越 MitoNet”目前还不能成立

我的判断是：这份报告在数值和执行记录上具有较高可信度，但实验公平性还存在几个重要的待确认因素。尤其 MitoNet 在 Guay 和 Müller 上接近零的 Dice，需要优先排除推理配置与数据预处理问题。

先区分三个不同的结论：

| 结论                               | 当前判断 |
| -------------------------------- | ---- |
| 当前评测协议下，G-B100 的 Dice 高于 MitoNet | 报告支持 |
| G-B100 在 12 个来源中有 8 个领先 MitoNet  | 报告支持 |
| 我们已经证明 DINOv3 预训练方法全面优于 MitoNet  | 不支持  |
| 我们已经在公平、独立的测试协议下超过 MitoNet       | 尚未证明 |

原因不是应该再设置大量前置审核，而是目前比较的两个系统存在实质性差别：

G-B100 使用了 156 个 MitoVerse 训练区域的线粒体监督标签，并在自监督预训练时接触过全部 MitoVerse 图像；MitoNet 使用官方预训练及监督权重，未针对这 156 个区域进行微调。

所以目前比较的是：

> MitoVerse 监督适配后的 DINOv3 Generalist vs 原版零样本 MitoNet。

这是有价值的完整系统比较，但不是等训练数据、等目标域暴露条件下的预训练方法比较。

## 一、发现一个需要优先核实的潜在不公平因素：Panoptic 输出不等于原始 Semantic 输出

我查了官方源码，发现这不是一般性的猜测，而是一个确实存在的机制差异。

MitoNet 的 PanopticDeepLab 推理大致为：

```
MitoNet
   ├── Semantic logits ────→ 线粒体前景概率
   │
   ├── Center heatmap ─────→ 实例中心
   │
   └── Offset vectors ─────→ 像素所属实例
                │
                ▼
         Panoptic grouping
                │
                ▼
         最终实例分割
```

在官方 `PanopticDeepLabRenderEngine` 实现中，如果没有检测到合格实例中心，实例分组可能生成全零的 instance map；随后 `merge_semantic_and_instance` 只保留成功分配实例的前景像素。

这意味着即使原始语义分支已经预测出一些线粒体，最终 panoptic 输出仍然可能接近空白。

我已核对的源码：

- [MitoNet PanopticDeepLabRenderEngine](https://github.com/volume-em/empanada/blob/main/empanada/inference/engines.py)
- [merge_semantic_and_instance](https://github.com/volume-em/empanada/blob/main/empanada/inference/postprocess.py)
- [官方 Engine2d](https://github.com/volume-em/empanada-napari/blob/main/empanada_napari/inference.py)

官方 empanada 也明确提供了 Semantic Only 推理选项，可跳过实例分组，直接进行语义分割。

[image](https://www.google.com/s2/favicons?domain=https://empanada.readthedocs.io\&sz=32)

empanada 0.1.4 documentation



这对你当前的比较极为重要。

你现在的 G-B100 输出是直接预测前景/背景的语义分割；而 MitoNet 的报告使用的是实例分组完成后的 panoptic 输出，再通过 `>0` 转为二值掩码。

所以现有 MV-4 并不是完全相同输出路径的二值分割比较。

我建议正式保留两项 MitoNet 成绩：

| 对照               | 使用的输出                 | 用途                   |
| ---------------- | --------------------- | -------------------- |
| MitoNet-Semantic | 官方语义分支，按官方阈值生成二值 mask | 与 G-B100 进行语义分割主对照   |
| MitoNet-Panoptic | 当前完整实例分组输出再转二值 mask   | 比较官方默认端到端实例系统的语义覆盖能力 |

请注意：这不是让你随意调整 MitoNet 的置信度阈值，而是利用同一个官方模型原本就支持的两种推理模式，把语义能力和实例分组能力分开衡量。

特别针对：

Guay · MitoNet

# 0.0143

Recall 0.0088

Müller · MitoNet

# 0.0002

Recall 0.0001

这两个来源几乎完全没有检出线粒体，远比普通的跨域性能下降更极端。

如果 Semantic Only 模式下这两个来源的表现明显恢复，就说明当前巨大差距至少有一部分来自实例中心检测或 panoptic 后处理，而不是 MitoNet 对线粒体的语义识别能力完全失效。

反过来，如果 Semantic Only 也接近零，才更值得考虑输入灰度分布、成像域变化或模型本身的泛化不足。

## 二、我核对到的其他不平衡因素

| 因素                      | 当前情况                                                            | 影响程度                   |
| ----------------------- | --------------------------------------------------------------- | ---------------------- |
| 监督训练数据不相等               | DINOv3 在 MitoVerse 156 个区域进行了监督训练；MitoNet 没有在这套训练区域上微调          | 很高                     |
| Semantic vs Panoptic 输出 | DINOv3 直接输出语义分割；MitoNet 先进行实例分组再转二值图                            | 很高                     |
| 输入强度分布                  | MitoVerse 的 uint8 图像经历来源相关归一化；MitoNet 使用自己的训练集均值和标准差            | 中高，需核实                 |
| 物理尺度和推理设置               | MitoNet 固定 scale=1，DINOv3 采用其已训练好的统一推理流程                        | 中高，尚未量化                |
| SSL 测试图像暴露              | B100k 在预训练时接触过所有 MitoVerse 图像                                   | 很高，影响泛化结论              |
| 2D 与 3D 能力              | 本轮 MitoNet 只运行了逐层 2D，没有使用 3D slice queue 或 orthoplane consensus | 影响与完整 3D MitoNet 系统的比较 |
| 实例分割评价                  | MitoNet 已保留实例预测；DINOv3 当前仅有语义预测                                 | 无法判断实例分割优劣             |

其中有三点值得具体说明。

### 1. 模型权重没有发现明显问题，但输入分布仍需留意

我确认 Zenodo 官方发布的 `MitoNet_v1.pth` MD5 为：

`b56fce28d877a3c35b7ca970cec73bc9`

与你的实际记录一致，文件大小也匹配。这支持你确实下载了官方完整权重，而不是 mini 或其他 checkpoint。

[image](https://www.google.com/s2/favicons?domain=https://zenodo.org\&sz=32)

Zenodo



我还核查了官方 MitoNet 配置，其预处理参数为：

```
norms:
  mean: 0.57571
  std: 0.12765
```

而 MitoVerse ingest 支持将原始 EM 图像映射为 uint8，部分来源经过强度截断和归一化。这可能改变图像相对于 MitoNet 训练数据的分布。[MitoVerse ingest 实现](https://github.com/PytorchConnectomics/mitoverse/blob/main/scripts/ingest.py)

这不代表现有预处理错误。只说明不能把“调用了官方 Preprocessor”直接等同于“输入一定与 MitoNet 训练分布匹配”。

尤其 Guay 和 Müller 这种近乎完全失败的情况，建议对照原始数据与当前 Zarr 图像的灰度方向、直方图和空间尺寸，确认是否发生了影响模型输入语义的变换。

### 2. Scale=1 是一种合法设置，但不是唯一合理的 MitoNet 运行方式

官方 empanada 文档明确建议在新 EM 数据上比较不同下采样尺度，并指出适度下采样有时反而能改善实例分割和减少过度分裂。

[image](https://www.google.com/s2/favicons?domain=https://empanada.readthedocs.io\&sz=32)

empanada-napari-v1.2 1.2.4 documentation



因此当前 scale=1 的结果可以保留为固定配置基线，但不宜据此宣称 MitoNet 在所有尺度设置下的最佳性能。

不过我不建议现在开展大规模参数搜索。这会重新制造你此前遇到的实验循环，而且可能造成开发集过度调参。

### 3. 目前 +0.2162 的整体优势受到两个极端失败来源的明显影响

我根据你报告中的 12 个来源 Dice 重新计算了敏感性分析：

| 统计范围           | G-B100 Dice | MitoNet Dice | B100 优势 |
| -------------- | ----------- | ------------ | ------- |
| 全部 12 来源       | 0.8264      | 0.6102       | +0.2162 |
| 排除 Guay、Müller | 0.8301      | 0.7308       | +0.0993 |
| 再排除 Han        | 0.8445      | 0.7831       | +0.0613 |

这是敏感性分析，不是建议删除来源或修改正式主结果。数值根据报告中四位小数的来源均值复算，存在微小舍入误差。

特别值得注意：

Guay 和 Müller 两个来源合计贡献了约 61.7% 的来源等权 Dice 优势。

因此，虽然排除它们后 G-B100 仍然领先，但整个比较的优势幅度已经大幅缩小。

这就是为什么需要优先确认这两个来源的 MitoNet 预测究竟出了什么问题。

## 三、现在不要重跑全部实验，只做一次针对性验证

我建议 Codex 下一步只执行：

MV-4B：MitoNet Semantic vs Panoptic Output Sanity Check

具体要求很简单：

1. 保留所有 MV-4 历史结果，绝不覆盖。
2. 先选 Guay、Müller 各一个现有验证区域，再选 Casser、Wei 各一个表现正常的参照区域。
3. 使用相同官方 MitoNet 权重、完全相同输入和归一化参数，分别获取原始 semantic probability、Semantic Only 二值预测、现有 Panoptic 二值预测。
4. 比较各自 Dice、Precision、Recall、前景比例，并检查语义前景有多少在实例分组阶段丢失。
5. 如果两条输出路径确实存在明显差异，再把官方 Semantic Only 推理扩展到完整 38 个开发区域。

不重新训练任何模型，不修改 DINOv3 的 TTA=OFF 设置，也不依据 GT 调整 MitoNet 阈值。

这样就可以快速区分：

| 验证结果               | 含义                         |
| ------------------ | -------------------------- |
| Semantic Only 明显更好 | Panoptic 分组是当前语义对照中的重要影响因素 |
| Semantic Only 仍接近零 | 进一步考虑输入预处理或真实域外识别失败        |
| 两者整体接近             | 现有 MitoNet 结果的解释会更加可靠      |

## 四、怎样表述当前成果才严谨？

目前可以写：

> 在包含 12 个 EM 来源、38 个开发区域的 transductive 多来源评测中，使用原版 Primus 和 MitoVerse 监督训练的 DINOv3-B100k，获得来源等权 Dice 0.8264，高于未经本项目数据微调的官方 MitoNet_v1 panoptic 输出所对应的语义 Dice 0.6102。

但目前不应写：

> 我们提出的自监督方法在公平的跨域泛化测试中全面超越 MitoNet 21.62 个百分点。

后一种表述同时忽略了监督数据、目标域图像暴露、语义与实例输出路径以及未独立测试的问题。

另外，你真正可以明确归因于现有 EM 预训练对照的改善，是在同等监督训练与 decoder 条件下：

\\[ Dice\_{\mathrm{B100}}-Dice\_{\mathrm{P0}}=+0.0127 \\]

这个结果虽然小得多，却更能说明你当前 DINOv3 持续预训练的实际研究价值。

## 最后判断

我没有发现报告中权重身份、结果算术或同图像评测的一项明确造假证据。 但我没有直接访问你的服务器和 38 个区域的原始预测文件，因此不能把交付报告的自检等同于我独立复现了整个实验。

目前发现的最大问题不是 SHA、脚本或评测是否能够正常运行，而是：

我们把一个经过实例分组的 MitoNet 输出，与直接语义分割的 DINOv3 进行了比较。

这可能对两个极端低分来源产生重大影响。它必须先查清楚。

我建议现在只做 MV-4B 这一项短实验。如果 Semantic Only 复核后 G-B100 仍在大多数来源领先，我们就能更有把握地判断模型的系统竞争力，然后再决定是否进一步开展同监督数据的 MitoNet fine-tuning 对照。

不需要推翻 MV-4，也不需要重训 DINOv3。现有结果保留；下一步只排除一个最有可能改变结论的关键混杂因素。