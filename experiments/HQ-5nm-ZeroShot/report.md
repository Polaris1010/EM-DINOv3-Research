# HQ-5nm 零样本二维语义预测

四组共处理 80 个模型－图像组合。源清单 25 张，实际可读 20 张，像素尺寸 5 nm。缺失 ID 为 lung_01, lung_02, lung_03, lung_04, lung_05。
源清单将全部视野标记为无标注。Dice、IoU、Precision、Recall 均缺少逐像素真值，留空。下表仅描述预测前景占比。

| 模型 | 图像数 | 预测前景占比 |
|---|---:|---:|
| Official-MitoNet-v1 | 20 | 8.88% |
| Barrier100 | 20 | 13.05% |
| B100 | 20 | 7.41% |
| P0 | 20 | 8.91% |

MitoNet 使用官方发布版权重和 Engine2d。Barrier100、B100、P0 分别搭配已有 CEM-MitoLab 40k Primus 语义头，固定 decoder seed 20261014。原始网格推理，未使用目标域标签、未调参。每张预测以原始 2048×2048 的 0/1 TIFF 保存。
