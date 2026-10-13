# 第六轮：原始来源追踪更新

2026-09-17。继续执行后的新增证据；未启动正式训练。

## 已完成

成功从 Europe PMC 获取 Han 2024 官方补充包（22,530,590 字节），ZIP 可完整打开；其中 Source Data 包包含 Dataset Neuroglancer Link.txt。

- 原始 EM 地址：`gs://vclem-xh/cerebellum/em`。
- 官方 info 元数据：XYZ 尺寸 57344 × 55296 × 880；分辨率 8 × 8 × 30 nm；uint8，单通道。
- 原始分割地址包括 `segmentation_16nm` 和 `segmentation_32nm`。
- 补充包 SHA256：`5ec240e7c1274ba4957a1e9326929cf52c48d398e7b73bef84f79354402131b0`。
- Source Data ZIP SHA256：`c4a8cf6a8cb307673d74df53b7acd25861c52d6aea22da2547e48a20db914aa6`。

[论文](https://www.nature.com/articles/s41467-024-50411-z)；[官方附件镜像](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11300613/supplementaryFiles)；[原始扫描元数据](https://storage.googleapis.com/vclem-xh/cerebellum/em/info)。

本地五个 Han 文件的 source 字段指向 MitoLE 导出 h5；未记录原始扫描裁剪坐标。已检查上游 EM、16nm 分割目录的顶层文件及本地导入脚本，未找到五个 ROI 的导出变换。只记录父数据集地址，不将浏览器 position 填成 ROI 原点，不将五个 ROI 当成五个独立 specimen。

另确认 muller21 对应作者 BetaSeg 项目和公开下载入口；论文描述七个细胞。尚未取得逐导出文件与原始动物、扫描的对应证据，不凭细胞文件名解除隔离。

[BetaSeg 作者项目页](https://betaseg.github.io/)。

## 当前决定

mask 修复与工程检查已经完成。本次补全上游来源地址，但没有完成全部训练候选与 validation / legacy_dev_test 的原始 specimen 独立性审计。SSL allowlist 保持为空，S2/S3/S4 不启动；没有新增 Dice 结果。

下一项必需证据是本地导出到原始扫描的映射：导出文件名、父扫描 URI、specimen 关联、ROI 起止坐标、坐标轴及重采样变换。应从原始导出记录补齐，不能用图像相似度单独证明独立。其他来源及 JRC 仍需对应扫描关联。当前不需要重复批准，但需要新增来源证据才能按原方案继续训练。

证据文件：han_upstream_source_evidence.json、han_neuroglancer_state.json、han_em_info.json、han_bucket_metadata_listing.json。此次未修改原始图像、原权重、旧评估表或既定划分。
