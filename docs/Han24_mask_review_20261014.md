> 历史v1核验记录。后续用户明确要求忽略mask、整幅GT参与loss；当前v2继承此用户指令，本页非零mask规则不再控制训练或评测。原图/GT/mask照片保留供人工查看。

# Han24有效区依据与人工复核图

本次仅核实Han24，不改变其他来源、MV-Generalist-v1、40k预算、encoder/decoder、采样或选模规则。没有参考任何模型Dice来决定mask极性，也没有重复全量GT审计。

## 原始标注语义

Han等2024年原论文的Methods / Mitochondria detection明确：标注局限于选定的苔藓纤维末端，模型loss只计算含已标注末端的区域。[原论文](https://www.nature.com/articles/s41467-024-50411-z)。

论文支持“局部末端内监督”，没有直接规定本地MitoVerse导出数组中0/非0的编码。未找到原exporter明确的极性声明，因此不能把“mask!=0”冒称为论文逐字规定。实际极性采用已有GT-mask空间证据，与原始标注语义相一致。

## 已有空间验证，直接继承

复用2026-09-17保存的 `han_mask_full_audit.json`、`han_outside_region_review.json` 和 `mask_policy_v2.json`，本轮不重新扫描五个体积。全量旧检查：非零mask覆盖7.5692%体素；57,909,557个正GT在非零区，79,683个正GT在零区，正GT包含率99.8626%。非零值为末端区域编号，不是线粒体类别；例外不能全部解释成一个像素的边界误差。

|体积|非零mask编号|旧检查：正GT在非零区|旧检查：正GT在零区|
|---|---:|---:|---:|
|neg0|13|10,563,695|29,895|
|neg1|14|12,653,281|268|
|pos0|1|12,536,634|104|
|pos1|11|11,422,723|49,068|
|pos2|12|10,733,224|348|

因此继续：`valid = mask != 0`，有效区使用 `mito > 0`；`mask == 0` 全部设 `ignore_index=-1`，包含其中正GT；不扩张mask、不将区外正标签改成背景。CE/Dice只计算有效像素，物理指标继续已有有效表面规则。这与当前MV3实际loader一致，无需更改或重新训练。

本次人工图只读三个已经用于MV3的非test体积neg0/pos0/pos2各一张确定性切片；没有新读取neg1/pos1的test像素。代码逐像素验证当前训练cache的有效mask与原始 `mask!=0` 一致。图不含模型预测，不构成重做完整审计。

## 合并照片

每份左到右：原始灰度图、完整语义GT（mito>0）、原始非零mask显示；上排整张切片，下排相同原生坐标局部放大。绿色为mask内正GT，紫红为mask外正GT，黄色为非零mask；三列同坐标，蓝框表示放大区域。原始整数标签和mask未被编辑，显示只作语义着色。

本地照片在 `E:\mira_website_demo\artifacts\mv3_control_20261014\Han24_review\`，服务器在新MV3目录 `review/Han24/`。原图/GT/mask组件PNG同时保存在服务器；仓库仅保存生成脚本、脱敏统计和本Markdown，不上传原始图像。

生成脚本：`scripts/render_han_review.py`；切片位置、mask编号、原始slice SHA和当前loader一致性见 `evidence/MV-3/Han24_review_receipt.json`。人工复核尚待用户完成，不代替对导出ROI坐标的独立认证。
