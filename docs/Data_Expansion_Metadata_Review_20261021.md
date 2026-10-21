# MitoVerse / CEM / IsoOrg / QuantEM 增量来源审查

日期：2026-10-21（香港时间）。状态：metadata review completed；候选未准入训练。

**决定：当前 P0→100k 继续保持 MitoVerse+CEM 设计不变。下一轮若单独扩数据，优先考虑新的组织×制样×成像条件组合；不整包加入 IsoOrg 或 QuantEM。** 本轮筛出15个公开候选入口，其中8个优先、7个条件性候选；没有下载图像、读取GT、修改采样器或启动训练。

## 1. 真正能增加什么

当前CEM并不缺少TEM/STEM/SBF-SEM这些模态名称。627个保留collection中，578个可通过官方来源表Sample UID精确连接；对应FIB-SEM 413、TEM 110、SBF-SEM 28、ssTEM 9、STEM 7、SEM 6、AT 4、ET 1，另49个来源元数据未匹配。这是collection数，不是独立acquisition数或有效训练比例。不能把新增TEM称为“第一次引入TEM”。

更有价值的补充是：P7发育期心脏/皮肤/胸腺/胰腺、鱼类发育视网膜、实体连续切片睾丸、塑料界面贴壁细胞SBF、非OpenOrganelle的人肝/肾/血管内皮，以及免疫金标脂肪细胞和工程化人心肌TEM。这些是metadata支持的新增采集条件；能否改善线粒体泛化仍是待检验假设，不保证解决Glycolytic muscle误检。

这次没有逐图测量结构差异，也不能将所有候选称为“已证实独立外部数据”。49个CEM collection缺来源映射，TEM-100别名和部分volume的原始采集身份也不完整。可确认的是不同命名采集/公开资产入口及其条件，而非精确新增的独立生物样本总数。

## 2. 输入与证据边界

- 当前SSL池：`experiments/D12/mitoverse_pool.json`，153 volumes、34,143合法Z；`cem_pool.json`，1,425,433图、627 collections、13个排除集合。原始212-volume MV catalog只用于历史与留出来源排查，不恢复旧训练池。
- 历史IsoOrg组文件：928个唯一子体积，40个可解析scan名称；原审计文件与最终CEM目录/官方UID表一并复用，不继承旧Stage0的训练禁入结论。
- QuantEM官方`quantem-base-model-sources/sources.csv`：本次下载实数657行，502 Public、117 Outreach、38 Internal；654种Dataset字符串、514种URL字符串。行数、名称数、DOI数均不是独立acquisition数。
- QuantEM revision：`99db73cef7ab1b28c46eaa166f01aa902136b23e`，CSV SHA256 `6688bedd0f69f6d7620b079e7ee1a4f675967525ed557d897e37188bd2471194`。权重卡提到的655不替代实际来源CSV的657行。
- 新增官方EMPIAR、Janelia/Figshare、Zenodo JSON，只读取metadata及目录；没有下载zip/tiff/zarr图像块。路径、SHA、字节数和访问回执均保存。

证据层级：**同命名acquisition** / **同DOI条目（可能包含多个采集）** / **同研究或benchmark家族** / **ROI关系未核实** / **像素重复未测试**。版本DOI去掉末尾`.vN`仅用于连接同一个发布条目，不断言版本间像素相同；`jrc_hela-2`与`jrc_hela-21`、`jrc_cos7-1a`与`jrc_cos7-11`不合并。

## 3. 四套数据的来源重叠

下表是已解析命名scan的交集下界，不是穷尽的跨库像素重叠数。QuantEM中只给DOI的条目，通过官方源metadata或精确CEM DOI映射补充scan名；未解析者仍未知。

| 数据组合 | 当前可连接的共同scan数 | 解释 |
|---|---:|---|
| 当前MV × 当前CEM | 6 | ctl-id8-1、hela-2、hela-3、jurkat-1、macrophage-2、sum159-1 |
| 当前MV × IsoOrg | 8 | 上述6个，加mus-kidney和mus-liver；完整历史MV为9个，另含cos7-1a |
| 当前CEM × IsoOrg | 18 | 最终CEM目录与UID metadata支持；完整CEM为19个，另含本轮池已排除的cos7-11 |
| 当前MV × QuantEM | 至少11 | 包含HeLa/Jurkat/Macro、肾/肝及肝分区等 |
| 当前CEM × QuantEM | 至少17 | 大量既有OpenOrganelle来源；另36个QuantEM行有CEM保留来源DOI匹配，不能把36行全计为独立scan |
| IsoOrg × QuantEM | 至少30 | 多数共同OpenOrganelle扫描；不能两次计为新增来源 |

所有上述跨库关系的ROI交叉范围均为UNVERIFIED，像素重复为NOT_TESTED。相同acquisition中非重叠区域可能有使用价值，但不增加独立采集来源数；不能仅凭名字相同就断言全部像素重复。

**对历史622/104数字的更新：** 928个IsoOrg子体积中，历史MV之外为622个；其中104个对应的13个额外scan，后续历史审计已有最终CEM目录及官方来源表支持，已不只是下载候选。扣除这些已知同源名称后，剩下**518个子体积、18个scan名称候选**。这18个也不是18个已经证明完全独立的新采集：例如pancreas-2/3的官方描述引用Mueller 2020胰岛研究，与现有Müller来源存在研究家族关联，需要区别具体采集。

若只与当前而非历史数据池比较，会得到20个名称未命中；其中cos7-1a是历史MV/开发来源，cos7-11被CEM排除政策移除，不能因此把它们当“新来源”重新引入。

## 4. 推荐的15个公开候选入口

A表示优先的下一轮候选；B表示价值存在但读入格式/粒度/分辨率仍有限制。排序依据来源条件与可访问性，未读取分割GT或用benchmark成绩挑选图像。15个入口包括12个3D资产/系列和3个TEM研究系列，不能写成15个独立acquisition。四个IsoOrg候选也出现在QuantEM中，以下只计一次。

| 优先级 | 原始入口 / acquisition | 可补充的条件 | spacing XYZ nm及主要限制 |
|---|---|---|---|
| A | [jrc_mus-heart-1](https://doi.org/10.25378/janelia.24020919) | P7小鼠心脏FIB-SEM | 8/8/8；一个采集，非40个独立IsoOrg来源 |
| A | [jrc_mus-skin-1](https://doi.org/10.25378/janelia.24085269) | P7皮肤、细胞外基质 | 8/8/8；不同组织是否同一动物未知 |
| A | [jrc_mus-thymus-1](https://doi.org/10.25378/janelia.24085287) | P7免疫组织、致密细胞 | 8/8/8；同一P7项目的相关性保留 |
| A | [jrc_mus-pancreas-4](https://doi.org/10.25378/janelia.23411843) | P7胰腺、不同制样 | 8/8/8；与成年低糖胰岛pancreas-2/3区分 |
| A | [EMPIAR-11759](https://www.ebi.ac.uk/empiar/EMPIAR-11759/) | 55hpf斑马鱼视网膜SBF | 8/8/50，官方details明示；只列16幅DM3 |
| A | [EMPIAR-12585 Scan1](https://www.ebi.ac.uk/empiar/EMPIAR-12585/) | 鳀鱼视网膜SBF | 25/25/50，details明示；尺度较粗 |
| A | [EMPIAR-12157](https://www.ebi.ac.uk/empiar/EMPIAR-12157/) | 小鼠睾丸AutoCuts/ATUM-SEM | 5/5/40，details明示；切片对齐未核实 |
| A | [Zenodo 11198586 Dataset04](https://doi.org/10.5281/zenodo.11198586) | 贴壁HeLa、塑料界面SBF | 10/10/50；raw/processed/video是同源派生版本 |
| B | [EMPIAR-13356](https://www.ebi.ac.uk/empiar/EMPIAR-13356/) | 人肝组织多尺度SBF | spacing未知；只考虑raw_images，不读取伴随标签 |
| B | [EMPIAR-12674](https://www.ebi.ac.uk/empiar/EMPIAR-12674/) | 非OpenOrganelle小鼠肾FIB | API字段30/30无显式单位/Z，本轮不填nm值 |
| B | [EMPIAR-11958 EM子目录](https://www.ebi.ac.uk/empiar/EMPIAR-11958/) | 提睾肌内血管内皮SBF | API字段语义需澄清；输出合并2次采集，排除X-ray数据 |
| B | [EMPIAR-12174](https://www.ebi.ac.uk/empiar/EMPIAR-12174/) | 大鼠胰腺FAST-EM、钕染色 | spacing未定；43切片×16拼图，不是43独立采集 |
| B | [Zenodo 13880353](https://doi.org/10.5281/zenodo.13880353) | CAV1KO脂肪细胞免疫金TEM | spacing未知；17图的独立采集数未披露 |
| B | [Zenodo 17137494 TEM子集](https://doi.org/10.5281/zenodo.17137494) | 心脏微血管I/R及H/R病理条件 | spacing未知；混合实验包中只有TEM部分适用 |
| B | [Zenodo 5056166](https://doi.org/10.5281/zenodo.5056166) | 机械负荷下工程化人心肌TEM | spacing未知；一个研究系列，独立样本数未知 |

四个A类Janelia候选的公开S3 `.zattrs`返回200；七个EMPIAR候选的公开data目录返回200；四个Zenodo候选记录为open并给出下载文件链接。仅证明公开入口可达，不冒充本地图像已下载、解码或已获得使用许可的完整复核。候选清单另保存原始许可字段。

备选`jrc_mus-kidney-2`的原始metadata明确有8nm体素和独立命名ID，但本次所测试的S3属性路径404，因此暂不列入15项主推荐；不据此宣称数据已下架。

## 5. 哪些暂不作为新增来源

- 已有六个共同OpenOrganelle scan及CEM中的HeLa系列、ctl-id8系列、pancreas-1：来源重复；没有ROI证据就不把它们包装成独立多样性增加。
- QuantEM的MitoEM 2.0条目和MICrONS系列：分别是benchmark/研究家族级关系，不断言每个资产都相同；已有MV训练与开发来源关联，当前不优先引入。
- IsoOrg pancreas-2/3：官方源metadata引用Mueller胰岛研究，与Müller开发来源存在关联风险；不在本轮推荐中。P7 pancreas-4有不同原始ID/样本年龄，单独记录。
- 同一EMPIAR/Zenodo条目的多切片、多版本、多个ROI：可以提供视野变化，不增加同等数量的独立来源。
- QuantEM Outreach/Internal条目：来源表存在不等于每个原始图像包已具备公开可读入口；本轮主推荐限于公开入口已核实的条目。

## 6. 下一轮是否值得引入

**值得单独做，但当前不实施。** 如果之后决定扩数据，优先从上表A类选择互补条件，按原始acquisition控制份额；TEM条件性候选适合补充制样与纹理差异，但需将图像系列/动物/病理状态分开记录。此建议不以增加总图像数为目标，也不将18个未匹配名称或15个推荐入口称为严格独立测试集。

本轮停止于来源登记。P0设计、采样器、原实验配置、checkpoint、注册表、网站均不变；不自动下载、训练、增加loss或开启额外审核阶段。需要最终独立测试时，从未参与方案选择的采集另行留出，不能把本轮用来选数据的来源自动作为盲测证明。

## 交付与复现

- `../evidence/Data-Expansion-20261021/recommended_candidates.csv`：15项候选，原始URL、组织/模态、spacing、可访问性、ROI/独立性限制。
- `four_dataset_overlap_summary.csv`、`quantem_overlap_all_rows.csv`、`isoorg_overlap_all_scans.csv`：6组关系、657行QuantEM与40个IsoOrg名称的全表。
- `current_cem_collections.csv`、`mitoverse_membership.csv`：当前627集合及历史212-volume成员关系。
- `input_receipts.json`、`metadata_receipts.json`、`access_checks.json`：来源快照、SHA及公开入口回执。
- `python scripts/review_data_expansion_metadata.py`：无网络、无图像读取的metadata重放，校验冻结输入SHA并生成交叉表。

原始资料：[QuantEM官方来源CSV](https://huggingface.co/datasets/ArrojoeDrigoLab/quantem-base-model-sources/blob/99db73cef7ab1b28c46eaa166f01aa902136b23e/sources.csv)、[OrgMIM分组文件](https://github.com/yanchaoz/OrgMIM/blob/9b0477098f713f3606306aaf9b5dcde625eb69c7/orgmim/utils/group_file.json)、[CEM官方来源工作表](https://ftp.ebi.ac.uk/empiar/world_availability/11037/data/metadata/cem_mitolab_metadata.xlsx)。候选的原始发布链接见表，未将第三方概述当作采集身份凭据。
