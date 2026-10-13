import csv
import hashlib
import json
import sys
from pathlib import Path

out=Path(__file__).resolve().parent
sys.path.insert(0,str(out/'vendor'))
import pyarrow.parquet as pq

def read(name):return json.loads((out/name).read_text(encoding='utf-8-sig'))
facts=read('audit_summary.json');lab=read('mitolab_archive_audit.json');probe=read('mitolab_duplicate_probe_evidence.json')
mv_revision='d0af9a6fd11faaf16ce0822233b7bbf72553ce7d'
report=f'''# Stage 0 — Data Governance & Protocol Lock：NO_GO

日期：2026-10-13。研究计划采用用户 V3.0。**本轮完成可获取证据的审计交付；严格数据成员冻结未完成，B_clean 不具备启动条件，也未启动。**

## 证据支持的结果

- 历史 B_full 全量采样池：212 volumes、13 sources、48,722 Z slices。保留原模型；报告使用 B_allMV_transductive_100k/150k。未读取测试标签作为 SSL loss 的证据来自既有训练实现与日志；这不是对每个 optimizer step 的独立重放验证。全量池资格与每张切片被实际抽到不是同一事实。
- 固定 MitoVerse HF revision `{mv_revision}`，当前官方 catalog 与本地历史清单的 212 volume ID 完全对应。OpenOrganelle 18 个官方命名 scan；IsoOrg 官方 group 文件 928 唯一样本标识、40 个名称解析候选 scan，均不能直接等同独立物理 acquisition 数。
- 对 CEM1.5M 最终 ZIP 读取中央目录，共 1,593,405 ZIP entries：1,592,753 TIFF 和 652 directory entries（含根目录）。651 来源目录中 640 个有图像、11 个空目录。没有完整下载图像归档；保存逐图名称、CRC、压缩大小与 byte offset 供选择性核验。
- CEM-MitoLab 官方 metadata 工作表有 605 个来源记录。按 UID 精确对应 CEM 最终 592/651 个目录；剩余 59 个目录未能对应（含部分空目录），不允许用相近名称自动补齐。不声称此工作表与论文 Supplementary File 1 是同一文件。
- 6 个共同命名 scan 对应 MitoVerse 33 volume 和 IsoOrg 155 样本；另 13 个 CEM/IsoOrg scan 对应 104 样本。上述 OpenOrganelle 命名来源已由最终 CEM 目录及官方来源表进一步支持，而非只停留在下载候选配置；真实物理 ROI/像素重叠范围仍未确定。
- CEM-MitoLab 最终归档实际有 {lab['images']:,} 张图像和同数量标签，{lab['source_directories']} 个含图像目录，按文件名逐一配对。与官方声明 21,860 对差 12 对。保留实际清单，不擅自删去 12 个资产。隐藏 Apple TIFF 资源文件计数为 {lab['apple_resource_fork_tiff_entries_excluded']}，差异不能归因于这类文件。
- 20 个 CRC+文件大小重复候选组（40 图）；CRC 本身不确证重复。从两个不同随机来源目录各抽取原图和标签，证实一对原图文件与解码 SHA256 都完全相同，但标签前景有 {probe['label_semantic_disagreement_pixels']:,}/{probe['total_image_pixels']:,} 像素不同（约 7.66%）。这是 L0 原图重复及标签分歧证据，尚不判定任一标签错误或推断原始 acquisition 身份。其余 19 组仍是候选。
- MitoNet 官方目录检索取得 6 个 vEM benchmark 目录和 100 行 TEM 来源 metadata；5 个 conrad23 volume 的 catalog 明指 EMPIAR-10982，不能对 B_allMV 声称未见。其他目录/TEM 是否满足完整训练暴露隔离仍须审计。
- 已有 GT crop 诊断为 424 个样本、1,696 个候选，覆盖全部 212 volume。这是 volume 包含被诊断 ROI 的证据，不表示整卷 GT 均被查看；已记录方法开发/GT 诊断曝光。开发曝光不会因重训 B_clean 被清零。

## 关键协议问题

1. CellMap overlay 的 147 crop 暂全部 train，文件明确说明官方 split 尚待应用，不能直接准入。
2. ME2-Jurkat/Macro/Podo 的 test0/train 与 test1/test 属于同名 scan；严格 acquisition 测试须从所有训练池排除整 scan。常规官方 region 测试另表报告。
3. ME2-Pyra 仅记录 wei20_mitoEM-H 为 train 且备注 held-out test region；未给出实际坐标。示例 README 区间不是该 benchmark 的经核验坐标，不得照抄。
4. Han24 mask 极性、其他来源标注完整性、连通实例语义和 ignore 处理未获得足够证据。除所抽 CEM-MitoLab 冲突配对外，本轮没有原始标签像素全面审计。
5. 历史 preliminary manifest hash 为 e25c88…，实际 formal 记录为 3fd614…；preliminary 的 source-uniform 采样被后续 uniform-legal-slice 正式运行取代。保留版本差异，不混合两份配置。正式 manifest 原始字节和 150k full checkpoint 的最终 hash 尚未取得；teacher export 的既有 hash 已另记。
6. 历史 NSD/HD95 使用 spacing=(1,1,1)，只能按 voxel/header units 解读；跨来源物理距离指标要求核验真实 nm spacing 和 resize 变换。

## 启动阻断项与解锁证据

| 阻断项 | 解锁所需证据 |
|---|---|
| 原始 acquisition/ROI 身份未形成完整映射 | 官方原始 scan ID、ROI 起点/形状、轴方向/缩放/重采样变换及数据主机元数据 |
| 最终 untouched test/external 成员未确定 | 避开已诊断区域的独立 acquisition；固定测试资产与训练排除表；跨模型历史曝光核验 |
| CEM 59 目录缺 metadata UID 精确映射 | 论文 Supplementary File 1 或归档来源说明；人工有证据的别名映射 |
| CEM-MitoLab 计数和重复标签分歧 | 核验差异资产、同图多标注语义和版本，记录隔离/仲裁政策 |
| 标注有效区未证实 | 原始转换脚本/标注说明、mask 极性和 pixel/instance 审计 |
| 训练与 evaluator resolved config 未正式冻结 | 从成功正式运行取得完整配置，冻结 decoder、NSD 容差、label budgets、seed 和后处理 |
| 历史哈希原件不完整 | 正式 manifest、100k/150k checkpoint/export 的原始 hash receipt 与日志 |

本机只有服务器路径和快照；原始 Zarr 位于 `/opt/data/3_117/tanghr/MitoVerse_data`，本轮未获得该主机的可用 SSH 连接。已提供只读 metadata collector。Supplementary File 1 的公开请求返回访问挑战 HTML，响应已保存；没有把 HTML 伪装成 XLSX。取得原始数据连接及缺失来源证据后继续核验，不能由超时或缺回复自动准入。

## 交付物的实际状态

- `master_acquisition_registry.parquet`：{facts['registry_rows']:,} 个不同粒度资产记录，另有 CSV 镜像。该行数不是 acquisition 数，也不是新增独立图像数。未知身份保持 null；CEM-MitoLab 为图像/标签配对，CEM 为来源目录，IsoOrg 为样本标识。
- `cross_dataset_overlap.csv`：区分官方声明、同名来源候选、最终目录证据与真正 SHA256 确证。只有抽取的一对原图确证 L0。
- `mitoverse_clean_splits.json`：保留所有 official 候选角色及阻断原因，全部 212 volume 目前 quarantine；train/dev/test/external 均未获准，test_freeze=false。
- `ssl_train_manifest.json`：entries=[]、training_permitted=false、status=NO_GO；不是可训练数据清单。
- `label_validity_audit.csv`：逐 MitoVerse volume 的声明/历史 mask 状态、未证实项及被抽 CEM-MitoLab 冲突标注。
- `data_exposure_matrix.csv`：现有/未来模型的 SSL、监督未知项、方法开发及 GT ROI 暴露；未来模型未训练，开发曝光按项目历史保留。
- `benchmark_protocol_v1.md`：Protocol R/S 规则、计算预算、seed、评价单位、置信区间与最终测试规则；明确哪些配置和数据成员尚未锁定。
- 本报告、source snapshots、hash receipts、CEM 逐图 inventory、MitoLab 配对清单、model aliases、region loader 与辅助审计工具。

region loader 和图像重叠检索共 9 个测试通过，覆盖同 acquisition 跨 split 拒绝、同卷区域相邻/重叠判定、跨 crop 坐标拒绝、越界裁剪拒绝、SSL 不读标签、mask 极性、旋转/翻转/裁剪/灰度反转以及恒定图像无匹配。测试使用可控合成数组，不等于真实 benchmark 已通过 loader 验证。

## 原始来源

[固定 MitoVerse 发布包](https://huggingface.co/datasets/pytc/MitoVerse/tree/{mv_revision})；[CEM1.5M 官方归档](https://www.ebi.ac.uk/empiar/EMPIAR-11035/)；[CEM-MitoLab 官方归档](https://www.ebi.ac.uk/empiar/EMPIAR-11037/)；[官方来源工作表](https://ftp.ebi.ac.uk/empiar/world_availability/11037/data/metadata/cem_mitolab_metadata.xlsx)；[MitoNet benchmark](https://www.ebi.ac.uk/empiar/EMPIAR-10982/)；[MitoNet 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9883049/)；[OrgMIM 官方 group 文件](https://github.com/yanchaoz/OrgMIM/blob/9b0477098f713f3606306aaf9b5dcde625eb69c7/orgmim/utils/group_file.json)。
'''
(out/'go_no_go_report.md').write_text(report,encoding='utf-8')
required=['master_acquisition_registry.parquet','cross_dataset_overlap.csv','mitoverse_clean_splits.json',
          'ssl_train_manifest.json','label_validity_audit.csv','data_exposure_matrix.csv','benchmark_protocol_v1.md','go_no_go_report.md']
for name in required:
    if not (out/name).is_file() or (out/name).stat().st_size==0:raise ValueError(f'Missing delivery: {name}')
table=pq.read_table(out/'master_acquisition_registry.parquet')
if table.num_rows!=facts['registry_rows']:raise ValueError('Parquet row reconciliation failed')
if len(set(table['asset_id'].to_pylist()))!=table.num_rows:raise ValueError('Duplicate asset IDs')
if set(table['admitted_role'].to_pylist())!={'quarantine'}:raise ValueError('Unexpected admission')
manifest=read('ssl_train_manifest.json');splits=read('mitoverse_clean_splits.json')
if manifest['entries'] or manifest['training_permitted'] or splits['test_freeze']:raise ValueError('NO_GO was bypassed')
with (out/'data_exposure_matrix.csv').open(encoding='utf-8-sig') as f:exposures=list(csv.DictReader(f))
for model in ['B_allMV_transductive_100k','B_allMV_transductive_150k']:
    r=[x for x in exposures if x['model']==model and x['asset_id'].startswith('MV:')]
    if len(r)!=212 or any(x['unseen_image_claim_allowed']!='False' for x in r):raise ValueError('Exposure completeness failed')
checks=dict(status='pass',registry_rows=table.num_rows,unique_asset_ids=True,
            delivery_files=len(required),NO_GO_enforced=True,B_allMV_exposure_volume_rows_each=212,
            real_pixel_pair_verified=probe['raw_decoded_pixels_equal'],tests_passed=9,
            full_protocol_lock=False,training_started=False)
(out/'delivery_validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
hashes={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in required}
(out/'delivery_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
