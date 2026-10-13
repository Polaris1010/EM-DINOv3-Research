# Stage 0 审计交付

入口：`go_no_go_report.md`。当前 **NO_GO**，数据成员未正式冻结，未启动训练。八个指定交付物位于本目录；`delivery_validation.json` 和 `delivery_sha256.json` 记录交付核验。

## 文件口径

Parquet registry 为混合粒度资产索引：MitoVerse volume、IsoOrg 样本标识、CEM 来源目录、CEM-MitoLab 原图/标签配对、官方来源 metadata 和 MitoNet benchmark 候选。`asset_id` 是唯一索引；真实 acquisition_id 未证实则 null，候选命名关联放在 acquisition_key_candidate，不能据此报告独立 acquisition 数。

空间属性 `shape_zyx`、`voxel_size_zyx`、`metadata_json` 是 JSON 字符串；单位、轴、坐标状态独立列存储。未知字段不填假值。`master_acquisition_registry.csv` 是可读镜像。image_path/label_path 可以是未访问的服务器数组路径或 `archive.zip::member`，须结合 path_status 使用。

`cem_final_image_inventory.csv.gz` 保存全部 CEM 最终图像路径及 ZIP 元数据，CRC 不是去重充分证据。`mitolab_duplicate_candidates.csv` 仅候选，已核验一对的证据在 `mitolab_duplicate_probe_evidence.json`。`sources/` 保存按版本固定的官方原件、范围请求 receipt 及有选择读取的原图/标签。

## 复现

Python 依赖：numpy、Pillow、pyarrow、PyYAML、openpyxl。额外依赖安装在本目录 vendor，不修改项目应用依赖。

```powershell
python -m unittest discover -s artifacts\data_governance_stage0_20261013 -p test_*.py -v
python artifacts\data_governance_stage0_20261013\assemble_cem_directory.py
python artifacts\data_governance_stage0_20261013\prepare_duplicate_probe.py
python artifacts\data_governance_stage0_20261013\decode_duplicate_probe.py
python artifacts\data_governance_stage0_20261013\build_governance.py
python artifacts\data_governance_stage0_20261013\finalize_delivery.py
```

以上使用已保存证据，仅本地重建。重新下载须固定原 URL/版本，检查范围 receipt、ETag、字节长度和 SHA256；禁止将新版本与本目录旧证据静默混合。`fetch_cem_directory.ps1` 仅下载 ZIP 中央目录，不下载图像；`fetch_range.ps1` 拒绝未遵守范围的完整归档响应。失败的完整目录请求残片保留为调查证据，不参与解析。

数据主机上只读收集 metadata（示例，尚未执行）：

```bash
python collect_remote_metadata.py --root /opt/data/3_117/tanghr/MitoVerse_data --out /path/to/audit/remote_metadata.json
```

高风险原图配对核验工具读取 CSV 的 image_a/image_b，产生文件/解码 hash、pHash、翻转/旋转、多尺度裁剪和反转 NCC 检索证据，不自动认定 acquisition 身份。选取的文件应是可处理的单幅 EM 图像或已审计导出的切片。

```bash
python overlap_image_evidence.py --pairs candidate_pairs.csv --out overlap_evidence.json
```

`region_loader.py` 是可注入只读 Zarr opener 的区域读取核心。它尚未集成远程训练 loader，且真实 benchmark 未通过端到端验证。禁止在其上游用旧 split 或缺 ROI 的 volume 覆盖生成可训练 manifest。当前 NO_GO manifest 无法加载；即使治理通过，本轮也不授权自动启动 B_clean。
