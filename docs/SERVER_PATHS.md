# 实际路径与访问条件

服务器为 `root@192.168.203.97`，科研根目录 `/opt/data/3_117/tanghr`。2026-10-13 SSH密码交互认证成功；没有保存凭据。SSH key/agent尝试失败不代表端口不可达。服务器路径是私有资产位置，其他人须自行获得访问权。

| 项目 | Windows资产/快照 | Linux实际路径 | Python | Git/状态 |
| --- | --- | --- | --- | --- |
| 独立研究仓库 | `E:\mira_website_demo\EM-DINOv3-Research` | 尚未新建服务器镜像 | 本地CPU工具 `C:\Python314\python.exe` | 本地main + chore/bootstrap-research-registry；private remote `Polaris1010/EM-DINOv3-Research` |
| 历史SSH/SFTP工具 | `E:\mira_website_demo\scripts\ssh_exec.py`、`sftp_upload_many.py`、`sftp_download_many.py`；依赖 `.deploy-tools` | 不适用 | 本地Python3.14 + `.deploy-tools` | 本轮复用既有SSH工具，未改工具 |
| DINOv3源码 | `artifacts/representation_diagnosis_live_20261013/sources/`等历史快照 | `/opt/data/3_117/tanghr/DINOv3/dinov3` | `/opt/data/3_117/tanghr/dinov3_shared/bin/python` | HEAD 91ffb6803c74aeb6927209b82560142743763dd7；有未跟踪历史文件 |
| MedDINOv3共享框架 | 本地实验源码快照；未复制整个框架 | `/opt/data/3_117/tanghr/Med_DINOv3/MedDINOv3` | `/opt/data/3_117/tanghr/envs/meddinov3/bin/python` | HEAD bd6a650dda4e32d22e913cfc28458df24bb39b83；历史modified文件保留 |
| 原版Primus实际入口 | `src/segmentation/official_primus.py` | `/opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013/src/official_primus.py` | meddinov3 | 上游锁a2460c2547899283446e520b84523454f7e25608；实际SHA b5398dc48bf5cc14068b4fed38c5f0a8c30bcbbe242d59475016e70912f248d3 |
| MitoVerse数据 | 只保存manifest/split metadata，不复制图像 | `/opt/data/3_117/tanghr/MitoVerse_data`，`data`，`catalog.json`，`splits` | 已发现 `envs/mitoverse/bin/python`，实际MV-1 reader使用meddinov3及历史deps路径 | catalog与store counts均212；根目录非Git仓库 |
| B_full100k | `artifacts/mitoverse_full_stage1_final`；新仓库 `experiments/B_full/B100k` | `/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120` | dinov3_shared | VERIFIED_COMPLETED |
| B_full150k | `artifacts/mitoverse_B_full_extend150k_20261012`；新仓库B150k记录 | `/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012` | dinov3_shared | VERIFIED_COMPLETED；100k+50k continuation |
| 原版CMCC/CMCT | `artifacts/primus_official_reprobe_20261013` | `/opt/data/3_117/tanghr/codex/primus_official_reprobe_20261013` | meddinov3 | 六组完成，TTA四翻转；head非MV-1 head |
| MV-1 | `artifacts/mv_generalist_stage1_20261013` | `/opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013` | meddinov3，已完成launcher与实际metric日志支持 | 九个head完成；TTA OFF |
| 既有MV-2正式无TTA任务 | `artifacts/mv_dev_v2_notta_20261013`有旧脚本；本轮新快照见experiments/MV-2 | `/opt/data/3_117/tanghr/codex/mv_dev_v2_notta_20261013` | meddinov3，当前ExecStart和进程核验 | 本轮快照运行中 |
| MV-2废弃四翻转版本 | `artifacts/mv_dev_v2_expansion_20261013` | `/opt/data/3_117/tanghr/codex/mv_dev_v2_expansion_20261013` | meddinov3 | SUPERSEDED；不并入正式结果 |
| Stage2A / Stage2A2 | `artifacts/mitoverse_full_stage2A_result`与stage2a2目录 | `/opt/data/3_117/tanghr/codex/mitoverse_full_stage2_20260930_082816`；`mitoverse_full_stage2a2_20261002` | 本轮不运行旧diagnostic环境 | FAIL / P暂停 / E与负面S pilot完成 |
| 旧31-volume控制 | `E:\mira_website_demo\artifacts\mito_pretrain_v2_control` | `/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227` | 历史执行细节UNVERIFIED | 保留，不恢复旧watcher |
| 来源重叠审计 | `artifacts/data_governance_stage0_20261013`、`mitoverse_round6/overlap_audit.csv`、`mito_pretrain_v2_control/handoff_overlap.py` | 服务器对应审计根目录未逐文件核验 | UNVERIFIED | 本轮登记本地既有CSV，不启动审计或下载CEM |

## 关键checkpoint

P0：`/opt/data/3_117/tanghr/DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`；SHA `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`。

B100k teacher：`/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120/runs/B_full/formal_attempt1/checkpoints/step000100000/teacher_backbone.pth`；SHA `9603877244ba40f3cbeccdea18d69b0d3b818dc0fc6b09f839fd6d4189ce2de5`。同目录full.pth SHA `12bbf09ffcd7f93cbad88c4e55df55d36b3a299990ad976bf29af51a118a265b`。

B150k teacher：`/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012/runs/B_full/formal_attempt1/checkpoints/step000150000/teacher_backbone.pth`；SHA `9c67d08faf508a14d25418014422c0dbd641a51c122161dbd31f8af0a386d2ee`。同目录full.pth SHA `71b71fca5994836c3c6712179874f9c4516fc04584ab09e1df182d613dfed120`。

九个MV-1 head的精确路径、大小、SHA、seed和metadata逐项列在 `experiments/registry.yaml` 与 `experiments/MV-1/checkpoint_inventory.json`。全部位于实际MV-1的runs目录；禁止用CMCC/CMCT head代替。SHA来自本轮读到的服务器既有input_hashes/latest_checkpoint，未重新流式扫描大型权重。

## 资源与网站

四张RTX4090；GPU0禁止。只读快照时网站进程PID1971241在GPU1/2各约458MiB，MV-2既有进程在1/2/3各约2212MiB；GPU0另有未识别分配，本轮不触碰。内存约503GiB；盘和GPU原始只读输出见 `evidence/server_health.txt`。`curl -I http://127.0.0.1:7860/` 本轮两次HTTP200，没有重启网站。

原始盘点输出留在 `E:\mira_website_demo\artifacts\research_bootstrap_20261013`，不提交raw bundle或大日志。新仓库只提交必要快照和来源索引。
