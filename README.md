# EM-DINOv3-Research

当前阶段为用户于2026-10-14授权的 [MV-3多来源监督Generalist训练](docs/MV3_PROTOCOL.md)，分支 `codex/mv3-generalist-training`。仅G-P0/G-B100，冻结encoder+原版Primus，共同40k预算与3 seeds；实际执行状态见 `experiments/MV-3/run_index.json`。MV2已完成，历史不改写。

跨来源电子显微镜 2D 线粒体分割研究的事实仓库。主模型为 DINOv3 ViT-L/16，当前冻结评测使用原版 MedDINOv3 Primus_Multiscale。GitHub 远程为 [Polaris1010/EM-DINOv3-Research](https://github.com/Polaris1010/EM-DINOv3-Research)，已核验 private；首个 PR 与最新交付状态见 [GITHUB](docs/GITHUB.md)。

新窗口从 [START_HERE.md](START_HERE.md) 开始；[交接](docs/NEW_WINDOW_HANDOFF.md)、[每轮记录](docs/work_logs/README.md) 和 [76份历史Markdown](docs/history/INDEX.md) 均随代码提交。以实际执行记录为准。

2026-10-13 本轮通过 SSH 只读核验服务器、环境、预训练结果和 MV-1 九个 head。发现 MV-2 无 TTA 扩展已在运行，覆盖 188 个 volume/ROI、12 个来源；没有启动任何新训练或推理，没有修改网站。

注册表由证据自动生成，`registry.yaml` 使用 JSON 语法（合法 YAML 1.2），无需安装依赖：

```text
python scripts/update_registry.py --check
python scripts/validate_repository.py
python -m unittest discover -s tests -v
```

本仓库保存配置、源码快照、结果表与元数据；数据、权重、预测仍在服务器。源码运行边界见 [src/README.md](src/README.md)，环境冻结记录属于既有环境盘点，不是可跨主机直接安装的 lockfile。服务器资源和访问条件见 [SERVER_PATHS](docs/SERVER_PATHS.md)。本轮初始化 PR 的审阅材料见 [BOOTSTRAP_PR](docs/BOOTSTRAP_PR.md)。
