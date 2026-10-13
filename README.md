# EM-DINOv3-Research

跨来源电子显微镜 2D 线粒体分割研究的事实仓库。主模型为 DINOv3 ViT-L/16，当前冻结评测使用原版 MedDINOv3 Primus_Multiscale。GitHub 远程为 [Polaris1010/EM-DINOv3-Research](https://github.com/Polaris1010/EM-DINOv3-Research)，已核验 private；首个 PR 与最新交付状态见 [GITHUB](docs/GITHUB.md)。

新窗口从 [AGENTS.md](AGENTS.md)、[PROJECT_STATE](docs/PROJECT_STATE.md)、[实验注册表](experiments/registry.yaml)、[NEXT_ACTION](docs/NEXT_ACTION.md) 开始。以实际执行记录为准。

2026-10-13 本轮通过 SSH 只读核验服务器、环境、预训练结果和 MV-1 九个 head。发现 MV-2 无 TTA 扩展已在运行，覆盖 188 个 volume/ROI、12 个来源；没有启动任何新训练或推理，没有修改网站。

注册表由证据自动生成，`registry.yaml` 使用 JSON 语法（合法 YAML 1.2），无需安装依赖：

```text
python scripts/update_registry.py --check
python scripts/validate_repository.py
python -m unittest discover -s tests -v
```

本仓库保存配置、源码快照、结果表与元数据；数据、权重、预测仍在服务器。源码运行边界见 [src/README.md](src/README.md)，环境冻结记录属于既有环境盘点，不是可跨主机直接安装的 lockfile。服务器资源和访问条件见 [SERVER_PATHS](docs/SERVER_PATHS.md)。本轮初始化 PR 的审阅材料见 [BOOTSTRAP_PR](docs/BOOTSTRAP_PR.md)。
