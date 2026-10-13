# Initialize research state and experiment registry

Base: `main`（初始scaffold）；head: `chore/bootstrap-research-registry`。此文件为首PR正文；实际PR链接与CI状态见 `docs/GITHUB.md`。

研究状态此前分散在混合网站工作区和服务器timestamp目录，历史计划与已完成记录容易混淆。该PR建立独立研究仓库，用带来源/SHA的精选实际配置、checkpoint metadata、源码和结果表，固定P0、B100k、B150k、原版CMCC/CMCT、MV-1九个head及已有MV-2运行状态。

## 变更范围

- 增加AGENTS、当前状态、数据协议、环境、路径和唯一下一项行动；保存只读服务器健康/环境/Git快照。
- 归档664份本地历史研究文件，包括76份Markdown；逐文件保留字节SHA和来源。增加START_HERE、新窗口交接及每轮工作日志，确保默认main可直接接手完整代码和记录。
- 从真实文件自动生成26条实验记录；缺失字段标UNVERIFIED，运行中单列VERIFIED_RUNNING。
- 保存精确原版Primus及历史trainer/reader/evaluation源码快照，不重构现有框架；源码位置变化的运行边界在src/README明示。
- 明确MV-1/MV-2 TTA=OFF，隔离废弃四翻转MV-2。解释CMCC旧factory registry与完成结果、Stage2A2旧STATUS与后来150k完成记录的冲突。
- 增加CPU-only来源哈希、metadata join、train/dev/test边界、无TTA协议校验及CI；负面测试覆盖encoder错配、head换位、TTA漂移和test混入。

## 固定实验变量与实际执行

本轮无科学实验变量改变；未启动训练、推理或调度，未改网站、环境、旧代码、预测或checkpoint。读取SSH资源状态，CPU安全读取九个head与teacher/student exports、官方权重。现有MV-2任务是本轮接管前已运行的无TTA任务，不是此PR启动。

本轮不产生新的模型指标。已有Dice表保留原decoder、split、seed、TTA和units。B100k/B150k明确为transductive SSL development assets；150k为100k+50k continuation，不是独立从零150k调度。

## 验证

```text
python scripts/update_registry.py --check
python scripts/validate_repository.py
python -m unittest discover -s tests -v
git diff --check
```

实际验证结果见 `docs/VALIDATION.md`。大型权重SHA复用服务器既有记录，未无理由重算；安全读取不支持full optimizer里的numpy状态，本轮不使用不受限pickle读取，该限制不影响现有teacher/head评测。配置、源码和精选text快照按字节SHA验证，权重/数据/全量预测/敏感配置不入库。

## 下一项科学决策

初始化PR完成并确认后，收取现有MV-2无TTA扩展的最终结果，比较B100k与P0在更多来源上的开发表现，分开官方validation与自定义dev。保持九个head与原推理协议，禁止重复训练或重新启动。

## 远程仓库

[Polaris1010/EM-DINOv3-Research](https://github.com/Polaris1010/EM-DINOv3-Research)，private已通过GitHub API核验。账户与创建方式经用户明确确认；owner repositories完整分页查重后创建。代码和记录的最终main交付以GitHub实际状态和回执为准。
