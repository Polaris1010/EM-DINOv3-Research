# Research takeover instructions

每个新窗口先读 `START_HERE.md`、`docs/NEW_WINDOW_HANDOFF.md`、最新 `docs/work_logs/` 记录，再读 `docs/PROJECT_STATE.md`、`experiments/registry.yaml`、`docs/NEXT_ACTION.md` 和最新已合并 PR。不能声称尚未发生的提交、推送或合并已经完成。

每轮完成用户授权的代码、配置、报告修改后，同步新增工作记录、更新交接和记录索引；事实变化时更新状态和registry。完成适用CPU检查后，将本轮代码和Markdown一并commit并push，汇报实际commit、分支、PR状态。用户已要求持续保留每次md记录和下一个窗口所需文件；不得只在聊天里总结。

## Evidence and claims

可信顺序：实际 resolved config、checkpoint metadata、源码 commit/SHA 和运行日志；已合并 PR 与实验注册表；研究报告；聊天计划。保留冲突原件，解释采用哪一项记录，不能把旧计划写成完成事实。状态使用 VERIFIED_COMPLETED、VERIFIED_FAILED、PAUSED、PLANNED、UNVERIFIED；已有任务正在运行时另用 VERIFIED_RUNNING，不能把它写成完成。缺失字段填 UNVERIFIED，禁止猜测路径或默认值。

研究目标：开发跨来源通用 2D EM 线粒体分割模型，研究 DINOv3 可迁移自监督表征，在公平协议下与官方 MitoNet 比较。优先复用既有模型和结果。

## Scope and resources

本轮仅接管、只读盘点和初始化事实仓库。禁止新训练、自动实验调度、恢复旧 watcher、修改网站、修改环境包、删除或覆盖历史结果与 checkpoint。物理 GPU 0 永远禁止使用；未来获授权任务只能使用 1/2/3，先核验已有进程和网站状态。不停止未知进程。现在已有 MV-2 在运行，初始化任务不得重新启动它。

扩大 MV-1 验证范围时，只改变验证覆盖。保持 encoder、九个 head、原版 Primus、TTA=OFF、normalization、sliding window、threshold、sampling、checkpoint-selection rule。CMCC/CMCT 的历史四翻转 TTA 不迁移到 MitoVerse。禁止增加 FFT gates、LAST、Gram、HRA、TopoMask 或新模块。

## Research decisions

问题分 BLOCKING、NON_BLOCKING、FUTURE。权重不能安全加载或图像/GT 尺寸不匹配，阻塞相关输入；acquisition 独立性未证实，不阻塞当前 transductive 开发验证，只限制严格泛化结论。B_clean 属于未来独立方案。不无限增加 NO_GO 或把诊断失败改写成成功。

B100k/B150k 是 transductive SSL development assets。无标签暴露不是监督标签泄露；其暴露图像结果不能声称为完全未见图像泛化。三个 decoder seeds 不增加独立生物 acquisition 数量。proofreading 声明与数值 QC 不等于独立重标注完整性认证。

## Repository and PR discipline

只提交必要源码、配置、小型表格、元数据、环境记录和脱敏证据。禁止提交原始图像/Zarr、checkpoint、大量预测、凭据、私钥、敏感 .env、网站配置、运行缓存或大日志。服务器路径只表示资产位置，不保证其他人可访问。远程必须 private；确认账户和创建方式，先复用同目标仓库。

`src/` 是原样历史源码快照，不因新位置就假定可启动。使用前读 `src/README.md`。不重构成熟框架。

每个 PR 说明上一版本实际状态、本次变量、固定配置、实际执行、指标变化和下一项科学决策。PR 不自动合并。更新事实时保存新的带时间戳快照和来源索引，不覆盖已有证据。用 `python scripts/update_registry.py` 从证据生成注册表，随后运行 `python scripts/update_registry.py --check`、`python scripts/validate_repository.py`、`python -m unittest discover -s tests -v`。这些检查只用 CPU，不运行实验。
