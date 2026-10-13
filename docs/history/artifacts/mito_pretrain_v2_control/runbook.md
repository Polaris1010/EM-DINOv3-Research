# MitoVerse LAST × TopoCL v2 执行接管说明

服务器运行目录：`/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227`。用户授权的正式执行已于 2026-09-26 启动。所有新产物只写入此目录；GPU0 禁止分配。

## 当前服务与批次

- `mito-pretrain-v2-seed0.service`：物理 GPU1=B、GPU2=L、GPU3=T，各 100,000 次 optimizer 更新；第一张卡释放后自动接手 LT。单个 `Guard` 每 5 秒核对 GPU 进程、每 10 秒核对资源和网站。资源/未知进程错误生成 `src/STOP.json` 并只终止本轮注册进程组。
- `mito-pretrain-v2-watcher.service`：CPU 依赖观察器，每 60 秒刷新 `stage_status.json`；正式 SSL 成功退出后启动冻结 probe，随后匹配微调，再按 validation 阈值启动独立 SSL seed1、条件 2.5D 和最终语义评估。每个下游阶段在 GPU 上先做 10 步 smoke，失败即停，不无限重试。
- 实时阶段文件：`stage_status.json`；详细队列状态：`src/formal_seed0_status.json`、后续各阶段 `*_status.json`；逐任务进度：`src/jobs/<job>/progress.json`、`steps.jsonl`、`hourly_summary.jsonl`。完整状态每 1000 更新保存，EMA 10k/30k/60k/100k 不可变导出。正式 SSL 只保留最近两个及固定里程碑完整 checkpoint。

## 只读监控

在服务器运行：

```bash
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
systemctl show mito-pretrain-v2-seed0 mito-pretrain-v2-watcher -p ActiveState -p MainPID -p Result --no-pager
cat "$run/stage_status.json"
cat "$run/src/formal_seed0_status.json"
cat "$run/src/jobs/formal_seed0_B_part0/progress.json"
test ! -e "$run/src/STOP.json"
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7860/
nvidia-smi --query-gpu=index,uuid,memory.free,temperature.gpu --format=csv,noheader,nounits
```

本地只读 SSH 脚本：`remote_all_status_compact.sh`、`remote_formal_status_compact.sh`、`remote_checkpoint_status.sh`。SSH 凭据不写入脚本或运行目录。

## 审查与恢复门槛

`evidence/formal_seed0_admission.json` 锁定了 batch16、四组短测、100 步稳定性、数值恢复和方法/数据哈希。四组样本链严格相同；恢复权重不是位级相同，最大差异约 1–2×10⁻⁵，低于同机两次连续运行的浮点差异。早期 BF16 总损失读数粗，1000 步约由 16.625 降至 16.5；继续按分项、梯度、特征方差和参数位移监控，不据 10k/30k 分数淘汰任务。

如服务异常，先核对 `STOP.json`、systemd 日志、`src/process_registry.json` 与 GPU UUID、网站状态及 checkpoint 收据。**不要在未查明资源冲突前清除 STOP 或重派任务。** 正式 SSL 协调器能从已核哈希的最近 checkpoint 创建新的 `partN`；系统重启后的 transient service 需要显式检查并重新启动。不得通过复制 seed0 checkpoint 改随机种子来代替独立 SSL seed1。

## 结果边界

## 2026-09-28 并行调度接管

用户要求两项都做、先评测新权重。原 B/L/T 已完成 100k；LT 在 96,140 步收到 SIGUSR1 后原子保存完整 checkpoint，核对哈希后接管为 `coordinator_overlap.py`。原有停止记录已归档到服务器 `evidence/overlap_handoff_*`，没有丢弃检查点或改动训练预算。服务仍叫 `mito-pretrain-v2-seed0`，watcher 已恢复。

- GPU1 恢复 LT；GPU2=N0、GPU3=B，先通过 10 步 smoke，再执行各 10k 冻结编码器分割头训练及全体积 validation。L/T/LT 的 100k checkpoint head0 随卡释放派发。
- `overlap_status.json` 是本阶段实时评测状态；普通 `probes_status.json` 尚未出现时，旧 `stage_status.json` 可能仍显示评测 pending，不能据此判定 GPU 空闲。
- 五个最终检查点 head0 探针完成后 overlap 服务退出，watcher 自动启动普通 probes。已有完成结果复用。`coordinator_probes.py` 在 GPU1/2 继续其余里程碑和 head seeds；GPU3 运行 `expanded_queue.py`。
- 扩展对比已完成本地清单 212 体积的目录/尺寸/中心切片可读性检查，复用旧 24 体积，补算 188 体积 ×（MitoNet + P1 三个 head seeds）。这是来源描述性扩展；31 个训练体积和 157 个历史重叠风险体积单列。han24 五个体积使用暂定 mask!=0 语义，不能混入确定标注口径的总体分数。
- 扩展 worker 使用原 `meddinov3` 环境：共享 SSL 环境缺少旧 Primus 的 einops。仅对未来 guard 的 `expanded_eval.py` 解释器选择增加分支；正在运行的 SSL 和新模型探针不受影响。
- MitoNet batch2 CPU 兼容性已验证，最大 logit 差 2.86e-6；完整 GPU smoke 在扩展批次启动时执行，检查与旧固定推理的概率差。P1 初始推理 bs8，MitoNet bs2，OOM 时减半并记录。正式 SSL batch16 和新模型探针 batch8 不变。
- 2026-09-28 21:11 北京时间：LT 96,800；N0/B 探针 1,025/10,000。首次 step1000 全体积 validation 源等权 Dice 分别为 0.1433223 / 0.1714874，仅为早期单 head seed 结果。

新增只读状态入口：`remote_dispatch_final_status.sh`、`remote_overlap_status.sh`。扩展进度见 `src/expanded_status.json`，完成结果见 `evidence/expanded_comparison_summary.json` 和 `expanded_comparison_volume.csv`。

## 结果解释

训练数据原始 specimen/scan/ROI 映射不完整；legacy_dev_test 曾在历史工作中查看，只是开发测试。MitoNet 采用归档的官方零样本固定协议，不能把与监督微调模型的分数差异解释成等数据量方法排名。完整 2D 实例指标未实现时只交付语义结果，不编造 PQ/AP。最终结果待依赖批次真实完成后生成于 `evidence/final_semantic_report.json` 和 `next_decision.md`。
