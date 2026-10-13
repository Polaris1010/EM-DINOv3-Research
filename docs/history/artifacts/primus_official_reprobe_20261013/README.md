# 原版 MedDINOv3 Primus：P0 / B100k / B150k 重新比较

当前研究的评测入口为 `src/frozen_probe_official.py`，直接导入本目录 `src/official_primus.py`；原版函数逐字保留。该文件对应 MedDINOv3 固定提交 `a2460c2547899283446e520b84523454f7e25608`，SHA256 `b5398dc48bf5cc14068b4fed38c5f0a8c30bcbbe242d59475016e70912f248d3`。

调用官方 `Primus_Multiscale(embed_dim=1024,patch_embed_size=16,num_classes=2,interaction_indices=[5,11,17,23])`，无 `last_gates` / `LaSt_FrequencyGate`。上游 DINOv3 ViT-L/16 backbone 全部冻结，仅训练官方 head，共 44,576,514 参数。

六组比较：P0 / B100k / B150k × CMCC / CMCT。全部 seed=20260907，head 从同一状态新初始化，3000 steps；延用历史 sampler、CE+Dice、学习率、split、验证 head 选择、滑窗与四翻转 TTA。目标仅 GT==1 / prediction==1。只替换 decoder import 和训练 forward；函数 AST 一致性见 build_audit.json。

三模型真实 smoke 通过，新的共同 head 初始化 hash 为 `9a8a1f6ea11dac604574db0133879fb3b6e2e0f0dba9fc6f7f1d0e5275128e69`。完整 encoder state 在 head 更新后不变；原版 forward 在历史 FP16 四翻转推理中输出有限。

官方函数位于独立实验目录，网站仍使用其既有部署代码。服务器共享历史 primus.py、旧 head、预训练权重、历史结果及 manifest 不覆写；今后的本研究原版对照使用此明确入口。

服务器目录：`/opt/data/3_117/tanghr/codex/primus_official_reprobe_20261013`。

结果文件（完成后）：results.json、summary.csv、per_volume_metrics.csv、official_vs_gated_deltas.csv、REPORT_zh.md。SD 为验证体积间样本 SD，不是三个 seeds 的 SD；HD95 为 voxel-equivalent，NSD@1，未校准 nm。

服务器直接抓取因 DNS 失败未完成，随后通过 GitHub connector 成功读取同一提交的原文件。核验 runtime 文件与官方文件的唯一差异是文件末尾多一个 LF；全部代码 AST 与函数一致，详见 github_source_verification.json。官方原始字节另存为 src/official_primus.upstream.py，运行文件未在评测中途改变。

`scripts/run_comparison.py` 只训练新 decoder，不包含 SSL 入口。仅物理 GPU1/2/3，允许已登记网站进程共存；发现未知新 GPU 进程或网站异常时只停止本次自己的 workers。完成后核验受保护输入的前后 SHA 并在 CPU 使用已固定 MedDINOv3 官方测试入口计算三指标。
