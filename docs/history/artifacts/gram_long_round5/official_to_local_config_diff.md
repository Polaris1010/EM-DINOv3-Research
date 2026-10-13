# 官方配置到本地执行的差异

参考：DINOv3 官方 `dinov3_vit7b16_gram_anchor.yaml`，具体仓库版本与工作区状态见远端 `repository_state.json`。

| 项目 | 官方设置 | 本轮 |
|---|---|---|
| 架构/heads | ViT-7B 对应 heads | 已有 P1 ViT-L 与原 heads；严格加载 |
| Gram 语义 | normalized / image-level / all tokens | 相同 |
| Teacher 输入 | 512，无颜色扰动 | 相同 |
| Student 输入 | global 256，local 112×8 | 相同 |
| λ | 0→2，随后保持 | G 相同；C 恒为 0，保留计算分支 |
| λ 时序 | 1M 起余弦 1000 步 | 映射为本轮第 0–999 步 |
| LR | 3e-5 × 4 × sqrt(global batch / 1024) | batch 16 为 1.5e-5 |
| LR/temperature warmup | 100k 步 | 虚拟 1M 位置已结束，温度 0.07 |
| AdamW / WD / EMA | (0.9,0.99) / 0.04 / 0.999 | 相同 |
| 梯度裁剪 / 分层 LR / patch LR | 30 / 0.98 / 0.2 | 相同 |
| 锚点更新 | 首次 1,010,000，此后每 10k，最多 3 次 | 本地完成 10k/20k/30k 后更新 |
| 初始锚点 | 需提供合适的早期教师 | P1；只检验保护 P1 的假设 |
| 数值执行 | 分布式 BF16 参数、编译 | 单 GPU FP32 master、BF16 autocast、activation checkpoint，compile 关闭 |
| 数据加载 | workers 10、缓存 | workers 4、prefetch 1、不缓存、不常驻 |
| 数据池 | 官方数据 | 过滤后的具名 EM/JRC 数据；详见排除清单 |
| 优化器初始化 | 官方训练状态 | P1 无 optimizer；新建，随后完整恢复 |

G/C 的 resolved config 除 `gram.loss_weight_schedule.end` 外应完全一致。实际输入哈希包含增强后的 tensors 与 masks，不仅是坐标。相同随机状态支持模型内部随机操作的对照；恢复回归对 student/EMA/anchor/optimizer 与采样哈希进行逐张量检查（atol=rtol=1e-6）。

保留官方 loss 调度，包括 local DINO 权重从 1 过渡至 0.5。不得将 Gram λ 与整个 loss 的数值尺度混为一谈。用 Gram loss 的反向 hook 记录真实梯度系数，核对其与记录的 λ 相符。

补充执行修正：SSL 启用 torch 确定性算法，同时关闭 matmul/cuDNN TF32。第 2 次短测的恢复检查发现 AdamW 动量差异约 3e-6，未通过原定 1e-6 容差；因此第 3 次短测重新从 P1 开始，不将失败短测计入正式训练。第 1 次短测仅结果汇总字段重复，修正后 X1 的 100 步梯度、权重更新和输出重载检查已通过。
