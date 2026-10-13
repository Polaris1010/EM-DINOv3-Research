# 官方自然图像权重与 B_full 的冻结评测对照

日期：2026-10-11。encoder 全部冻结，只训练相同的 Primus decoder 与 gates，各 3,000 step，seed=20260907。

| 数据集 | 验证例数 | 官方自然权重宏 Dice | B_full 宏 Dice | B_full−官方（百分点） |
|---|---:|---:|---:|---:|
| Dataset111_CMCC | 3 | 0.7853 | 0.8867 | +10.13 |
| Dataset131_CMCT | 2 | 0.9561 | 0.9615 | +0.53 |

配对控制已验证：decoder 初始参数相同，采样、loss、训练循环、seed、数据 split 相同；B_full 兼容性 batch 输出差为 0。两个 encoder 均严格加载 368 个 backbone keys。完整评测使用同一验证体积、原尺寸恢复、四翻转；验证标签正负 voxel 总数一致。

两个模型各自根据 stride-20 验证 Dice 选 decoder checkpoint，再做完整验证。这是同一协议下的验证集比较；两项任务分别只有 3 例和 2 例，且仅一个 seed，不能据此声称统计显著或未见 source 的泛化改善。SSL 与下游原图体积重叠仍需单独审计。

逐例结果：case_comparison.csv；协议检查与依赖文件 hash：protocol_match.json；原始 control metrics 位于 runs/official_natural_<dataset>/metrics.json。P 保持暂缓，未启动任何 SSL 长训。
