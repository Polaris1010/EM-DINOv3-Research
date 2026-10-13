# MitoVerse round4 执行记录

用户已批准开始训练。服务器实验目录：
`/opt/data/3_117/tanghr/codex/mitoverse_round4_20260909_103239/`

本轮流程：来源划分及权重锁定 → 三组100步smoke → N1/N2各500步统一预处理试验 → P0/P1/P2各三个seed、每次10k步 → 冻结评测配置 → 完整体积验证和测试 → 报告后暂停。

- GPU 1=P0，GPU 2=P1，GPU 3=P2，每卡独立进程；GPU0不使用。
- batch固定8，crop256，BF16冻结特征，FP32分割头；本轮不尝试16，避免改变统一预算。
- head LR沿用round3冻结probe实际配置1e-4，而非后续微调建议3e-5。AdamW beta=(0.9,0.98)、WD0.05、clip12、50步warmup及10k cosine。
- P2固定为round3的batch2/lambda0/较小LR/2000步checkpoint。
- 31训练体积：han24、muller21、kunduri22、turner22。来源等权、来源内体积采样；50%按已采样到的训练前景坐标采样。
- 8验证体积：guay21、mekuc20。
- 16测试体积：test1=wei20、jiang25、casser20；test2=xiao18、haberl18；test3=conrad23。各来源全部归到同一split。
- OpenOrganelle全部排除，避免与P2选择使用过的CMCC/CMCT重叠。
- 严格SSL未见来源尚未证实。当前EM缓存包含655916个图像路径，不能替代不可变的历史采样日志；不确定项保留uncertain。
- P0转换把qkv bias mask乘入bias后移除mask，并移除storage tokens；其输出与原自然架构不等价，解释P1-P0时必须保留这一限制。
- 相同seed共用head/gates权重文件与80000条patch及增强schedule；运行中断言样本序号、无encoder梯度、所有head梯度有限，并验证保存/重载输出完全一致。
- 每1000步使用每验证体积最多8张固定切片的完整XY滑窗选头。最终指标评测全部原始体素；选择曲线不是最终成绩。
- TTA关闭、阈值0.5在评测前固定，不用test调参。滑窗256、步长0.5、Gaussian融合，保存float16概率和uint8二值Zarr。
- 统一资源guard每30秒记录GPU/进程树RSS/内存/swap/IO及网站HTTP健康。未知GPU进程或资源阈值触发时仅停止本轮注册进程组；网站服务保留。
- 不自动延长10k步、不启动Gram/边界/2.5D/实例分割或DDP。

运行状态在服务器status.json；训练输出位于jobs/<stage>_<P0|P1|P2>_seed<0|1|2>_<N1|N2>/train.log。

## 已验证的启动状态

三组100-step smoke全部通过，初始化与patch schedule哈希一致，重载输出完全相同，退出后显存释放。滑窗重叠/边缘补齐保持常数概率及指标检查通过。

六组归一化pilot结束，三编码器与两个validation来源等权均值N1=0.053164、N2=0.070203，因此统一选择N2。数值来自500步、固定验证切片的选择试验，不是最终完整体积指标。

P0/P1/P2的seed0正式10k训练已启动。后续自动依次运行seed1/2，再进行完整volume评测与生成final_report.md；所有步骤均在资源guard下运行。

补充历史暴露限制：conrad23在第一轮smoke中曾用于5个验证patch。本轮不用test调参，但该来源不是历史上从未查看过的封存测试。
