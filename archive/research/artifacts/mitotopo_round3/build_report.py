"""Build the final Chinese report only once every planned downstream comparison is complete."""
from pathlib import Path
import json
BASE=Path(__file__).resolve().parent
m=json.loads((BASE/'comparison.json').read_text());assert m['status']=='complete',m['pending']
cached=json.loads((BASE/'cached_metrics.json').read_text())
native=json.loads((BASE/'jobs/em_native_recheck/metrics.json').read_text())
mito=json.loads((BASE/'jobs/mitonet_full_v2/metrics.json').read_text())
scale=json.loads((BASE/'mitonet_pregrid_cpu/metrics.json').read_text())
fusion=json.loads((BASE/'fusion_metrics.json').read_text())
smoke=json.loads((BASE/'batch16_feasibility.json').read_text())
newfusion=json.loads((BASE/'em_probe_fusion_metrics.json').read_text())
ds=['Dataset111_CMCC','Dataset131_CMCT'];models=m['models'];best=max(models,key=lambda k:models[k]['equal_dataset_mean']);gramkeys=['gram200k','b2_w1','b2_w2','b4_w2','b2_w2_lr1'];bg=max(gramkeys,key=lambda k:models[k]['equal_dataset_mean'])
def score(model,d):return model['datasets'][d]['macro_dice']
lines=['# CMCC / CMCT 验证、Gram 消融与改进建议','',
'本轮已完成共 7 组 2,000 步 SSL 参数实验、9 种编码器的匹配分割头验证，以及 MitoNet / 现有 EM-DINO 的完整体积验证。额外完成默认 batch=16 的 10 步运行可行性检查。','',
f"匹配分割头的两数据集等权均值最高者：**{models[best]['label']}**；Gram 候选中最高者：**{models[bg]['label']}**。这是单种子、固定监督预算的验证排名，不代表充分收敛或独立测试下的性能上限。",'',
'## 可用基线与竞争模型','',
'以下 EM / DINO / SegFormer 历史模型已在本地训练集做过有监督训练；MitoNet 为官方预训练模型直接推理。这一表不能直接证明架构优劣。EM 的本轮重新推理与历史分数基本一致。MitoNet 使用单轴逐切片网络和官方 2D 后处理，未做三轴集成、三维实例拼接或本地微调。','',
'| 模型 | CMCC Dice | CMCT Dice | 来源 |','|---|---:|---:|---|',
f"| EM-DINOv3 + Primus | {score(native,ds[0]):.6f} | {score(native,ds[1]):.6f} | 本轮重新推理 |"]
for trainer,label in [('dinov3_base_primus_multiscale_Trainer','自然图像 DINOv3 + Primus'),('dinoUNetTrainer','DINO-UNet'),('segformerTrainer','SegFormer')]:lines.append(f"| {label} | {cached['datasets'][ds[0]]['models'][trainer]['macro_dice']:.6f} | {cached['datasets'][ds[1]]['models'][trainer]['macro_dice']:.6f} | 已保存预测重新计分 |")
lines += [f"| MitoNet v1 原生切片尺度 | {score(mito,ds[0]):.6f} | {score(mito,ds[1]):.6f} | 本轮完整推理、官方后处理 |",f"| MitoNet 使用 CMCC 预处理网格 | {scale['macro_dice']:.6f} | — | 额外尺度检查 |",'',
'## Gram 是否提升','',
'相同初始化的 Primus 分割头与 gates、冻结 ViT-L、同一训练采样序列、每个数据集 3,000 个监督更新。训练用 BF16 提取冻结特征、FP32 更新分割头；CE+Dice、AdamW、batch=1。每 1,000 步用固定 stride-20 验证切片选头，最终完整体积使用 224 滑窗、半窗步长、Gaussian 融合和四种翻转 TTA。','',
'| 编码器 / SSL 参数 | CMCC Dice | CMCT Dice | 等权均值 |','|---|---:|---:|---:|']
for v in models.values():lines.append(f"| {v['label']} | {score(v,ds[0]):.6f} | {score(v,ds[1]):.6f} | {v['equal_dataset_mean']:.6f} |")
lines += ['','同 batch、同学习率的 λ=0 对照，才能用于判断本轮预算内的 Gram 作用：','','| 配对实验 | CMCC ΔDice（百分点） | CMCT ΔDice（百分点） |','|---|---:|---:|']
for name,v in m['contrasts'].items():lines.append(f"| {name} | {v['dataset_delta_pp'][ds[0]]:+.3f} | {v['dataset_delta_pp'][ds[1]]:+.3f} |")
lines += ['','旧版 200k 与初始 EM 的差值包含额外训练量、训练域与 Gram 的共同影响，不能全部归因于 Gram。batch=4 每次实验看过 8,000 张采样图像，batch=2 为 4,000 张；跨 batch 排名同时改变了图像数量和损失统计。小幅正负变化未经多种子验证，不声称统计显著。','',
'![匹配验证](probe_comparison.png)','',
'多条学习曲线到 3,000 步仍在上升。该结果回答固定预算下哪组更好，不能排除更长下游训练改变排序。','',
'![学习曲线](probe_learning_curves.png)','',
'## 参数正确性审计','',
'**发现并修正：旧版学习率缩放漏乘 4。** 已直接执行本地官方 `build_schedulers_v2` 函数验证：`sqrt_wrt_1024` 规则为 `4 × sqrt(global_batch/1024)`。batch=2 时旧值为 1.325825e-6，修正后为 5.303301e-6；batch=4 为 7.5e-6，batch=16 为 1.5e-5。参数组还叠加 layerwise decay=0.98 和 patch embedding multiplier=0.2。','',
'GramLoss 的归一化尺度不变性、相同特征零损失、权重与梯度线性缩放、有限梯度均通过检查。global crop=256、local crop=112×8、Gram teacher=512；normalized=true、img_level=true、tokens_used=all、remove_neg=false，teacher resize 为 bicubic 且 antialias=false。AdamW β=(0.9,0.99)、weight decay=0.04、clip=30、EMA=0.999、teacher temperature=0.07。','',
'Gram 目标权重在前 1,000 个本地步骤过渡到设定值。按虚拟调度偏移 1,000,000，锚点刷新落在本地 10k / 20k / 30k，最多 3 次。本轮 2k 实验没有触发刷新。旧版 checkpoint 中锚点权重与初始 EM 不同，说明确有更新；旧日志未逐次记录刷新次数，不将源码推算冒充逐次运行证据。','',
'**并非官方 ViT-7B 全规模复现。** 使用现有 ViT-L 及匹配的 DINO/iBOT heads，继承现有 student 配置（含 drop_path=0.3 等），而官方 ViT-7B 配方含不同结构/heads、drop_path=0.4、每卡 batch=16 等。使用 FP32 主参数和 BF16 autocast、单卡独立实验，起始锚点为现有 EM 125k teacher。新实验从 teacher 导出初始化 student/EMA，并重建 AdamW，没有恢复原始 125k 训练的完整优化器状态，因此不是同一训练运行的无缝续训。虚拟调度到百万步不意味着已真实训练百万步。','']
r=smoke.get('result')
if r and r['status']=='complete':lines.append(f"默认 batch=16 的 10 步 ViT-L 试跑成功，峰值 allocated={r['peak_allocated']/1024**3:.2f} GiB、reserved={r['peak_reserved']/1024**3:.2f} GiB。batch=2/4 是消融设置，不是硬件上限。该试跑无下游成绩，不能用小 batch 结果断言默认 batch 的效果。")
else:lines.append('默认 batch=16 可行性结果详见 batch16_feasibility.json，不据失败或未完成试跑给出质量分数。')
lines += ['','官方参考：[Gram 配置](https://github.com/facebookresearch/dinov3/blob/main/dinov3/configs/train/dinov3_vit7b16_gram_anchor.yaml)、[训练实现](https://github.com/facebookresearch/dinov3/blob/main/dinov3/train/train.py)。本地实际代码散列与调度函数输出见 official_scheduler_audit.json。','',
'## 数据与计分边界','',
'使用 Dataset111_CMCC / Dataset131_CMCT 原始 fold_0：CMCC 为 crop113、crop110、crop3；CMCT 为 liver_crop145、kidney_crop156。完整验证共 1,481 张切片。每种最终预测的体素总数与前景真值数均通过一致性断言。原始标签只有 0/1，因此报告 Dice、IoU、召回与混淆计数，不伪报实例 PQ/AP。','',
'CMCC 的预处理尺寸与 plans spacing 推算不一致。本轮依据实际保存的图像网格、原始形状及裁剪元数据恢复预测，最终在原始网格计分；预处理 -1 只用于监督训练忽略区，不用来排除原始验证体素。原始 CMCC 训练集 14/27 个体积标签全背景，仍保留；监督采样 50% 从有前景训练体积及坐标采样，50% 从全部训练体积采样。','',
'SSL 仅用 jiang25_vol1，不使用这两个数据集的验证图像。原始划分是 ROI 级，尚未证明 specimen 独立或基础模型预训练完全无重叠。验证集参与参数/头选择，本结果不是独立测试集成绩。','',
'## 非 Gram 的改进线索','',
f"EM 与 MitoNet 原生预测取并集，CMCC Dice={fusion['datasets'][ds[0]]['union']['macro_dice']:.6f}，较 EM 提升 {100*(fusion['datasets'][ds[0]]['union']['macro_dice']-score(native,ds[0])):.2f} 个百分点；CMCT Dice={fusion['datasets'][ds[1]]['union']['macro_dice']:.6f}，较 EM 下降 {100*(score(native,ds[1])-fusion['datasets'][ds[1]]['union']['macro_dice']):.2f} 个百分点。它表明模型互补，但不是通用替代方案，也不是 Gram 收益。",'',
'![基线切片对照](baseline_overlays.png)','',
'图中的旧版 Gram 接旧头仅为兼容性诊断。其完整 Dice 为 CMCC 0、CMCT 0.04342；正式表征比较使用上方重新训练的相同初始化分割头。切片由真值前景数量选取，仅作展示，排名使用完整体积。','',
'## 下一版建议','',
'优先推进：充分训练/分阶段解冻下游头并重复种子；使用训练 ROI 的多域 SSL 数据；核对单通道复制与 RGB 归一化差异；在可行的默认 batch 下做配对对照，训练跨过锚点更新点；检验高分辨率局部细节和置信度融合。不要只延长单一体积的 Gram 训练。详见 [下一版执行方案](next_iteration_plan.md)。','',
'## 交付与复现','',
'完整逐体积计数及配对差值见 comparison.json；基线见 cached_metrics.json、fusion_metrics.json；参数审计与资源日志随交付包保存。服务器目录为 `/opt/data/3_117/tanghr/codex/mitotopo_20260909_round3/`，大 checkpoint 和完整预测留在 jobs 各运行目录。仅 GPU 1–3 参与，GPU 0 未使用；原网站服务保留。异常/中断尝试不参与最终排名，使用已保存头恢复评估的来源已记录。']
c=m['contrasts']['Gram λ2 vs λ0, b2 old LR']['dataset_delta_pp']
lines[4:4]=[f"**核心结论：修正 LR 下的三个同 batch 配对实验中，Gram 均未提高 CMCC/CMCT 宏平均 Dice。较小 LR 下，λ=2 相对 λ=0 的变化为 CMCC {c[ds[0]]:+.3f}、CMCT {c[ds[1]]:+.3f} 个百分点。** 这些是单种子短程验证结果，不是默认 batch=16 或长期训练的普遍结论。",'']
idx=lines.index('## 参数正确性审计')+2
lines[idx:idx]=['另从保存的优化器状态核对了各运行 58 个参数组的实际 LR 与 weight decay，全部通过。较小 LR 是显式的 0.25 倍消融覆盖值；不能仅看名义 YAML 推断实际 LR，详见 effective_optimizer_audit.json。λ=0 保留相同 Gram 计算分支但加权为零；有效 λ 来自 loss_weight_schedule，而非孤立读取静态 loss_weight 字段。','']
idx=lines.index('## 非 Gram 的改进线索')+2
lines[idx:idx]=[f"另一个不含 Gram 的融合（已有微调 EM 与新训练冻结 EM 分割头取并集）得到 CMCC {score(newfusion,ds[0]):.6f}、CMCT {score(newfusion,ds[1]):.6f}。它在 CMCC 更强，但明显损害 CMCT，不能统一替代原模型。",'']
(BASE/'validation_report.md').write_text('\n'.join(lines),encoding='utf-8');print('final validation_report.md written')
