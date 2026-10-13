import csv,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path(__file__).resolve().parent/'p1_mitonet_analysis_20260927'
e=json.loads((base/'audited_evidence.json').read_text());rows=e['rows']
def avg(xs):return statistics.mean(xs)
sources=sorted({(x['split'],x['source']) for x in rows},key=lambda x:(x[0]!='validation',x[1]))
summary=[]
for split,source in sources:
 s={'split':split,'source':source,'volumes':len({x['volume_id'] for x in rows if x['source']==source})}
 for model in ('MitoNet','P1'):
  rr=[x for x in rows if x['model']==model and x['source']==source]
  for metric in ('dice','precision','recall'):s[f'{model}_{metric}']=avg(x[metric] for x in rr)
  s[f'{model}_fnr_macro']=1-s[f'{model}_recall'];s[f'{model}_fdr_macro']=1-s[f'{model}_precision']
  sums={k:sum(x[k] for x in rr)/(3 if model=='P1' else 1) for k in ('tp','fp','fn','tn')}
  gt=sums['tp']+sums['fn']
  s[f'{model}_fn_per_gt_micro']=sums['fn']/gt;s[f'{model}_fp_per_gt_micro']=sums['fp']/gt
  s[f'{model}_fn_share_of_errors_micro']=sums['fn']/(sums['fn']+sums['fp'])
  s[f'{model}_pred_over_gt_micro']=(sums['tp']+sums['fp'])/gt
  if model=='P1':
   vals=[avg(x['dice'] for x in rr if x['seed']==seed) for seed in range(3)]
   s['P1_dice_seed_min']=min(vals);s['P1_dice_seed_max']=max(vals)
 s['MitoNet_minus_P1_dice']=s['MitoNet_dice']-s['P1_dice']
 s['excess_P1_fn_per_gt']=s['P1_fn_per_gt_micro']-s['MitoNet_fn_per_gt_micro']
 s['excess_P1_fp_per_gt']=s['P1_fp_per_gt_micro']-s['MitoNet_fp_per_gt_micro']
 summary.append(s)
for name,data in [('per_volume.csv',rows),('per_source.csv',summary)]:
 with (base/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
totals={split:{model:avg(s[f'{model}_dice'] for s in summary if s['split']==split) for model in ('MitoNet','P1')} for split in ('validation','legacy_dev_test')}
(base/'summary.json').write_text(json.dumps({'sources':summary,'split_macro_dice':totals},indent=2),encoding='utf-8')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,axes=plt.subplots(1,3,figsize=(15,5.4),sharey=True)
y=list(range(len(summary)))
for ax,metric,title in zip(axes,['dice','fnr_macro','fdr_macro'],['Dice (higher is better)','Miss rate: FN / (TP + FN)','False discovery: FP / (TP + FP)']):
 for offset,model,color in [(-.18,'MitoNet','#247ba0'),(.18,'P1','#ef8354')]:
  ax.barh([v+offset for v in y],[100*s[f'{model}_{metric}'] for s in summary],height=.32,label=model,color=color)
 ax.set_title(title);ax.set_xlim(0,100);ax.set_xlabel('%');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
axes[0].set_yticks(y,[s['source']+(' [val]' if s['split']=='validation' else ' [dev]') for s in summary]);axes[0].invert_yaxis();axes[0].legend(loc='lower right')
fig.suptitle('Official MitoNet vs EM-DINOv3 P1 + trained segmentation head\n24 full volumes; P1 averaged over 3 head seeds; volume-macro within source',fontsize=12)
fig.tight_layout();fig.savefig(base/'source_comparison.png',dpi=180);plt.close(fig)
lines=['# 官方 MitoNet 与 EM-DINOv3 P1：现有检查点对比','',
'证据核查日期：2026-09-27。复用服务器完整全体积评估，8 个 validation 体积与 16 个 legacy_dev_test 体积，共 8 个来源。P1 使用修复 mask 后的三个已训练分割头，逐 seed 评估后取均值；不是集成预测。',
'','| 来源 | 划分 | 体积数 | MitoNet Dice | P1 Dice | 差值（百分点） | MitoNet 漏检率 | P1 漏检率 | MitoNet 误检占比 | P1 误检占比 |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for s in summary:
 lines.append(f"| {s['source']} | {s['split']} | {s['volumes']} | {s['MitoNet_dice']*100:.2f}% | {s['P1_dice']*100:.2f}% | {s['MitoNet_minus_P1_dice']*100:+.2f} | {s['MitoNet_fnr_macro']*100:.2f}% | {s['P1_fnr_macro']*100:.2f}% | {s['MitoNet_fdr_macro']*100:.2f}% | {s['P1_fdr_macro']*100:.2f}% |")
lines+=['','Dice、漏检率、误检占比先在来源内对体积等权，P1 再对三个 head seed 等权。漏检率 = FN/(TP+FN)，误检占比 = FP/(TP+FP)，均为前景像素指标，不是实例个数指标。','',
'## 来源等权汇总','']
for split,d in totals.items():lines.append(f"- {split}: MitoNet {d['MitoNet']:.6f}; P1 {d['P1']:.6f}; 差值 {(d['MitoNet']-d['P1'])*100:+.2f} 个百分点。")
lines+=['','## 错误面积分解','',
'下表按来源汇总像素计数，P1 对 seed 取平均，再用相同真实前景总面积归一化。ΔFN 与 ΔFP 是 P1 减 MitoNet，正数表示 P1 多出的错误面积；此口径用于错误面积归因，与上表体积等权口径不同。','',
'| 来源 | P1 错误像素中 FN 占比 | ΔFN / GT | ΔFP / GT |','|---|---:|---:|---:|']
for s in summary:lines.append(f"| {s['source']} | {s['P1_fn_share_of_errors_micro']*100:.2f}% | {s['excess_P1_fn_per_gt']*100:+.2f}% | {s['excess_P1_fp_per_gt']*100:+.2f}% |")
lines+=['','## 分析结论','',
'- 官方 MitoNet 在 8 个来源中的 7 个优于 P1，且这 7 个来源均高于 P1 三个 head seed 中的最好值。guay21 上 P1 略高，但两者 Dice 都低于 5%。',
'- casser20、xiao18、wei20 的主要差距是 P1 误检；尤其 xiao18，P1 recall 为 96.37%，但 precision 仅 27.42%，大量背景被判为前景。',
'- jiang25 的差距以漏检为主：P1 漏检率 75.08%，MitoNet 为 13.86%；误检也有差距。',
'- mekuc20 的 P1 同时存在严重漏检和误检，漏检错误面积略占主导；漏检率 91.46%，预测前景中 79.29% 为误检。不能仅以“预测太少”解释。',
'- conrad23 两类错误都很严重；按来源汇总的额外错误面积，P1 多出的误检占主要部分。该来源不同体积权重差异大，应继续查看 per_volume.csv。',
'- haberl18 的 P1 几乎完全失效，漏检率 98.80%、误检占比 98.12%；MitoNet 自身漏检率也有 81.96%。相对 MitoNet 的额外错误面积以 FP 为主，但不能忽略两者严重漏检。',
'- guay21 两者都几乎漏掉全部前景，同时检出的部分也大多不正确，应优先核查成像尺度、图像强度、标签语义及样本对应关系，再考虑调整模型。',
'',
'优先后续工作：按来源查看同位置 TP/FP/FN 叠图；casser20/xiao18/wei20 重点检查误检形态，jiang25/mekuc20 重点检查漏检和域差异，guay21/haberl18 优先核查数据与推理适配。以上错误形态本身不能确定具体成因。阈值如需校准，应仅在预先定义的验证集上选择，不能用这些开发测试来源逐个调优后作为独立成绩。',
'','## 复核与边界','',
'- 当前 P1 编码器和官方 MitoNet 权重 SHA256 与锁定协议一致；P1 当前分割头哈希与开发测试评估记录一致。',
'- 96 条模型/seed/体积结果完整，Dice 从 TP/FP/FN 重算一致；每个体积四份评估的真实前景和背景总数一致，预测网格及切片覆盖完整。',
'- 另读取 96 个中心切片的保存预测、标签与有效区进行检查；保存的二值预测与概率阈值仅允许 float16 在 0.5 附近的舍入差异。这不是重新执行全体积推理，也不是全量标签重计数。',
'- MitoNet 为官方未量化 MitoNet_v1 的固定 2D 语义前景推理，原始网格、512 tile / 256 stride、官方归一化、阈值 0.5、无 TTA；不等同于官方完整 3D 实例流程。',
'- P1 为旧 EM 持续预训练编码器（training_124999）加 mask 修复后训练的分割头，不是正在训练的 B/L/T/LT。',
'- 三个分割头均完成 10k 训练；使用原验证规则选定的 head_best，seed0/1/2 对应 step 3000/1000/1000，不是按本次测试成绩重新挑选。',
'- legacy_dev_test 已用于历史开发，两模型预训练数据暴露未完全厘清；不能将差距解释为严格未见域优势或架构的因果优势。',
'- 本次无需新增 GPU 推理，训练队列保持运行。',
'','完整证据见 audited_evidence.json；来源汇总见 per_source.csv；逐体积结果见 per_volume.csv。']
(base/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'sources':summary,'split_macro_dice':totals},indent=2))
