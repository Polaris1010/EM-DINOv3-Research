"""Aggregate only the prespecified midpoint and its two fixed historical comparators."""
import csv,statistics,math
from common import R,PA,A2,PARENT,LATE,MODEL,read,atomic,sha
def csvout(name,rows):
 with (R/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def table(path):
 with path.open() as f:return list(csv.DictReader(f))
def mean(a):
 a=[x for x in a if x is not None];return statistics.mean(a) if a else None
def fmt(x):return 'NA' if x is None else f'{x:.6f}'
def main():
 roots={'D2-30k':PARENT,'D2-100k':LATE,MODEL:R};dev=[];sources=[];benchmark=[];muscle=[]
 for model,root in roots.items():
  folder=root/'downstream/runs'/(MODEL if model==MODEL else 'D2')/'seed20261014'
  rr=[read(p) for p in folder.glob('*_metrics.json')];assert len(rr)==38 and len({r['region_id'] for r in rr})==38
  dev.extend({**r,'model':model} for r in rr)
  for source in sorted({r['source_id'] for r in rr}):
   group=[r for r in rr if r['source_id']==source];sources.append({'model':model,'source_id':source,'regions':len(group),**{k:mean([r[k] for r in group]) for k in ['dice','precision','recall','hd95_nm','nsd_16nm','predicted_foreground_fraction']}})
  j=read(root/'benchmark'/(MODEL if model==MODEL else 'D2')/'metrics.json');b=[dict(x,model=model) for x in j['datasets']]
  extra=read(R/'benchmark'/model/'metrics.json')
  for row in b:
   if row['dataset']=='glycolytic_muscle':
    m=next(x for x in extra['datasets'] if x['dataset']=='glycolytic_muscle');row.update(pixel_AP=m['pixel_AP'],pixel_AUROC=m['pixel_AUROC'],far_FP_rate=m['far_FP_rate'])
  benchmark+=b;muscle.extend(dict(x,model=model) for x in extra['images'] if x['dataset']=='glycolytic_muscle')
  if model==MODEL:csvout('PilotB_new_MB1_per_image.csv',j['images'])
 csvout('PilotB_dev_per_region.csv',dev);csvout('PilotB_dev_per_source.csv',sources);csvout('PilotB_MB1_per_dataset.csv',benchmark);csvout('PilotB_muscle_Primus_per_image.csv',muscle)
 pairs=[]
 for row in [x for x in dev if x['model']==MODEL]:
  a=next(x for x in dev if x['model']=='D2-100k' and x['region_id']==row['region_id'])
  assert row['valid_voxels']==a['valid_voxels'] and row['gt_foreground_voxels']==a['gt_foreground_voxels']
  pairs.append({'region_id':row['region_id'],'source_id':row['source_id'],'D2_100k_dice':a['dice'],'midpoint_dice':row['dice'],'delta':row['dice']-a['dice']})
 csvout('PilotB_dev_paired_vs_100k.csv',pairs)
 linear=[];raw=table(A2/'tables/baseline_source_metrics.csv');rank=table(A2/'tables/baseline_pixel_discrimination.csv')
 for row in raw:
  j=next(x for x in rank if x['model']==row['model'] and x['source']==row['source'] and x['scope']=='all_valid')
  linear.append({'model':row['model'],'source':row['source'],**{k:float(row[k]) for k in ['dice','precision','recall','far_FP_rate']},'pixel_AUROC':float(j['AUROC']),'pixel_AP':float(j['AP'])})
 for row in table(R/'linear/pixel_metrics.csv'):linear.append({k:row[k] if k in ['model','source'] else float(row[k]) for k in ['model','source','dice','precision','recall','far_FP_rate','pixel_AUROC','pixel_AP']})
 csvout('PilotB_Linear23_comparison.csv',linear)
 summary={}
 for model in roots:
  ss=[x for x in sources if x['model']==model];bb=[x for x in benchmark if x['model']==model];assert len(ss)==12 and len(bb)==7
  gm=next(x for x in bb if x['dataset']=='glycolytic_muscle');ll=next(x for x in linear if x['model']==model and x['source']=='glycolytic_muscle')
  summary[model]={'MV3_source_equal_Dice':mean([x['dice'] for x in ss]),'MB1_dataset_equal_Dice':mean([x['dice'] for x in bb]),'muscle_Primus_Dice':gm['dice'],'muscle_Primus_AP_mean_per_plane':gm['pixel_AP'],'muscle_Primus_far_FP_rate':gm['far_FP_rate'],'muscle_Linear23_pooled_AP':ll['pixel_AP'],'muscle_Linear23_Dice':ll['dice']}
 atomic(R/'PilotB_summary.json',{'status':'complete','models':summary,'alpha':.5,'Linear_performance_never_gated_Primus':True,'new_SSL_steps':0,'AP_definitions':'Linear pooled pixels in64 fixed FOVs; Primus muscle mean per-plane semantic pixel AP over full volume','causal_limits':['Weight combination utility is not evidence of an anchoring loss mechanism','Independent probes/heads; no encoder common-mode attribution','Exposed development sets, one decoder seed; not a new blind external test'],'stop_after':True})
 lines=['# Pilot B：D2固定轨迹中点实验','', '## 实测汇总','', '| 模型 | MV3来源等权Dice | MB1七组等权Dice | 肌肉Primus Dice | 肌肉Primus AP（逐层均值） | 肌肉远端FP率 | 肌肉Linear AP（固定视野池化） |','|---|---:|---:|---:|---:|---:|---:|']
 for model,s in summary.items():lines.append('| '+model+' | '+' | '.join(fmt(s[k]) for k in ['MV3_source_equal_Dice','MB1_dataset_equal_Dice','muscle_Primus_Dice','muscle_Primus_AP_mean_per_plane','muscle_Primus_far_FP_rate','muscle_Linear23_pooled_AP'])+' |')
 lines+=['','新模型只由两正式EMA在float64中固定0.5平均并转回各原始dtype。无新SSL、系数搜索或其他融合变体。Linear23使用原312个train视野、500步和seed20261027；随后无条件完成原版Primus40k、seed20261014、TTA OFF。','',
 '## 完整开发来源','', '| 来源 | D2/30k | D2/100k | Trajectory-Lerp05 | 中点−100k |','|---|---:|---:|---:|---:|']
 for source in sorted({x['source_id'] for x in sources}):
  d={x['model']:x['dice'] for x in sources if x['source_id']==source};lines.append('| '+source+' | '+' | '.join(fmt(d[m]) for m in roots)+' | '+fmt(d[MODEL]-d['D2-100k'])+' |')
 lines+=['','## 固定MB1七组','', '| 数据集 | D2/30k | D2/100k | Trajectory-Lerp05 | 中点−100k |','|---|---:|---:|---:|---:|']
 for dataset in sorted({x['dataset'] for x in benchmark}):
  d={x['model']:x['dice'] for x in benchmark if x['dataset']==dataset};lines.append('| '+dataset+' | '+' | '.join(fmt(d[m]) for m in roots)+' | '+fmt(d[MODEL]-d['D2-100k'])+' |')
 lines+=['','## 解释与停止','', '本轮只检验固定单Encoder权重组合的实用性。成功不能证明谱压缩是错误来源、轨迹锚定机制成立或新SSL loss有效；失败也不否定预训练创新路线。空间共模方向仍属暂停而非已被直接干预证伪。','',
 'Primus连续概率使用原sliding源码，仅暴露阈值前返回值；两历史端点只补充肌肉固定推理，并逐像素验证mask等于既有预测。Linear与Primus AP的聚合单位不同，方向可作描述性对照，绝对数值不混排。未用MB1标签选择参数。','',
 '新Encoder记录encoder.json；源码/协议/环境/采样与head哈希独立保存。全部38开发区域、七组2595张MB1图像完成；完成后停止，不启动Pilot C、Gram、TopoCL、LAST、SVD或新融合。','']
 (R/'PilotB_generated_report.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
