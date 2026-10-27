"""Five fixed systems; no model selection, GT fitting or changed historical rows."""
from pathlib import Path
import os,json,csv,statistics
R=Path(os.environ['PILOT_C2_ROOT']);B=R.parent/'pilot_b_trajectory_lerp05_20261027';NEW=['C0_Vanilla_45k','C1_Barrier_45k'];HIST=['D2-30k','D2-100k','Trajectory-Lerp05'];MODELS=HIST+NEW
def read(p):return json.loads(p.read_text())
def table(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def num(v):return None if v in ('',None) else float(v)
def mean(v):
 a=[num(x) for x in v if x not in ('',None)];return statistics.mean(a) if a else None
def csvout(name,rows):
 with (R/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for x in rows for k in x)));w.writeheader();w.writerows(rows)
def main():
 dev=table(B/'PilotB_dev_per_region.csv');sources=table(B/'PilotB_dev_per_source.csv');bench=table(B/'PilotB_MB1_per_dataset.csv');linear=table(B/'PilotB_Linear23_comparison.csv');muscle=table(B/'PilotB_muscle_Primus_per_image.csv');spectra=[]
 keys=['dice','precision','recall','hd95_nm','nsd_16nm','predicted_foreground_fraction']
 for model in NEW:
  root=R/'systems'/model;folder=root/'downstream/runs'/model/'seed20261014';rows=[dict(read(p),model=model) for p in folder.glob('*_metrics.json')];assert len(rows)==38
  dev+=rows
  for source in sorted({x['source_id'] for x in rows}):
   group=[x for x in rows if x['source_id']==source];sources.append({'model':model,'source_id':source,'regions':len(group),**{k:mean([x[k] for x in group]) for k in keys}})
  metrics=read(root/'benchmark'/model/'metrics.json');assert len(metrics['images'])==2595 and len(metrics['datasets'])==7
  bench +=metrics['datasets'];muscle +=[x for x in metrics['images'] if x['dataset']=='glycolytic_muscle'];csvout(f'PilotC_{model}_MB1_per_image.csv',metrics['images'])
  linear+=table(root/'linear/pixel_metrics.csv');spectra+=table(root/'linear/spectrum.csv')
 csvout('PilotC_dev_per_region.csv',dev);csvout('PilotC_dev_per_source.csv',sources);csvout('PilotC_MB1_per_dataset.csv',bench);csvout('PilotC_Linear23_comparison.csv',linear);csvout('PilotC_Linear23_new_spectrum.csv',spectra);csvout('PilotC_muscle_Primus_per_image.csv',muscle)
 paired=[]
 for region in sorted({x['region_id'] for x in dev}):
  a={x['model']:x for x in dev if x['region_id']==region};assert len(a)==5
  assert len({(int(x['valid_voxels']),int(x['gt_foreground_voxels'])) for x in a.values()})==1
  paired.append({'region_id':region,'source_id':a[NEW[0]]['source_id'],'C0_Dice':num(a[NEW[0]]['dice']),'C1_Dice':num(a[NEW[1]]['dice']),'Midpoint_Dice':num(a[HIST[2]]['dice']),'C1_minus_C0':num(a[NEW[1]]['dice'])-num(a[NEW[0]]['dice']),'C1_minus_Midpoint':num(a[NEW[1]]['dice'])-num(a[HIST[2]]['dice'])})
 csvout('PilotC_paired_dev_deltas.csv',paired)
 source_delta=[]
 for source in sorted({x['source_id'] for x in sources}):
  a={x['model']:x for x in sources if x['source_id']==source}
  source_delta.append({'source':source,**{m:num(a[m]['dice']) for m in MODELS},'C1_minus_C0_Dice':num(a[NEW[1]]['dice'])-num(a[NEW[0]]['dice']),'C1_minus_Midpoint_Dice':num(a[NEW[1]]['dice'])-num(a[HIST[2]]['dice']),'C1_minus_C0_NSD':num(a[NEW[1]]['nsd_16nm'])-num(a[NEW[0]]['nsd_16nm'])})
 csvout('PilotC_source_deltas.csv',source_delta)
 summary={}
 for model in MODELS:
  ss=[x for x in sources if x['model']==model];bb=[x for x in bench if x['model']==model];assert len(ss)==12 and len(bb)==7
  gm=next(x for x in bb if x['dataset']=='glycolytic_muscle');lin=next(x for x in linear if x['model']==model and x['source']=='glycolytic_muscle')
  summary[model]={'MV3_source_equal_Dice':mean([x['dice'] for x in ss]),'MV3_source_equal_NSD':mean([x['nsd_16nm'] for x in ss]),'MB1_dataset_equal_Dice':mean([x['dice'] for x in bb]),'muscle_Primus_Dice':num(gm['dice']),'muscle_Primus_AP_mean_per_plane':num(gm['pixel_AP']),'muscle_Primus_AUROC_mean_per_plane':num(gm['pixel_AUROC']),'muscle_Primus_Recall':num(gm['recall']),'muscle_Primus_far_FP_rate':num(gm['far_FP_rate']),'muscle_Linear23_AP':num(lin['pixel_AP']),'muscle_Linear23_Dice':num(lin['dice'])}
 curves=[]
 for model in NEW:
  for rank in range(3):
   selected={}
   with (R/'runs'/model/f'steps_rank{rank}.jsonl').open() as f:
    for line in f:
     x=json.loads(line)
     if x['step']%100==0:selected[x['step']]=x
   curves +=[{k:x.get(k) for k in ['model','rank','step','loss','grad_norm','lr','elapsed_seconds','iteration_seconds','barrier_raw','active_fraction','barrier_backward_seen','peak_allocated_bytes']} for x in selected.values()]
 csvout('PilotC_SSL_curves_stride100.csv',curves)
 result={'status':'complete','models':summary,'primary_comparison':'C1 versus same-budget C0 at45k','diagnostic_baseline':'Pilot B fixed0.5 midpoint, historical30k and100k','fixed_lambda':1,'fixed_margin':.05,'paired_SSL':read(R/'paired_SSL_verification.json'),'additional_experiments_started':False,'scientific_limits':['Student iBOT/DropPath versus unmasked eval anchor: activation does not identify forgetting','Matched45k comparison does not prove prevention of100k forgetting','One decoder seed and exposed MB1; no significance or blind external-test claim']}
 (R/'PilotC_summary.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Pilot C：固定45k配对干预结果','','| 系统 | MV3来源等权Dice | NSD | MB1七组Dice | 肌肉Dice | 肌肉AP | 肌肉Recall | 肌肉远端FP |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for model in MODELS:
  x=summary[model];lines.append('| '+model+' | '+' | '.join(f'{x[k]:.6f}' for k in ['MV3_source_equal_Dice','MV3_source_equal_NSD','MB1_dataset_equal_Dice','muscle_Primus_Dice','muscle_Primus_AP_mean_per_plane','muscle_Primus_Recall','muscle_Primus_far_FP_rate'])+' |')
 lines+=['','固定lambda1、margin0.05、bottom50%、Chebyshev>2、layer23；仅两个模型各从原30k完整状态续训15k。正式训练没有额外诊断student forward或VJP，Barrier在同一backward中参与全部实际更新。随后原版冻结Primus40k/seed20261014、Linear23及全部38dev/七组MB1完成。','', '本轮只能直接判断同45k预算C1相对C0的效果，不能据激活率证明遗忘机制，也不能在C0尚未退化时宣称防止100k遗忘。完成后停止，无额外训练或调参。','']
 (R/'PilotC_generated_report.md').write_text('\n'.join(lines));print(json.dumps(summary))
if __name__=='__main__':main()
