"""Seven fixed systems and absolute100k effects; no metric-dependent selection."""
from pathlib import Path
import os,json,csv,statistics,math
R=Path(os.environ['PILOT_C100_ROOT']);B=R.parent/'pilot_c2_paired_45k_20261028'
NEW=['C0_Vanilla_100k','C1_Barrier_100k'];HIST=['D2-30k','D2-100k','Trajectory-Lerp05','C0_Vanilla_45k','C1_Barrier_45k'];MODELS=HIST+NEW
def read(p):return json.loads(p.read_text())
def table(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def num(v):return None if v in ('',None) else float(v)
def mean(v):
 a=[num(x) for x in v if x not in ('',None)];return statistics.mean(a) if a else None
def difference(a,b,k):
 x,y=num(a.get(k)),num(b.get(k))
 return x-y if x is not None and y is not None and math.isfinite(x) and math.isfinite(y) else None
def csvout(name,rows):
 with (R/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for x in rows for k in x)));w.writeheader();w.writerows(rows)
def main():
 dev=table(B/'PilotC_dev_per_region.csv');sources=table(B/'PilotC_dev_per_source.csv');bench=table(B/'PilotC_MB1_per_dataset.csv');linear=table(B/'PilotC_Linear23_comparison.csv');muscle=table(B/'PilotC_muscle_Primus_per_image.csv');spectra=[]
 assert (len(dev),len(sources),len(bench),len(linear),len(muscle))==(190,60,35,25,1510)
 keys=['dice','precision','recall','hd95_nm','nsd_16nm','predicted_foreground_fraction']
 for model in NEW:
  root=R/'systems'/model;folder=root/'downstream/runs'/model/'seed20261014';rows=[dict(read(p),model=model) for p in folder.glob('*_metrics.json')];assert len(rows)==38
  dev+=rows
  for source in sorted({x['source_id'] for x in rows}):
   group=[x for x in rows if x['source_id']==source];sources.append({'model':model,'source_id':source,'regions':len(group),**{k:mean([x[k] for x in group]) for k in keys}})
  metrics=read(root/'benchmark'/model/'metrics.json');assert len(metrics['images'])==2595 and len(metrics['datasets'])==7
  bench+=metrics['datasets'];muscle +=[x for x in metrics['images'] if x['dataset']=='glycolytic_muscle'];csvout(f'PilotC100_{model}_MB1_per_image.csv',metrics['images'])
  linear+=table(root/'linear/pixel_metrics.csv');spectra+=table(root/'linear/spectrum.csv')
 csvout('PilotC100_dev_per_region.csv',dev);csvout('PilotC100_dev_per_source.csv',sources);csvout('PilotC100_MB1_per_dataset.csv',bench);csvout('PilotC100_Linear23_comparison.csv',linear);csvout('PilotC100_Linear23_new_spectrum.csv',spectra);csvout('PilotC100_muscle_Primus_per_image.csv',muscle)
 paired=[]
 for region in sorted({x['region_id'] for x in dev}):
  a={x['model']:x for x in dev if x['region_id']==region};assert len(a)==7
  assert len({(int(x['valid_voxels']),int(x['gt_foreground_voxels'])) for x in a.values()})==1
  c0,c1=NEW;old0,old1=HIST[3:];v0=difference(a[c0],a[old0],'dice');v1=difference(a[c1],a[old1],'dice')
  paired.append({'region_id':region,'source_id':a[c0]['source_id'],**{m:num(a[m]['dice']) for m in MODELS},'C1_minus_C0_100k':difference(a[c1],a[c0],'dice'),'C1_minus_Midpoint':difference(a[c1],a[HIST[2]],'dice'),'C0_45k_to100k':v0,'C1_45k_to100k':v1,'change_difference_C1_minus_C0':v1-v0})
 csvout('PilotC100_paired_dev_deltas.csv',paired)
 deltas=[]
 for source in sorted({x['source_id'] for x in sources}):
  a={x['model']:x for x in sources if x['source_id']==source};v0=difference(a[NEW[0]],a[HIST[3]],'dice');v1=difference(a[NEW[1]],a[HIST[4]],'dice')
  deltas.append({'source':source,**{m:num(a[m]['dice']) for m in MODELS},'C1_minus_C0_Dice':difference(a[NEW[1]],a[NEW[0]],'dice'),'C1_minus_Midpoint_Dice':difference(a[NEW[1]],a[HIST[2]],'dice'),'C1_minus_C0_NSD':difference(a[NEW[1]],a[NEW[0]],'nsd_16nm'),'C0_45k_to100k':v0,'C1_45k_to100k':v1,'change_difference_C1_minus_C0':v1-v0})
 csvout('PilotC100_source_deltas.csv',deltas)
 bd=[]
 for dataset in dict.fromkeys(x['dataset'] for x in bench):
  a={x['model']:x for x in bench if x['dataset']==dataset};out={'dataset':dataset}
  for k in ['dice','iou','precision','recall','nsd_16nm','pixel_AP','pixel_AUROC','far_FP_rate','predicted_foreground_fraction']:
   out[k+'_C1_minus_C0']=difference(a[NEW[1]],a[NEW[0]],k);out[k+'_C1_minus_Midpoint']=difference(a[NEW[1]],a[HIST[2]],k)
  v0=difference(a[NEW[0]],a[HIST[3]],'dice');v1=difference(a[NEW[1]],a[HIST[4]],'dice');out.update(C0_45k_to100k_Dice=v0,C1_45k_to100k_Dice=v1,change_difference_C1_minus_C0=v1-v0);bd.append(out)
 csvout('PilotC100_MB1_paired_deltas.csv',bd)
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
 csvout('PilotC100_SSL_curves_stride100.csv',curves)
 comparisons={'C1_minus_C0_100k':{k:summary[NEW[1]][k]-summary[NEW[0]][k] for k in summary[NEW[0]]},'C1_minus_Midpoint':{k:summary[NEW[1]][k]-summary[HIST[2]][k] for k in summary[NEW[0]]}}
 for model,old in zip(NEW,HIST[3:]):comparisons[model+'_45k_to100k']={k:summary[model][k]-summary[old][k] for k in summary[model]}
 comparisons['change_difference_C1_minus_C0']={k:comparisons[NEW[1]+'_45k_to100k'][k]-comparisons[NEW[0]+'_45k_to100k'][k] for k in summary[NEW[0]]}
 result={'status':'complete','models':summary,'comparisons':comparisons,'primary_comparison':'Absolute C1 versus matched C0 at100k','change_difference_role':'Descriptive; not sufficient to establish net benefit from unequal45k performance','fixed_lambda':1,'fixed_margin':.05,'paired_SSL':read(R/'paired_SSL_verification.json'),'anchor_remains_original30k':True,'additional_experiments_started':False,'scientific_limits':['Masked DropPath student versus unmasked eval anchor: efficacy does not identify a unique mechanism','One decoder seed; MB1 already exposed diagnostic; no label-based parameter selection']}
 (R/'PilotC100_summary.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Pilot C100 fixed paired endpoint','','All authorized SSL/readout/evaluation budgets completed; no additional training.','', '| System | MV3 Dice | NSD | MB1 Dice | Muscle Dice | AP | Recall | Far FP |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for model in MODELS:
  x=summary[model];lines.append('| '+model+' | '+' | '.join(f'{x[k]:.6f}' for k in ['MV3_source_equal_Dice','MV3_source_equal_NSD','MB1_dataset_equal_Dice','muscle_Primus_Dice','muscle_Primus_AP_mean_per_plane','muscle_Primus_Recall','muscle_Primus_far_FP_rate'])+' |')
 lines+=['','Primary effect is C1 minus C0 at the absolute100k endpoint. Compare all sources and the fixed midpoint; lower FP alone is insufficient if Dice/AP/Recall deteriorate. Change-from45k is descriptive because the arms already differed at45k.','Original30k anchor, lambda1/margin0.05/bottom50%/Chebyshev>2/layer23; two55k continuations, matched Primus40k, Linear500 and full38dev/7MB1 each. No metric gating, threshold/scale search, new loss or extension.','']
 (R/'PilotC100_generated_report.md').write_text('\n'.join(lines));print(json.dumps(summary))
if __name__=='__main__':main()