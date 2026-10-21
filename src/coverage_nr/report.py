"""Fixed 100k readout report; historical D2 is an explicitly unmatched reference."""
import csv,json,statistics
import numpy as np
from common import R,read,atomic,sha

def csvout(name,rows):
 with (R/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 model='Coverage-NR';folder=R/'downstream/runs'/model/'seed20261014'
 rows=[read(p) for p in folder.glob('*_metrics.json')];assert len(rows)==38
 keys=['dice','precision','recall','hd95_nm','nsd_16nm','predicted_foreground_fraction']
 source=[]
 for s in sorted({r['source_id'] for r in rows}):
  vv=[r for r in rows if r['source_id']==s]
  source.append({'model':model,'source_id':s,'regions':len(vv),**{k:statistics.mean(r[k] for r in vv if r[k] is not None) for k in keys}})
 assert len(source)==12
 csvout('Coverage_NR_per_region_metrics.csv',rows);csvout('Coverage_NR_per_source_metrics.csv',source)
 bm=read(R/'benchmark'/model/'metrics.json');assert len(bm['images'])==2595
 csvout('Coverage_NR_MB1_per_image.csv',bm['images']);csvout('Coverage_NR_MB1_per_dataset.csv',bm['datasets'])
 plan=np.load(R/'plan/ids.npy',mmap_mode='r');values=plan[plan>=0]
 counts=np.bincount(values,minlength=1425433);assert np.all(counts>=1)
 # The trainer checks every consumed index; successful100k therefore consumed this complete plan.
 assert read(R/'runs'/model/'formal/result.json')['completed_steps']==100000
 coverage={'unique_cem_ids':int(np.count_nonzero(counts)),'coverage':1.0,'cem_draws':len(values),'mv_draws':len(plan)-len(values)}
 hashes=[]
 for step in [10000,30000,100000]:
  p=R/'runs'/model/'formal/checkpoints'/f'step{step:09d}/teacher_backbone.pth'
  receipt=read(p.with_suffix('.json'));assert sha(p)==receipt['sha256'];hashes.append({'kind':'EMA_encoder','step':step,'path':str(p),'sha256':receipt['sha256']})
 p=folder/'head_step040000.pth';hashes.append({'kind':'head','step':40000,'path':str(p),'sha256':sha(p)})
 csvout('Coverage_NR_checkpoint_hashes.csv',hashes)
 old=R.parent/'ssl_diversity_d12_100k_20261018'
 history=list(csv.DictReader((old/'D12_100k_source_metrics.csv').open()))
 comparisons=[]
 for row in source:
  ref=next(v for v in history if v['model']=='D2' and v['source_id']==row['source_id'])
  comparisons.append({'source_id':row['source_id'],'NR_dice':row['dice'],'historical_D2_dice':float(ref['dice']),'difference':row['dice']-float(ref['dice'])})
 csvout('Coverage_NR_vs_historical_D2.csv',comparisons)
 summary={'status':'complete','model':model,'dev_macro':{k:statistics.mean(r[k] for r in source) for k in keys},
          'MB1_macro_dice':statistics.mean(r['dice'] for r in bm['datasets']),'coverage':coverage,
          'strict_sampling_ablation':False,'historical_D2_segmented_LR':True,'current_continuous_100k_LR':True,
          'further_experiments_started':False}
 atomic(R/'Coverage_NR_summary.json',summary)
 lines=['# Coverage-NR P0→100k final report','',
 'One exploratory full-coverage run; historical D2 differs in collection distribution and LR history. No causal coverage-only claim. Fixed frozen Primus seed20261014,40k,156train/38dev,TTA OFF. MB1 is a previously exposed diagnostic, not independent blind testing.','',
 'Development source-equal metrics: '+json.dumps(summary['dev_macro']),
 'MB1 seven-dataset equal Dice: '+str(summary['MB1_macro_dice']),
 'Consumed ID coverage: '+json.dumps(coverage),'',
 '| Source | NR Dice | historical D2 | difference |','|---|---:|---:|---:|']
 for r in comparisons:lines.append('| '+r['source_id']+' | '+str(r['NR_dice'])+' | '+str(r['historical_D2_dice'])+' | '+str(r['difference'])+' |')
 lines+=['','Per-region/source and MB1 per-image/dataset metrics, checkpoint SHA256 and immutable sampling plan accompany this report. Inspect per-source regressions and Glycolytic muscle before judging value; ID coverage is not structure coverage. No unseen-acquisition claim or automatic D3/longer training. This authorized run ends here.']
 (R/'Coverage_NR_final_report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
