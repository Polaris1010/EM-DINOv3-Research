"""Plots of the completed fixed comparison; no candidate or parameter selection."""
from pathlib import Path
import argparse,csv,json
import matplotlib
matplotlib.use('Agg');matplotlib.rcParams['svg.fonttype']='none'
import matplotlib.pyplot as plt
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--review-dir',required=True);a=p.parse_args()
 r=Path(a.run_dir);review=Path(a.review_dir);out=r/'figures';out.mkdir(exist_ok=True);review.mkdir(parents=True,exist_ok=True)
 def rows(name):
  with (r/name).open(newline='') as f:return list(csv.DictReader(f))
 def save(fig,name):
  fig.savefig(out/(name+'.svg'),bbox_inches='tight');fig.savefig(review/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(fig)
 models=['D2-30k','D2-100k','Trajectory-Lerp05'];short=['D2-30k','D2-100k','Lerp05'];colors=['#cf9b4e','#ba555b','#387f85']
 summary=json.loads((r/'PilotB_summary.json').read_text())['models'];dev=rows('PilotB_dev_per_source.csv');bench=rows('PilotB_MB1_per_dataset.csv');linear=rows('PilotB_Linear23_comparison.csv')
 fig,axes=plt.subplots(1,3,figsize=(11,3.6))
 for ax,key,title in zip(axes,['MV3_source_equal_Dice','MB1_dataset_equal_Dice','muscle_Primus_Dice'],['MV3: 12-source equal Dice','MB1: 7-group equal Dice','Muscle: Primus Dice']):
  vals=[summary[m][key] for m in models];ax.bar(short,vals,color=colors);ax.set_title(title);ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
  for i,v in enumerate(vals):ax.text(i,v+.015,f'{v:.4f}',ha='center',fontsize=9)
 fig.tight_layout();save(fig,'PilotB_fixed_system_comparison')
 names=sorted({(int(x['dataset_id']),x['dataset']) for x in bench});fig,ax=plt.subplots(figsize=(12,4))
 for j,(model,color) in enumerate(zip(models,colors)):
  values=[float(next(x['dice'] for x in bench if x['model']==model and x['dataset']==name)) for _,name in names]
  ax.bar([i+(j-1)*.24 for i in range(7)],values,width=.24,label=short[j],color=color)
 ax.set_xticks(range(7),[n.replace('_',' ') for _,n in names],rotation=22,ha='right');ax.set_ylabel('Mean per-image Dice');ax.set_ylim(0,1);ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();save(fig,'PilotB_MB1_seven_groups')
 names=sorted({x['source_id'] for x in dev});ds=[]
 for s in names:
  d={x['model']:float(x['dice']) for x in dev if x['source_id']==s};ds.append(d[models[2]]-d[models[1]])
 fig,ax=plt.subplots(figsize=(12,3.7));ax.bar(range(12),ds,color=['#387f85' if x>=0 else '#ba555b' for x in ds]);ax.axhline(0,color='#777',lw=.7);ax.set_xticks(range(12),names,rotation=35,ha='right');ax.set_ylabel('Lerp05 - D2-100k Dice');ax.set_title('MV3 full development: paired fixed-seed systems');ax.grid(axis='y',alpha=.2);fig.tight_layout();save(fig,'PilotB_MV3_source_deltas')
 sources=['openorganelle','guay21','muller21','glycolytic_muscle','hela_cell'];human=['OpenOrganelle','Guay','Mueller','Muscle','HeLa'];fig,ax=plt.subplots(figsize=(10,4))
 dl=[];dp=[]
 for s in sources:
  x={v['model']:float(v['dice']) for v in linear if v['source']==s};dl.append(x[models[2]]-x[models[1]])
  group=[v for v in (bench if s in ['glycolytic_muscle','hela_cell'] else dev) if v.get('dataset',v.get('source_id'))==s];x={v['model']:float(v['dice']) for v in group};dp.append(x[models[2]]-x[models[1]])
 ax.bar([i-.17 for i in range(5)],dl,width=.34,label='Linear23: pooled fixed FOVs',color='#cf9b4e');ax.bar([i+.17 for i in range(5)],dp,width=.34,label='Primus: full development/benchmark',color='#387f85');ax.axhline(0,color='#777',lw=.7);ax.set_xticks(range(5),human);ax.set_ylabel('Lerp05 - D2-100k Dice');ax.set_title('Readout directions (different spatial sampling; descriptive comparison)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout();save(fig,'PilotB_readout_directions')
 fig,axes=plt.subplots(1,2,figsize=(8,3.6))
 for ax,key,title,mult in zip(axes,['muscle_Primus_AP_mean_per_plane','muscle_Primus_far_FP_rate'],['Muscle Primus semantic AP','Muscle remote-background FP (%)'],[1,100]):
  values=[summary[m][key]*mult for m in models];ax.bar(short,values,color=colors);ax.set_title(title);ax.grid(axis='y',alpha=.2)
  for i,v in enumerate(values):ax.text(i,v+.015*max(values),f'{v:.4f}',ha='center',fontsize=9)
 fig.tight_layout();save(fig,'PilotB_muscle_AP_FP')
if __name__=='__main__':main()
