from common import *
import csv,shutil,statistics,time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def csvwrite(name,rows):
 with (BASE/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def meanstd(x):return dict(mean=float(np.mean(x)),sd=float(np.std(x,ddof=1)),values=x)
def main():
 norm=json.loads((BASE/'normalization_selection.json').read_text())['selected'];allrows=[];sr=[];runs={}
 for pid in ['P0','P1','P2']:
  for s in range(3):
   out=BASE/'jobs'/f'evaluate_{pid}_seed{s}_{norm}';r=json.loads((out/'metrics.json').read_text());assert r['status']=='complete' and r['full'];runs[pid,s]=r
   for vid,v in r['volumes'].items():allrows.append(dict(id=pid,seed=s,volume_id=vid,**{k:x for k,x in v.items() if k!='seam_diagnostic'}))
   for source in r['source_dice']:
    vv=[v for v in r['volumes'].values() if v['source']==source];sr.append(dict(id=pid,seed=s,source=source,split=vv[0]['split'],**{k:float(np.mean([v[k] for v in vv])) for k in ['dice','iou','precision','recall']}))
   ordered=sorted(r['volumes'],key=lambda v:r['volumes'][v]['dice']);folder=BASE/'failure_cases'/f'{pid}_seed{s}';folder.mkdir(parents=True,exist_ok=True)
   for label,v in [('best',ordered[-1]),('worst',ordered[0])]:shutil.copy2(out/'previews'/f'{v}.png',folder/f'{label}_{v}.png')
   previews=BASE/'predictions_preview'/f'{pid}_seed{s}';previews.mkdir(parents=True,exist_ok=True)
   for p in (out/'previews').glob('*.png'):shutil.copy2(p,previews/p.name)
 csvwrite('volume_metrics.csv',allrows);csvwrite('source_metrics.csv',sr)
 # Each run evaluates exactly the same volume IDs, grid counts, and foreground counts.
 ref=runs['P0',0]['volumes']
 for run in runs.values():
  assert set(run['volumes'])==set(ref)
  for v,r in run['volumes'].items():
   assert sum(r[k] for k in ['tp','fp','fn','tn'])==sum(ref[v][k] for k in ['tp','fp','fn','tn'])
   assert r['tp']+r['fn']==ref[v]['tp']+ref[v]['fn']
 groups=['validation','test1','test2','test3','all_test'];summary={}
 for group in groups:
  summary[group]={}
  for pid in ['P0','P1','P2']:
   vals=[]
   for s in range(3):
    rr=[r for r in sr if r['id']==pid and r['seed']==s and (r['split'].startswith('test') if group=='all_test' else r['split']==group)];vals.append(float(np.mean([r['dice'] for r in rr])))
   summary[group][pid]=meanstd(vals)
  for a,b in [('P1','P0'),('P2','P1')]:summary[group][a+'-'+b]=meanstd([x-y for x,y in zip(summary[group][a]['values'],summary[group][b]['values'])])
 exposure={}
 for pid in ['P0','P1','P2']:
  exposure[pid]={}
  for status in ['yes','no','uncertain']:
   vals=[]
   for s in range(3):
    rr=[r for r in allrows if r['id']==pid and r['seed']==s and r['split'].startswith('test') and r['ssl_pretrain_seen']==status]
    sources=set(r['source'] for r in rr);vals.append(float(np.mean([np.mean([r['dice'] for r in rr if r['source']==src]) for src in sources])) if sources else None)
   exposure[pid][status]=meanstd(vals) if all(v is not None for v in vals) else {'available':False,'reason':'No qualifying held-out volume with proven exposure status'}
 contrasts={}
 for a,b in [('P1','P0'),('P2','P1')]:
  per={}
  for source in sorted({r['source'] for r in sr if r['split'].startswith('test')}):
   aa=sorted([r for r in sr if r['source']==source and r['id']==a],key=lambda r:r['seed']);bb=sorted([r for r in sr if r['source']==source and r['id']==b],key=lambda r:r['seed']);per[source]=meanstd([x['dice']-y['dice'] for x,y in zip(aa,bb)])
  d=summary['all_test'][a+'-'+b];positive=sum(v['mean']>0 for v in per.values());collapse=any(np.mean([r[k] for r in sr if r['id']==a and r['split'].startswith('test')])<.5*np.mean([r[k] for r in sr if r['id']==b and r['split'].startswith('test')]) for k in ['precision','recall'])
  supported=all(x>0 for x in d['values']) and positive>len(per)/2 and not collapse
  if a=='P2':supported=supported and d['mean']>d['sd']
  contrasts[a+'-'+b]=dict(per_source=per,positive_sources=positive,total_sources=len(per),precision_recall_collapse=collapse,descriptive_rule_pass=supported,strict_unseen_generalization='not established')
 selected='P2' if contrasts['P2-P1']['descriptive_rule_pass'] else 'P1'
 result=dict(status='complete',normalization=norm,summary=summary,exposure=exposure,contrasts=contrasts,recommended_encoder=selected,strict_unseen_proven=False,limitations=['P0 adapted architecture; storage removal not output-equivalent','P2 preselected on CMCC/CMCT; OpenOrganelle excluded','Current EM image cache not immutable historical exposure manifest','Conrad23 had five validation patches inspected in round1 smoke; current full-volume test is not historically pristine','Test3 unseen-candidate only','3 seeds descriptive, no test-driven tuning','10k budget, no automatic extension'])
 save(BASE/'metrics_summary.json',result)
 fig,axes=plt.subplots(1,2,figsize=(13,5));colors={'P0':'#617fa5','P1':'#279886','P2':'#db983d'}
 for pid in colors:
  ys=[]
  for s in range(3):
   c=json.loads((BASE/'jobs'/f'train_{pid}_seed{s}_{norm}'/'curve.json').read_text());axes[0].plot([r['step'] for r in c],[r['source_macro_dice'] for r in c],color=colors[pid],alpha=.45,label=pid if s==0 else None)
  a=summary['all_test'][pid];axes[1].bar(pid,a['mean'],yerr=a['sd'],color=colors[pid],capsize=5)
 axes[0].set(title='Validation selection curves (subsampled Z)',xlabel='Steps',ylabel='Source-equal Dice');axes[0].legend();axes[1].set(title='Full held-out volumes: 3-seed mean +/- SD',ylabel='Source-equal Dice',ylim=(0,1));fig.tight_layout();fig.savefig(BASE/'learning_curves.png',dpi=160);plt.close(fig)
 lines=['# MitoVerse P0-P2 冻结表征验证','',f'统一归一化：{norm}；batch=8，256 crop，三个 seed，每次10,000步。最终指标覆盖完整原始体积，source等权。','', '| 模型 | held-out source Dice 均值 | seed标准差 |','|---|---:|---:|']
 for p in ['P0','P1','P2']:
  r=summary['all_test'][p];lines.append(f"| {p} | {r['mean']:.6f} | {r['sd']:.6f} |")
 lines+=['','## 七项结论','',f"1. P1 相对 P0 的预注册描述性规则：{contrasts['P1-P0']['descriptive_rule_pass']}。P0存在结构转换影响，不能解释为纯预训练因果效应。",f"2. P2 相对 P1 的规则：{contrasts['P2-P1']['descriptive_rule_pass']}。",'3. 逐来源改善见下表和 source_metrics.csv。','4. 没有足够证据确认严格 SSL 未见来源，不能证明完全未见域收益。',f'5. 建议进入下一轮分阶段微调：{selected}，本轮不自动启动。','6. 全来源分组避免同来源跨split；P2筛选数据OpenOrganelle排除。历史预训练曝光证据存在不确定性；conrad23在第一轮smoke中曾用于5个验证patch，不能称为历史上从未查看过的封存测试。见split_audit.md和pretraining_exposure.csv。','7. 根据来源差值和失败案例决定是否分阶段解冻；若P2证据不足停止额外SSL搜索，使用P1。','', '| 对比 / source | Dice差值 | seed标准差 |','|---|---:|---:|']
 for contrast,c in contrasts.items():
  for source,d in c['per_source'].items():lines.append(f"| {contrast} / {source} | {d['mean']:+.6f} | {d['sd']:.6f} |")
 lines+=['','![学习曲线](learning_curves.png)','','多处设置依据validation选定；test未参与参数选择。完整概率图与二值预测在jobs/evaluate_*/predictions。failure_cases按完整体积Dice选择最好与最差，展示中央切片。资源日志记录所有任务合计RSS与各卡显存；逐运行时间和峰值allocated显存见jobs/train_*/result.json。','','本轮完成后暂停：不自动延长训练、不启动Gram、边界、2.5D或实例分割。']
 (BASE/'final_report.md').write_text('\n'.join(lines));print('FINAL_REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
