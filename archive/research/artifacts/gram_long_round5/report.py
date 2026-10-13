import os,json,csv,sys,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parent;R4=BASE.parent/'mitoverse_round4_20260909_103239'
def main():
 prefix=sys.argv[1];rows=[];by={}
 for name,paths in [('P1',[R4/'jobs'/f'evaluate_P1_seed{s}_N2'/'metrics.json' for s in range(3)]),('G',[BASE/'jobs'/f'{prefix}_G_30000_head{s}_eval'/'metrics.json' for s in range(3)]),('C',[BASE/'jobs'/f'{prefix}_C_30000_head{s}_eval'/'metrics.json' for s in range(3)]),('X1',[BASE/'jobs'/f'{prefix}_X1_seed{s}_eval'/'metrics.json' for s in range(3)])]:
  by[name]={}
  for seed,p in enumerate(paths):
   r=json.load(open(p));assert r['status']=='complete';vs={k:v for k,v in r['volumes'].items() if v['split'].startswith('test')};assert vs
   sources={}
   for source in sorted({v['source'] for v in vs.values()}):
    selected=[v for v in vs.values() if v['source']==source];m={k:statistics.mean(v[k] for v in selected) for k in ['dice','iou','precision','recall']};sources[source]=m;rows.append(dict(model=name,seed=seed,source=source,**m))
   assert len(sources)==6;by[name][seed]=dict(sources=sources,dice=statistics.mean(v['dice'] for v in sources.values()))
 with (BASE/'comparison_source_seed.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 report={k:dict(mean_dice=statistics.mean(v['dice'] for v in d.values()),sd_dice=statistics.stdev(v['dice'] for v in d.values()),seed_dice=[d[s]['dice'] for s in range(3)]) for k,d in by.items()}
 pairs={}
 for x,y in [('G','C'),('G','P1'),('C','P1'),('X1','P1')]:
  delta=[by[x][s]['dice']-by[y][s]['dice'] for s in range(3)];sources={src:statistics.mean(by[x][s]['sources'][src]['dice']-by[y][s]['sources'][src]['dice'] for s in range(3)) for src in by[x][0]['sources']}
  pairs[f'{x}-{y}']=dict(mean_delta=statistics.mean(delta),seed_deltas=delta,positive_seeds=sum(v>0 for v in delta),source_deltas=sources,positive_sources=sum(v>0 for v in sources.values()))
 gc=pairs['G-C'];retention=dict(dice_gate=gc['mean_delta']>=.005,seed_gate=gc['positive_seeds']>=2,source_gate=gc['positive_sources']>=4,precision_recall='Review comparison_source_seed.csv for collapse; no automatic qualitative pass',deployment_gain=report['G']['mean_dice']>report['P1']['mean_dice'])
 (BASE/'comparison_summary.json').write_text(json.dumps(dict(models=report,paired=pairs,gram_retention=retention,scope='Development holdouts; three head seeds, one SSL seed; no claim of independent SSL replication'),indent=2))
 lines=['# 第五轮实验结果','', '| 模型 | 源等权测试 Dice，均值 ± SD |','|---|---:|']
 for k,v in report.items():lines.append(f"| {k} | {v['mean_dice']:.6f} ± {v['sd_dice']:.6f} |")
 lines+=['','## 配对差值','']
 for k,v in pairs.items():lines.append(f"- {k}: {v['mean_delta']:+.6f}；{v['positive_seeds']}/3 seeds 为正，{v['positive_sources']}/6 来源改善。")
 lines+=['','仅统计 test 分组，validation 不混入测试均值。逐来源 precision/recall 见 comparison_source_seed.csv。G>C 且 G<P1 只能解释为缓解退化。三 head seeds 不是三次 SSL 复现。','', '本轮结束，等待结果审核；未运行 X2/X3。']
 (BASE/'round5_results.md').write_text('\n'.join(lines));print(json.dumps(report))
if __name__=='__main__':main()
