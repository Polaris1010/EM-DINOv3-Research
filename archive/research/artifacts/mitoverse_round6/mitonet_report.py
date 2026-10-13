"""Aggregate complete per-volume evidence only; label partial coverage explicitly."""
import json,csv,time
from pathlib import Path
from settings import BASE,R4
from guard import save

def main():
    split=json.loads((R4/'split_mitoverse_probe_v1.json').read_text());mito={}
    for shard in [0,1]:
        p=BASE/'jobs'/f'mitonet_M0_shard{shard}_full'/'volume_metrics_partial.json'
        if p.exists():mito.update(json.loads(p.read_text()))
    rows=[]
    for vid,v in mito.items():
        for variant,metric in [('MitoNet_M0_semantic',v),('MitoNet_M0_postprocessed',v['postprocessed_binary'])]:
            rows.append(dict(model=variant,seed='',volume_id=vid,source=v['source'],split=v['split'],**{k:metric[k] for k in ['dice','precision','recall','tp','fp','fn','tn']}))
    for seed in range(3):
        p=BASE/'jobs'/f'masked_P1_seed{seed}_validation'/'metrics.json'
        if p.exists():
            d=json.loads(p.read_text());assert d['full']
            for vid,v in d['volumes'].items():rows.append(dict(model='P1_mask_corrected',seed=seed,volume_id=vid,source=v['source'],split='validation',**{k:v[k] for k in ['dice','precision','recall','tp','fp','fn','tn']}))
    p=BASE/'mitonet_comparison_volume.csv';tmp=p.with_suffix('.tmp')
    with tmp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=['model','seed','volume_id','source','split','dice','precision','recall','tp','fp','fn','tn']);w.writeheader();w.writerows(rows)
    tmp.replace(p)
    summary=[]
    for model,seed,sp in sorted({(r['model'],str(r['seed']),r['split']) for r in rows}):
        selected=[r for r in rows if r['model']==model and str(r['seed'])==seed and r['split']==sp]
        expected=set(split['validation']) if sp=='validation' else set(sum([split[k] for k in ['test1','test2','test3']],[]))
        sources={s:sum(r['dice'] for r in selected if r['source']==s)/sum(r['source']==s for r in selected) for s in {r['source'] for r in selected}}
        complete={r['volume_id'] for r in selected}==expected
        summary.append(dict(model=model,seed=seed,split=sp,complete=complete,volumes=len(selected),expected_volumes=len(expected),source_dice=sources,source_macro_dice=sum(sources.values())/len(sources) if complete else None))
    save(BASE/'mitonet_comparison_summary.json',dict(time=time.time(),rows=summary,interpretation='M0 native-grid 2D foreground comparison; not official 3D instance benchmark; no target-domain tuning for MitoNet; corrected P1 target supervised; partial coverage is not an aggregate score'))

if __name__=='__main__':main()
