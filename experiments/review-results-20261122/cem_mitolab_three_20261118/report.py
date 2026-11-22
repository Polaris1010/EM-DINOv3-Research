from pathlib import Path
import csv,json,statistics as st
from collections import defaultdict
from prepare import X
models=['P0','B100','Barrier100']
runs=[json.loads((X/m/'complete.json').read_text()) for m in models]
for key in ['steps','seed','sampling','initial_sha256','manifest_sha256','source_sha256']:
    assert len({r[key] for r in runs})==1,key
out=[]
for model in models:
    for split in ['cem_validation','mb1']:
        rows=json.loads((X/model/(split+'.json')).read_text())
        groups=defaultdict(list)
        for row in rows: groups[row['group']].append(row)
        for group,items in groups.items():
            result=dict(split=split,group=group,model=model,n=len(items))
            for metric in ('dice','iou','precision','recall'):
                vals=[r[metric] for r in items if r[metric] is not None]
                result[metric]=st.mean(vals) if vals else None
            out.append(result)
with (X/'by_group.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
lines=['# CEM-MitoLab matched frozen Primus comparison','','Seed 20261014, 40000 updates. Equal collection or benchmark group means of image-level Dice. Empty GT and empty prediction score 1. Official MitoNet uses its released architecture.','','| Evaluation | P0 | B100 | Barrier100 | B100 minus P0 | Barrier100 minus B100 |','|---|---:|---:|---:|---:|---:|']
def table(label,v):
    lines.append('| '+label+' | '+' | '.join(f'{a:.4f}' for a in [*v,v[1]-v[0],v[2]-v[1]])+' |')
for split in ['cem_validation','mb1']:
    subset=[r for r in out if r['split']==split]
    table(split,[st.mean(r['dice'] for r in subset if r['model']==m) for m in models])
    if split=='mb1':
        for g in sorted({r['group'] for r in subset}):
            table(g,[next(r['dice'] for r in subset if r['model']==m and r['group']==g) for m in models])
ref=X.parent/'mb1_official_benchmark_20261015/MB1_per_dataset_model_means.csv'
with ref.open() as f: official=[float(r['dice']) for r in csv.DictReader(f) if r['model']=='Official-MitoNet-v1']
assert len(official)==7
lines+=['',f'Official MitoNet historical MB-1 macro Dice: {st.mean(official):.4f}.','',
        'The B100 to Barrier100 difference includes SSL data and sampling changes in addition to the barrier. MB-1 was previously exposed in method development. One downstream seed was used.']
(X/'final_report.md').write_text('\n'.join(lines)+'\n')
