import json,csv,hashlib
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
valid=[];quality=[]
for i in range(20):
 a=np.load(BASE/'comparison_01'/f'mitonet_{i:02}.npz');r=a['raw'];ok=int(r.max())>int(r.min());quality.append(dict(index=i,raw_min=int(r.min()),raw_max=int(r.max()),std=float(r.std()),truth_foreground=int((a['truth']>0).sum()),nonconstant=ok))
 if ok:valid.append(i)
(BASE/'patch_quality.json').write_text(json.dumps(quality,indent=2))
def agg(rows):
 tp,fp,fn,tn=[sum(r[k] for r in rows) for k in ['tp','fp','fn','tn']];itp,ifp,ifn=[sum(r[k] for r in rows) for k in ['instance_tp','instance_fp','instance_fn']];den=itp+.5*(ifp+ifn)
 return dict(n=len(rows),dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 1.,pq=sum(r['iou_sum'] for r in rows)/den if den else 1.,f1_50=itp/den if den else 1.,tp=tp,fp=fp,fn=fn,tn=tn,instance_tp=itp,instance_fp=ifp,instance_fn=ifn)
results={}
for folder in ['comparison_01','trained_probe_01','native224_probe','light_decoder_01']:
 for f in (BASE/folder).glob('*.json'):
  d=json.loads(f.read_text())
  if 'rows' not in d:continue
  rows=d['rows'];vr=[r for r in rows if r['index'] in valid]
  results[f.stem]=dict(source=str(f.relative_to(BASE)),all20=agg(rows),nonconstant=agg(vr),per_domain={v:agg([r for r in vr if r['volume']==v]) for v in sorted({r['volume'] for r in vr})})
(BASE/'metrics_summary.json').write_text(json.dumps(results,indent=2))
with (BASE/'comparison_table.csv').open('w',newline='') as f:
 fields=['model',*next(iter(results.values()))['nonconstant'].keys()];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for model,d in results.items():w.writerow(dict(model=model,**d['nonconstant']))
for model,d in results.items():print(model,d['nonconstant'])
