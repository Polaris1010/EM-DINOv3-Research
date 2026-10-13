"""Expanded existing-model comparison on one GPU alongside new-model probes."""
import csv,json,time,statistics
from pathlib import Path
from settings import BASE
from guard import save
OLD=Path('/opt/data/3_117/tanghr/codex/mitoverse_round6_20260917_044606')

def aggregate():
 rows=[]
 for shard in (0,1):
  d=json.loads((OLD/f'jobs/mitonet_M0_shard{shard}_full/result.json').read_text())
  for vid,v in d['volumes'].items():rows.append({'model':'MitoNet','seed':0,'volume_id':vid,'source':v['source'],'category':v['split'],**{k:v[k] for k in ('tp','fp','fn','tn','dice','precision','recall')}})
 for seed in range(3):
  for suffix in ('validation','legacy'):
   d=json.loads((OLD/f'jobs/masked_P1_seed{seed}_{suffix}/metrics.json').read_text())
   for vid,v in d['volumes'].items():rows.append({'model':'P1','seed':seed,'volume_id':vid,'source':v['source'],'category':'validation' if suffix=='validation' else 'legacy_dev_test',**{k:v[k] for k in ('tp','fp','fn','tn','dice','precision','recall')}})
 for model,seed in [('MitoNet',0)]+[('P1',s) for s in range(3)]:
  d=json.loads((BASE/f'jobs/expanded_{model}_seed{seed}/result.json').read_text())
  assert d['status']=='complete' and d['full']
  for vid,v in d['volumes'].items():rows.append({'model':model,'seed':seed,'volume_id':vid,'source':v['source'],
   'category':'provisional_han24_mask' if v['source']=='han24' else v['interpretation'],**{k:v[k] for k in ('tp','fp','fn','tn','dice','precision','recall')}})
 expected={r['volume_id'] for r in json.loads((BASE.parent/'data/expanded_all_manifest.json').read_text())}
 for model,seed in [('MitoNet',0)]+[('P1',s) for s in range(3)]:
  rr=[r for r in rows if r['model']==model and r['seed']==seed]
  assert len(rr)==len(expected) and {r['volume_id'] for r in rr}==expected
 for vid in expected:
  rr=[r for r in rows if r['volume_id']==vid]
  assert len({(r['tp']+r['fn'],r['fp']+r['tn']) for r in rr})==1
 dest=BASE.parent/'evidence'
 with (dest/'expanded_comparison_volume.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=[]
 for source,category,model in sorted({(r['source'],r['category'],r['model']) for r in rows}):
  rr=[r for r in rows if (r['source'],r['category'],r['model'])==(source,category,model)]
  summary.append({'source':source,'category':category,'model':model,'volumes':len({r['volume_id'] for r in rr}),
    **{m:statistics.mean(r[m] for r in rr) for m in ('dice','precision','recall')}})
 save(dest/'expanded_comparison_summary.json',{'status':'complete','source_rows':summary,'volumes_per_model':len(expected),
  'p1_head_seeds':3,'time':time.time(),'interpretation':'Source macro metrics; separate prior train, historical overlap, legacy validation/development, and provisional Han mask categories. No pristine test claim.'})

def run_expanded(guard,gpu=3):
 completed=[]
 try:
  audit=json.loads((BASE.parent/'evidence/expanded_coverage.json').read_text())
  if audit['blocked']:raise RuntimeError(f"Expanded coverage blocked: {audit['blocked']}")
  save(BASE/'expanded_status.json',{'status':'running','gpu':gpu,'new_volumes':audit['new_inference_volumes'],'started':time.time()})
  for model,seed in [('MitoNet',0),('P1',0),('P1',1),('P1',2)]:
   for smoke in (True,False):
    name=f'expanded_{model}_seed{seed}'+('_smoke' if smoke else '')
    path=BASE/'jobs'/name
    if not path.exists():
     args=['--model',model,'--seed',str(seed)]+(['--smoke'] if smoke else [])
     path=guard.job(name,gpu,'expanded_eval.py',args)
    result=json.loads((path/'result.json').read_text())
    assert result['status']=='complete'
    if not smoke:assert len(result['volumes'])==audit['new_inference_volumes']
   completed.append({'model':model,'seed':seed})
   save(BASE/'expanded_status.json',{'status':'running','completed':completed,'gpu':gpu,'updated':time.time()})
  aggregate();save(BASE/'expanded_status.json',{'status':'completed','completed':completed,'ended':time.time()})
 except Exception as exc:
  guard.stop_all(repr(exc));save(BASE/'expanded_status.json',{'status':'blocked','reason':repr(exc),'completed':completed});raise
