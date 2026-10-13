"""Package completed MV-2 evidence; no inference or training is performed here."""
from pathlib import Path
import csv,json,time,hashlib,tarfile,statistics,math,collections

R=Path(__file__).resolve().parents[1]
OLD=R.parent/'mv_generalist_stage1_20261013'
MODELS=['P0','B100k','B150k'];SEEDS=[20261013,20261014,20261015]
METRICS=['dice','precision','recall','hd95_nm','nsd_16nm']
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8<<20):h.update(b)
 return h.hexdigest()
def write(p,j):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(j,indent=2,ensure_ascii=False));q.replace(p)
def csvout(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def avg(v):
 v=[x for x in v if x is not None]
 return statistics.mean(v) if v else None
def sd(v):return statistics.stdev(v) if len(v)>1 and all(x is not None and math.isfinite(x) for x in v) else None

def main():
 while True:
  p=R/'STATUS.json'
  status=read(p) if p.exists() else {}
  if status.get('status')=='failed':raise RuntimeError('Validation failed; results are not complete')
  if status.get('status')=='complete':break
  time.sleep(30)
 m=read(R/'MV_dev_validation_manifest.json');assert m['tta']=='OFF' and not m['training_permitted']
 unavailable=read(R/'evidence/cache_completion.json')['unusable'];excluded={x['id'] for x in unavailable}
 expected={(e['id'],model,seed) for e in m['entries'] if e['id'] not in excluded for model in MODELS for seed in SEEDS}
 rows=[read(p) for p in sorted(R.glob('runs/*/seed*/*_metrics.json'))]
 actual={(r['volume_id'],r['model'],r['seed']) for r in rows}
 assert actual==expected and len(rows)==len(expected),'Missing or duplicate model/seed/volume rows'
 assert all(r['tta']=='OFF' and not r['head_supervised_exposed'] for r in rows)
 reused=[r for r in rows if r['result_origin']=='MV1_direct_reuse_NO_RERUN']
 assert len(reused)==36
 for r in reused:
  p=OLD/'runs'/r['model']/f"seed{r['seed']}"/'per_volume_metrics.csv'
  previous=next(x for x in csv.DictReader(p.open()) if x['volume_id']==r['volume_id'])
  assert all(r[k]==float(previous[k]) for k in METRICS)
  assert all(r[k]==int(previous[k]) for k in ['tp','fp','fn','tn'])
  assert sha(p)==r['old_metric_csv_sha256']
 for r in rows:
  tp,fp,fn,tn=[r[k] for k in ['tp','fp','fn','tn']]
  assert r['dice']==(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.)
  assert r['valid_voxels']==tp+fp+fn+tn
 volume=[]
 for vid in sorted({r['volume_id'] for r in rows}):
  for model in MODELS:
   rs=[r for r in rows if r['volume_id']==vid and r['model']==model]
   assert len(rs)==3
   item={'volume_id':vid,'source_id':rs[0]['source_id'],'model':model,'n_seeds':3}
   for k in METRICS:item[k+'_mean']=avg([r[k] for r in rs]);item[k+'_sd_seeds']=sd([r[k] for r in rs])
   volume.append(item)
 csvout(R/'MV_volume_comparison.csv',volume)
 source_rows=list(csv.DictReader((R/'MV_source_comparison.csv').open()))
 for r in source_rows:
  vs=[v for v in volume if v['source_id']==r['source_id'] and v['model']==r['model']]
  for k in METRICS:r[k+'_sd_volumes_seed_averaged']=sd([v[k+'_mean'] for v in vs])
 csvout(R/'MV_source_comparison.csv',source_rows)
 verification={'status':'PASS','tta':'OFF','rows':len(rows),'new_inference_rows':len(rows)-36,
  'MV1_direct_reused_rows':36,'old_metrics_exactly_unchanged':True,
  'completed_dev_regions':len({r['volume_id'] for r in rows}),'completed_sources':len({r['source_id'] for r in rows}),
  'unusable':unavailable,'no_training':True,'sd_definition':'sd_seeds: sample SD across the three decoder seeds; sd_volumes_seed_averaged: descriptive SD across seed-averaged volume/ROI scores, not independent acquisition uncertainty'}
 write(R/'evidence/delivery_verification.json',verification)
 files=[p for p in R.rglob('*') if p.is_file() and p.suffix in {'.json','.csv','.md','.py','.sh','.yaml'} and not any(part in {'cache','__pycache__'} for part in p.relative_to(R).parts)]
 write(R/'evidence/artifact_hashes.json',{str(p.relative_to(R)):sha(p) for p in files})
 archive=R/'MV2_delivery.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for p in files+[R/'evidence/artifact_hashes.json']:tar.add(p,arcname=str(p.relative_to(R)))
 write(R/'DELIVERY_READY.json',{'status':'complete','archive':str(archive),'archive_sha256':sha(archive),'verification':verification})
 print('DELIVERY_READY',json.dumps(verification),flush=True)
if __name__=='__main__':main()
