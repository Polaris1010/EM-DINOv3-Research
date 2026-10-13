"""Audit all cataloged volumes for descriptive existing-checkpoint evaluation."""
import csv,json,time,collections
from pathlib import Path
import zarr,numpy as np
S=Path(__file__).resolve().parent;R=S.parent
OLD=Path('/opt/data/3_117/tanghr/codex/mitoverse_round6_20260917_044606')
def main():
 rows=list(csv.DictReader((OLD/'data_manifest_v3.csv').open()))
 eligible=[];blocked=[]
 for row in rows:
  try:
   g=zarr.open_group(row['local_path'],mode='r');img=g['img'];lab=g['mito']
   assert img.shape==lab.shape and len(img.shape)==3 and img.dtype==np.uint8
   if 'mask' in g:assert g['mask'].shape==img.shape
   z=img.shape[0]//2;a=np.asarray(img[z]);b=np.asarray(lab[z]);assert a.shape==b.shape
   v=np.asarray(g['mask'][z])!=0 if 'mask' in g else np.ones(a.shape,bool)
   row.update(shape=list(img.shape),mask_present='mask' in g,center_valid_pixels=int(v.sum()),
      center_foreground_pixels=int(((b>0)&v).sum()),
      interpretation='train_exposure' if row['split']=='train' else 'historical_overlap_risk' if row['split']=='excluded' else row['split'],
      mask_semantics='provisional_mask_nonzero_valid' if row['original_dataset_id']=='han24' else 'mask_nonzero_valid_when_present')
   eligible.append(row)
  except Exception as e:blocked.append({'volume_id':row['volume_id'],'reason':repr(e)})
 existing=json.loads((Path('/opt/data/3_117/tanghr/codex/mitoverse_round4_20260909_103239')/'split_mitoverse_probe_v1.json').read_text())
 reused=set(existing['validation']+existing['test1']+existing['test2']+existing['test3'])
 new=[r for r in eligible if r['volume_id'] not in reused]
 payload={'status':'audited','time':time.time(),'catalog_volumes':len(rows),'eligible_volumes':len(eligible),
  'reuse_existing_volumes':len(reused),'new_inference_volumes':len(new),'blocked':blocked,
  'by_source':dict(collections.Counter(r['original_dataset_id'] for r in eligible)),
  'by_interpretation':dict(collections.Counter(r['interpretation'] for r in eligible)),
  'mask_caveat':'han24 results provisional under mask!=0; do not mix into confirmed-label aggregate',
  'no_independence_claim':True}
 (R/'evidence/expanded_coverage.json').write_text(json.dumps(payload,indent=2))
 (R/'data/expanded_eval_manifest.json').write_text(json.dumps(new,indent=2))
 (R/'data/expanded_all_manifest.json').write_text(json.dumps(eligible,indent=2))
 print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
