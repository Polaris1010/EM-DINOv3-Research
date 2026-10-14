"""CPU-only validation of matched, already-produced MV4 artifacts; no inference."""
import collections,csv,datetime,hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def validate(root=ROOT):
 folder=root/'experiments/MV-4';c=read(root/'configs/evaluation/MV-4_protocol.json')
 m=read(root/c['manifest']);expected={r['id'] for r in m['entries'] if r['supervised_role']=='dev'}
 assert len(expected)==38
 rows=list(csv.DictReader((folder/'MV4_per_region_metrics.csv').open(encoding='utf-8')))
 assert len(rows)==266
 groups=collections.defaultdict(list)
 for r in rows:groups[r['model']].append(r)
 assert set(groups)=={'G-P0','G-B100','Official-MitoNet-v1'}
 for model,rr in groups.items():
  assert {r['region_id'] for r in rr}==expected and len(rr)==(38 if model=='Official-MitoNet-v1' else 114)
  keys=[(r['region_id'],r.get('seed','')) for r in rr];assert len(keys)==len(set(keys))
 for r in rows:
  tp,fp,fn,tn=[int(r[k]) for k in ['tp','fp','fn','tn']]
  assert tp+fp+fn+tn==int(r['valid_voxels'])
  assert tp+fn==int(r['gt_foreground_voxels'])
  assert int(r['prediction_foreground_voxels'])==tp+fp
  dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
  assert math.isclose(float(r['dice']),dice,abs_tol=1e-12)
  assert math.isclose(float(r['predicted_foreground_fraction']),(tp+fp)/(tp+fp+fn+tn),abs_tol=1e-12)
  assert r['manifest_sha256']==c['manifest_sha256'] and r['tta'] in ['False','false']
 old=list(csv.DictReader((root/'experiments/MV-3/v2/MV3_per_region_metrics.csv').open(encoding='utf-8')))
 lookup={(r['model'],r['seed'],r['region_id']):r for r in old}
 for model in ['G-P0','G-B100']:
  for r in groups[model]:
   original=lookup[model,r['seed'],r['region_id']]
   for k in ['dice','precision','recall','hd95_nm','nsd_16nm','head_sha256','valid_voxels','valid_mask_policy']:
    assert r[k]==original[k],f'Changed reused DINOv3 result {k}'
 summary=read(folder/'MV4_summary.json');sources=sorted({r['source_id'] for r in rows})
 assert len(sources)==12
 for model in groups:
  macro=statistics.mean(statistics.mean(float(r['dice']) for r in groups[model] if r['source_id']==s) for s in sources)
  assert math.isclose(macro,summary['source_macro'][model]['dice'],abs_tol=1e-12)
 status=read(folder/'STATUS.json');assert status['status']=='complete' and status['mv3_inputs_unchanged']
 assets=read(root/'evidence/MV-4/official_assets.json');weight=assets['files']['MitoNet_v1.pth']
 assert weight['publisher_md5_verified'] and weight['publisher_md5']==c['official_weight_md5'] and weight['bytes']==c['official_weight_bytes']
 assert len(weight['sha256'])==64
 psha=hashlib.sha256((root/'configs/evaluation/MV-4_protocol.json').read_bytes()).hexdigest()
 for r in groups['Official-MitoNet-v1']:
  assert r['checkpoint_sha256']==weight['sha256'] and r['protocol_sha256']==psha
  assert r['physical_gpu'] in ['1','2','3']
  j=read(folder/'runs/Official-MitoNet-v1'/(r['region_id']+'_inference.json'))
  assert j['status']=='complete' and j['all_native_z_inferred'] and j['gt_used_for_inference'] is False
  assert j['protocol_sha256']==psha and j['checkpoint_sha256']==weight['sha256']
 before=read(root/'evidence/MV-4/mv3_inputs_before.json');after=read(root/'evidence/MV-4/mv3_inputs_after.json')
 assert before==after
 result={'status':'PASS','verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':266,'mitonet_rows':38,'reused_dinov3_rows':228,'sources':12,'protected_mv3_inputs_unchanged':True,'publisher_weight_verified':True,'source_macro':summary['source_macro']}
 (folder/'MV4_results_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
 return result
if __name__=='__main__':print(json.dumps(validate(),indent=2))
