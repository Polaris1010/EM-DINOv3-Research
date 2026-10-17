"""CPU-only D1/D2 protocol and image-holdout validation."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 f=ROOT/'experiments/D12';pool=read(f/'mitoverse_pool.json');cem=read(f/'cem_pool.json');p=read(ROOT/'configs/pretrain/D12_protocol.json')
 assert p['ssl_steps']==30000 and p['ssl_milestones']==[10000,30000] and p['global_batch']==48 and p['world_size']==3 and p['batch_per_gpu']==16 and p['extra_losses']==[] and not p['tta']
 one=read(f/'D1_manifest.json');two=read(f/'D2_manifest.json');a=dict(one);b=dict(two)
 assert a.pop('cem_fraction')==0 and b.pop('cem_fraction')==.5 and a.pop('model')=='D1' and b.pop('model')=='D2' and a==b
 assert one['mitoverse_pool_sha256']==sha(f/'mitoverse_pool.json') and one['cem_pool_sha256']==sha(f/'cem_pool.json')
 assert pool['volume_count']==len(pool['volumes'])==153 and pool['source_count']==len(pool['sampling_groups'])==12 and pool['legal_slices']==34143
 denied={v['volume_id'] for v in pool['excluded']};seen=set()
 for row in pool['volumes']:
  vid=row['volume_id'];assert vid not in seen|denied and not vid.startswith('conrad23') and 'lucchi' not in vid and 'openorganelle_jrc_fly' not in vid;seen.add(vid)
  assert row['legal_slices']==sum(b-a for a,b in row['allowed_z'])>0
  for a,b in row['allowed_z']:
   assert 0<=a<b<=row['shape'][0]
   for box in pool['heldout_regions'].get(vid,[]):assert b<=box[0][0] or a>=box[0][1]
 assert cem['images']==sum(g['images'] for g in cem['groups'])==1425433 and len(cem['groups'])==627
 assert cem['images']+sum(g['images'] for g in cem['excluded_collections'])==1592753
 total=sum(math.sqrt(g['images']) for g in cem['groups'])
 for g in cem['groups']:assert g['last_id']-g['first_id']+1==g['images'] and math.isclose(g['probability'],math.sqrt(g['images'])/total,rel_tol=1e-12)
 assert math.isclose(sum(g['probability'] for g in cem['groups']),1,abs_tol=1e-12)
 result={'status':'PASS','matched_protocol':True,'known_MV3_dev_pixels_excluded_from_MitoVerse_SSL':True,'known_test_and_MB1_related_assets_excluded':True,'CEM_unknown_aliases':'UNVERIFIED, not certified independent','MitoVerse_volumes':153,'MitoVerse_sources':12,'CEM_images':1425433,'CEM_nonempty_collections':627,'GPU_used':False}
 (f/'D12_local_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result))
if __name__=='__main__':main()
