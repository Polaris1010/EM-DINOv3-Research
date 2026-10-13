"""Compare cached execution to direct official CPU functions, including mutation."""
from pathlib import Path
import sys,json,hashlib
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import numpy as np,torch
from monai.metrics import compute_hausdorff_distance as hd,compute_surface_dice as nsd
from metric_cache import install
from mv_data import manifest,open_cache
torch.set_num_threads(2)
cached_hd,cached_nsd,cache=install(hd,nsd)
def measures(h,n,p,g,spacing):
 return [h(p,g,include_background=True,percentile=95,spacing=spacing)]+[n(p,g,class_thresholds=[t],include_background=True,spacing=spacing) for t in [8,16,32]]+[h(p,g,include_background=True,percentile=95,spacing=[1]*len(spacing)),n(p,g,class_thresholds=[1],include_background=True,spacing=[1]*len(spacing))]
checks=[]
for row in manifest()['entries']:
 if row['supervised_role']!='dev':continue
 im,gt=open_cache(row);pred=np.load(R/'runs/P0/seed20261013'/(row['id']+'_prediction.npy'),mmap_mode='r')
 z=min(32,gt.shape[0]);y=min(160,gt.shape[1]);x=min(160,gt.shape[2]);start=(gt.shape[0]//2-z//2,gt.shape[1]//2-y//2,gt.shape[2]//2-x//2)
 slices=tuple(slice(s,s+n) for s,n in zip(start,[z,y,x]))
 p=torch.from_numpy(np.asarray(pred[slices],dtype=bool).copy())[None,None];g=torch.from_numpy(np.asarray(gt[slices],dtype=bool).copy())[None,None]
 assert p.any() and g.any(),row['id']
 direct=measures(hd,nsd,p,g,row['spacing_zyx_nm']);memo=measures(cached_hd,cached_nsd,p,g,row['spacing_zyx_nm'])
 assert all(torch.equal(a,b) for a,b in zip(direct,memo)),row['id']
 # Reusing identical tensor objects after mutation must invalidate affected entries.
 p[0,0,0,0,0]=~p[0,0,0,0,0]
 assert torch.equal(hd(p,g,include_background=True,percentile=95,spacing=row['spacing_zyx_nm']),cached_hd(p,g,include_background=True,percentile=95,spacing=row['spacing_zyx_nm']))
 checks.append({'volume_id':row['id'],'roi_shape':list(p.shape[2:]),'six_metrics_bit_exact':True,'mutation_guard_pass':True})
print('CACHE_COUNTS',cache.hits,cache.misses,'KEYS',list(cache.entries),flush=True)
assert cache.hits>=16 and cache.misses>=8
receipt={'status':'PASS','backend':'same official MONAI functions and SciPy CPU EDT','checks':checks,'hits':cache.hits,'misses':cache.misses,'cache_source_sha256':hashlib.sha256((R/'src/metric_cache.py').read_bytes()).hexdigest(),'applied_to':'later seeds only; first-seed direct scores are reused'}
(R/'evidence/metric_memoization_verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
