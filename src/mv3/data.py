"""Versioned source-aware supervised loader; historic arrays stay read-only."""
import collections,hashlib,json,os
from pathlib import Path
import numpy as np
R=Path(os.environ.get('MV3_ROOT',Path(__file__).resolve().parents[1]))
OLD=R.parent/'mv_generalist_stage1_20261013';MV2=R.parent/'mv_dev_v2_notta_20261013'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(p)
def manifest():return read(R/'MV-Generalist-v1.json')
def is_valid(mask,source):return mask!=0 if source=='han24' else mask==0

def prepare():
 """Reuse existing caches; only materialize missing regions and corrected Han masks."""
 import zarr
 m=manifest();sources={};counts=[]
 for row in sorted(m['entries'],key=lambda r:np.prod([b-a for a,b in r['region_zyx']])):
  out=R/'cache'/row['id'];out.mkdir(parents=True,exist_ok=True);receipt=out/'CACHE.json'
  if receipt.exists():counts.append(read(receipt));continue
  wanted=row['region_zyx'];shape=[b-a for a,b in wanted];chosen=None
  for root in [MV2,OLD]:
   d=root/'cache'/row['volume_id']
   if not (d/'image.npy').exists() or not (d/'label.npy').exists():continue
   metadata=read(d/'QC.json') if (d/'QC.json').exists() else None
   if metadata is None:continue
   bounds=metadata.get('region_zyx')
   if bounds is None:
    previous=read(root/('MV_dev_validation_manifest.json' if root==MV2 else 'MV-Dev-v1.json'))
    original=next(e for e in previous['entries'] if e.get('volume_id',e['id'])==row['volume_id']);bounds=original['region_zyx']
   if all(a<=c<e<=b for (a,b),(c,e) in zip(bounds,wanted)):
    chosen=(d,bounds);break
  if chosen:
   d,bounds=chosen;paths={k:str(d/(k+'.npy')) for k in ['image','label']}
   offsets=[[c-a,e-a] for (a,b),(c,e) in zip(bounds,wanted)]
   assert list(np.load(paths['image'],mmap_mode='r').shape)==[b-a for a,b in bounds]
  else:
   group=zarr.open_group(row['image_path'],mode='r');paths={k:str(out/(k+'.npy')) for k in ['image','label']}
   im=np.lib.format.open_memmap(paths['image'],mode='w+',dtype=np.uint8,shape=tuple(shape))
   lab=np.lib.format.open_memmap(paths['label'],mode='w+',dtype=np.uint8,shape=tuple(shape))
   for z in range(0,shape[0],8):
    key=(slice(wanted[0][0]+z,min(wanted[0][1],wanted[0][0]+z+8)),slice(*wanted[1]),slice(*wanted[2]))
    im[z:z+8]=np.asarray(group['img'][key]);lab[z:z+8]=np.asarray(group['mito'][key])>0
    atomic(R/'cache_progress.json',{'region':row['id'],'z':z,'total':shape[0]})
   im.flush();lab.flush();del im,lab;offsets=[[0,n] for n in shape]
  valid_path=None;valid_z=list(range(shape[0]));valid_voxels=int(np.prod(shape))
  if row['mask_present']:
   group=zarr.open_group(row['image_path'],mode='r');valid_path=str(out/'valid.npy')
   valid=np.lib.format.open_memmap(valid_path,mode='w+',dtype=np.uint8,shape=tuple(shape));valid_z=[];valid_voxels=0
   for z in range(0,shape[0],8):
    key=(slice(wanted[0][0]+z,min(wanted[0][1],wanted[0][0]+z+8)),slice(*wanted[1]),slice(*wanted[2]))
    v=is_valid(np.asarray(group['mask'][key]),row['source_id']);valid[z:z+len(v)]=v
    valid_voxels+=int(v.sum());valid_z.extend(z+i for i,a in enumerate(v) if a.any())
   valid.flush();del valid
  assert valid_voxels>0 and valid_z,'No usable annotation; do not silently drop this region'
  j={'id':row['id'],'volume_id':row['volume_id'],'source_id':row['source_id'],'supervised_role':row['supervised_role'],'paths':paths,'offsets':offsets,'shape_zyx':shape,'valid_path':valid_path,'valid_z':valid_z,'valid_voxels':valid_voxels,'valid_mask_policy':row['valid_mask_policy'],'region_zyx':wanted,'manifest_sha256':sha(R/'MV-Generalist-v1.json'),'reused_read_only':bool(chosen)}
  atomic(receipt,j);counts.append(j);print('CACHE_READY',row['id'],j['valid_voxels'],flush=True)
 atomic(R/'cache_complete.json',{'status':'PASS','regions':len(counts),'manifest_sha256':sha(R/'MV-Generalist-v1.json')})
 atomic(R/'MV3_actual_supervision.json',{'rows':counts,'by_source':{s:{role:sum(j['valid_voxels'] for j in counts if j['source_id']==s and j['supervised_role']==role) for role in ['train','dev']} for s in sorted({j['source_id'] for j in counts})}})

def open_arrays(row):
 j=read(R/'cache'/row['id']/'CACHE.json');assert j['manifest_sha256']==sha(R/'MV-Generalist-v1.json')
 key=tuple(slice(a,b) for a,b in j['offsets'])
 im=np.load(j['paths']['image'],mmap_mode='r')[key];lab=np.load(j['paths']['label'],mmap_mode='r')[key]
 valid=np.load(j['valid_path'],mmap_mode='r') if j['valid_path'] else None
 assert list(im.shape)==list(lab.shape)==j['shape_zyx']
 return im,lab,valid,j

class Samples:
 def __init__(self,seed):
  self.seed=seed;self.rows=[r for r in manifest()['entries'] if r['supervised_role']=='train'];self.sources=sorted({r['source_id'] for r in self.rows})
  self.handles={r['id']:open_arrays(r) for r in self.rows}
 def sample(self,step,device='cuda',forced_source=None):
  import torch
  rng=np.random.default_rng(np.random.SeedSequence([self.seed,step]));source=forced_source or self.sources[int(rng.integers(len(self.sources)))]
  rows=[r for r in self.rows if r['source_id']==source];weights=[len(self.handles[r['id']][3]['valid_z']) for r in rows]
  index=int(rng.integers(sum(weights)));k=int(np.searchsorted(np.cumsum(weights),index,side='right'));row=rows[k]
  im,lab,valid,j=self.handles[row['id']];z=j['valid_z'][index-sum(weights[:k])];a=im[z];b=lab[z];v=valid[z].astype(bool) if valid is not None else None;H,W=a.shape
  fg=bool(rng.random()<.5);ys=xs=None
  if fg:ys,xs=np.nonzero((b>0)&v if v is not None else b>0)
  if ys is None or len(ys)==0:
   fg=False
   if v is not None:ys,xs=np.nonzero(v)
  if ys is not None and len(ys):
   i=int(rng.integers(len(ys)));y=int(np.clip(ys[i]-128,0,max(0,H-256)));x=int(np.clip(xs[i]-128,0,max(0,W-256)))
  else:y=int(rng.integers(max(1,H-256+1)));x=int(rng.integers(max(1,W-256+1)))
  a=np.array(a[y:y+256,x:x+256],dtype=np.float32,copy=True);b=np.array(b[y:y+256,x:x+256],dtype=np.int64,copy=True)
  if v is not None:b[~v[y:y+256,x:x+256]]=-1
  pad=((0,256-a.shape[0]),(0,256-a.shape[1]));a=np.pad(a,pad,constant_values=127.5);b=np.pad(b,pad,constant_values=-1)
  flips=[bool(rng.random()<.5),bool(rng.random()<.5)]
  for axis,flip in enumerate(flips):
   if flip:a=np.flip(a,axis);b=np.flip(b,axis)
  assert (b>=0).any()
  coords={'region':row['id'],'volume':row['volume_id'],'source':source,'z_native':z+row['region_zyx'][0][0],'y_native':y+row['region_zyx'][1][0],'x_native':x+row['region_zyx'][2][0],'foreground_centered':fg,'flips':flips}
  return torch.from_numpy(((a/255-.5)/.25).copy())[None,None].to(device),torch.from_numpy(b.copy())[None].to(device),coords

if __name__=='__main__':prepare()
