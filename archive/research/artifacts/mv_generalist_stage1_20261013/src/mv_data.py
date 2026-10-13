"""Audited source-volume development loader; final test arrays are never opened."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,obj):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def partition_check(m):
 if m['claim']!='transductive_source_volume_development' or m['unseen_acquisition_claim_permitted']:
  raise ValueError('This loader cannot certify unseen acquisition generalization')
 rows=m['entries'];ids=[r['id'] for r in rows]
 assert len(ids)==len(set(ids)) and not set(ids)&set(m['reserved_test_ids'])
 for r in rows:
  assert r['axes']=='ZYX' and r['coordinate_frame']=='stored_array' and r['supervised_role'] in {'train','dev'}
  assert r['source_volume_identity'] and r['annotation_status']=='curator_declared_proofread'
  assert r['valid_mask_policy']=='complete_without_ignore_declared'
  for (a,b),n in zip(r['region_zyx'],r['shape_zyx']):assert type(a)==type(b)==int and 0<=a<b<=n
  # Same physical acquisition may recur but original file MUST NOT recur across roles.
 for a in rows:
  for b in rows:
   if a['supervised_role']!=b['supervised_role']:
    assert a['image_path']!=b['image_path'] and a['source_volume_identity']!=b['source_volume_identity']
def manifest(locked=True):
 m=json.loads((R/'MV-Dev-v1.json').read_text());partition_check(m)
 if locked:assert m['status']=='locked_development' and m['training_permitted']
 return m
def open_cache(row):
 d=R/'cache'/row['id']
 return np.load(d/'image.npy',mmap_mode='r'),np.load(d/'label.npy',mmap_mode='r')
class Samples:
 def __init__(self,seed,sampler='source_balanced'):
  assert sampler in {'source_balanced','slice_uniform'}
  self.rows=[r for r in manifest()['entries'] if r['supervised_role']=='train']
  self.sources=sorted({r['source_id'] for r in self.rows});self.seed=seed;self.sampler=sampler
  self.handles={r['id']:open_cache(r) for r in self.rows}
 def sample(self,step,device='cuda',forced_source=None):
  import torch
  rng=np.random.default_rng(np.random.SeedSequence([self.seed,step]))
  selected=self.rows
  if forced_source is not None:selected=[r for r in selected if r['source_id']==forced_source]
  elif self.sampler=='source_balanced':
   s=self.sources[int(rng.integers(len(self.sources)))];selected=[r for r in selected if r['source_id']==s]
  weights=np.array([r['region_zyx'][0][1]-r['region_zyx'][0][0] for r in selected])
  index=int(rng.integers(weights.sum()));chosen=int(np.searchsorted(np.cumsum(weights),index,side='right'))
  row=selected[chosen];z=index-int(weights[:chosen].sum());im,lab=self.handles[row['id']]
  a=np.asarray(im[z]);b=np.asarray(lab[z]);H,W=a.shape
  foreground=False
  if rng.random()<.5:
   ys,xs=np.nonzero(b)
   if len(ys):
    k=int(rng.integers(len(ys)));y=int(np.clip(int(ys[k])-128,0,max(0,H-256)));x=int(np.clip(int(xs[k])-128,0,max(0,W-256)));foreground=True
  if not foreground:y=int(rng.integers(max(1,H-256+1)));x=int(rng.integers(max(1,W-256+1)))
  a=np.array(a[y:y+256,x:x+256],dtype=np.float32,copy=True);b=np.array(b[y:y+256,x:x+256],dtype=np.int64,copy=True)
  pad=((0,256-a.shape[0]),(0,256-a.shape[1]));a=np.pad(a,pad,constant_values=127.5);b=np.pad(b,pad,constant_values=-1)
  if rng.random()<.5:a=np.flip(a,0);b=np.flip(b,0)
  if rng.random()<.5:a=np.flip(a,1);b=np.flip(b,1)
  coords={'volume':row['id'],'source':row['source_id'],'z':z,'y':y,'x':x,'foreground_centered':foreground}
  return torch.from_numpy(((a/255-.5)/.25).copy())[None,None].to(device),torch.from_numpy(b.copy())[None].to(device),coords
