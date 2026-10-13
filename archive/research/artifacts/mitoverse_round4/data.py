from common import *
from torch.utils.data import Dataset,DataLoader
import time

def create_schedules():
 split=json.loads((BASE/'split_mitoverse_probe_v1.json').read_text());rows=split['volumes'];locs={}
 rng=np.random.default_rng(8921)
 for vid in split['train']:
  r=rows[vid];a=zarr.open_array(r['label_path'],mode='r');coords=[]
  for z in sorted(set(np.linspace(0,a.shape[0]-1,min(16,a.shape[0]),dtype=int).tolist())):
   sl=np.asarray(a[z]);yx=np.argwhere(sl>0)
   if len(yx):
    chosen=yx[rng.choice(len(yx),min(256,len(yx)),replace=False)];coords.extend([[z,int(y),int(x)] for y,x in chosen])
  locs[vid]=coords;print('FG_INDEX',vid,len(coords),flush=True)
 save(BASE/'train_foreground_coordinates.json',locs)
 sources=sorted({rows[v]['source'] for v in split['train']})
 pools={s:[v for v in split['train'] if rows[v]['source']==s] for s in sources}
 positive={s:[v for v in pools[s] if locs[v]] for s in sources}
 assert all(positive.values()),'Every training source needs at least one sampled foreground volume'
 for s in range(3):
  rng=np.random.default_rng(s);out=BASE/f'train_patch_schedule_seed{s}.jsonl'
  with out.open('w') as f:
   for i in range(10000*8):
    source=sources[int(rng.integers(len(sources)))];fg=bool(rng.random()<.5);pool=positive[source] if fg else pools[source];vid=pool[int(rng.integers(len(pool)))];d,h,w=json.loads(rows[vid]['shape_z_y_x'])
    if fg:z,cy,cx=locs[vid][int(rng.integers(len(locs[vid])))];y=min(max(0,cy-128),max(0,h-256));x=min(max(0,cx-128),max(0,w-256))
    else:z=int(rng.integers(d));y=int(rng.integers(max(1,h-255)));x=int(rng.integers(max(1,w-255)))
    q=dict(index=i,volume_id=vid,source=source,z=z,y=y,x=x,force_foreground=fg,flip_y=bool(rng.integers(2)),flip_x=bool(rng.integers(2)),rot90=int(rng.integers(4)))
    f.write(json.dumps(q,separators=(',',':'))+'\n')
 save(BASE/'schedule_audit.json',{'samples_per_seed':80000,'sha256':{str(s):sha(BASE/f'train_patch_schedule_seed{s}.jsonl') for s in range(3)},'source_sampling':'uniform source, uniform volume within chosen foreground/all pool','positive_ratio':.5,'foreground_index':'up to 16 evenly spaced TRAIN slices per volume, at most 256 coordinates each; zeros mean none in sampled slices, not proven empty volume','augmentation':'recorded flips and rotations, no hidden stochastic augmentation','train_only':True})
 print('SCHEDULES_COMPLETE',flush=True)

class Patches(Dataset):
 def __init__(self,s,count,norm):
  self.rows=json.loads((BASE/'split_mitoverse_probe_v1.json').read_text())['volumes'];self.norm=norm;self.handles={}
  with (BASE/f'train_patch_schedule_seed{s}.jsonl').open() as f:self.schedule=[json.loads(next(f)) for i in range(count)]
 def __len__(self):return len(self.schedule)
 def __getitem__(self,i):
  q=self.schedule[i];v=q['volume_id'];r=self.rows[v];assert r['split']=='train'
  if v not in self.handles:
   if len(self.handles)>=2:self.handles.clear()
   self.handles[v]=(zarr.open_array(r['image_path'],mode='r'),zarr.open_array(r['label_path'],mode='r'))
  a,b=self.handles[v];z,y,x=q['z'],q['y'],q['x'];im=np.asarray(a[z,y:y+256,x:x+256]);lab=np.asarray(b[z,y:y+256,x:x+256])>0
  assert im.dtype==np.uint8
  if q['force_foreground']:assert lab.any()
  t=lab.astype(np.int64);pad=((0,256-im.shape[0]),(0,256-im.shape[1]));im=np.pad(im,pad,mode='reflect');t=np.pad(t,pad,constant_values=-1)
  for axis,k in [(0,'flip_y'),(1,'flip_x')]:
   if q[k]:im=np.flip(im,axis);t=np.flip(t,axis)
  im=np.rot90(im,q['rot90']);t=np.rot90(t,q['rot90'])
  return torch.from_numpy(normalize(im,self.norm).copy()),torch.from_numpy(t.copy()),i

def loader(s,count,norm):
 return DataLoader(Patches(s,count,norm),batch_size=8,shuffle=False,num_workers=2,prefetch_factor=1,persistent_workers=False,pin_memory=False,multiprocessing_context='spawn',generator=torch.Generator().manual_seed(s))
if __name__=='__main__':create_schedules()
