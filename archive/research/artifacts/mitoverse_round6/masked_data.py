"""Round6 corrected supervised loader; annotation masks are not SSL sampling inputs."""
import sys,json
from pathlib import Path
from settings import BASE,R4,ROOT
sys.path.insert(0,str(R4))
import common as c
import torch,numpy as np,zarr
from torch.utils.data import Dataset,DataLoader
IGNORE_INDEX=-1
def supervised_target(instances,region=None):
 target=(np.asarray(instances)>0).astype(np.int64)
 if region is not None:
  region=np.asarray(region)
  if region.shape!=target.shape:raise ValueError('Image/label/annotation-region grids must agree')
  target[region==0]=IGNORE_INDEX
 return target
class MaskedPatches(Dataset):
 def __init__(self,seed,count=80000,norm='N2'):
  self.rows=json.loads((R4/'split_mitoverse_probe_v1.json').read_text())['volumes'];self.norm=norm;self.handles={}
  policy=json.loads((BASE/'mask_policy_v2.json').read_text());assert policy['status']=='validated' and policy['valid_region']=='mask != 0'
  with (BASE/f'train_patch_schedule_masked_seed{seed}.jsonl').open() as f:self.schedule=[json.loads(next(f)) for _ in range(count)]
 def __len__(self):return len(self.schedule)
 def __getitem__(self,i):
  q=self.schedule[i];r=self.rows[q['volume_id']];assert r['split']=='train'
  if q['volume_id'] not in self.handles:
   if len(self.handles)>=2:self.handles.clear()
   root=zarr.open_group(str(Path(r['image_path']).parent),mode='r');self.handles[q['volume_id']]=root
  g=self.handles[q['volume_id']];z,y,x=q['z'],q['y'],q['x'];sl=(z,slice(y,y+256),slice(x,x+256));im=np.asarray(g['img'][sl]);lab=np.asarray(g['mito'][sl]);region=np.asarray(g['mask'][sl]) if 'mask' in g else None
  assert im.shape==lab.shape and im.dtype==np.uint8
  target=supervised_target(lab,region);assert (target>=0).any()
  if q['force_foreground']:assert (target==1).any()
  padding=((0,256-im.shape[0]),(0,256-im.shape[1]));im=np.pad(im,padding,mode='reflect');target=np.pad(target,padding,constant_values=IGNORE_INDEX)
  for axis,key in [(0,'flip_y'),(1,'flip_x')]:
   if q[key]:im=np.flip(im,axis);target=np.flip(target,axis)
  im=np.rot90(im,q['rot90']);target=np.rot90(target,q['rot90'])
  return torch.from_numpy(c.normalize(im,self.norm).copy()),torch.from_numpy(target.copy()),i
def loader(seed,count,norm='N2'):
 return DataLoader(MaskedPatches(seed,count,norm),batch_size=8,shuffle=False,num_workers=2,prefetch_factor=1,persistent_workers=False,pin_memory=False,multiprocessing_context='spawn',generator=torch.Generator().manual_seed(seed))
