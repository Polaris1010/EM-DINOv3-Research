"""Image-only hierarchical draws; no labels, masks, or foreground filtering."""
import json,csv,random,hashlib
from pathlib import Path
import numpy as np,torch,zarr
from PIL import Image
from torch.utils.data import Dataset
from dinov3.data import DataAugmentationDINO

class Images(Dataset):
    def __init__(self,cfg,start,count,pool):
        self.start=start;self.count=count;self.pool=Path(pool);self.handles={}
        self.rows=list(csv.DictReader((self.pool/'ssl_dataset_manifest.csv').open()))
        self.schedule=np.load(self.pool/'ssl_schedule.npy',mmap_mode='r')
        self.aug=DataAugmentationDINO(global_crops_scale=cfg.crops.global_crops_scale,local_crops_scale=cfg.crops.local_crops_scale,local_crops_number=cfg.crops.local_crops_number,global_crops_size=cfg.crops.global_crops_size,local_crops_size=cfg.crops.local_crops_size,gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,gram_teacher_no_distortions=cfg.crops.gram_teacher_no_distortions,horizontal_flips=cfg.crops.horizontal_flips,mean=cfg.crops.rgb_mean,std=cfg.crops.rgb_std)
    def __len__(self):return self.count
    def raw(self,i):
        j,z,y,x,seed=map(int,self.schedule[i]);r=self.rows[j]
        if r['kind']=='png':
            with Image.open(r['path']) as f:im=f.convert('RGB')
        else:
            if j not in self.handles:
                if len(self.handles)>=2:self.handles.clear()
                self.handles[j]=zarr.open_array(r['path'],mode='r')
            a=self.handles[j];im=np.asarray(a[z,y:y+512,x:x+512])
            im=np.pad(im,((0,max(0,512-im.shape[0])),(0,max(0,512-im.shape[1]))),mode='reflect')
            im=Image.fromarray(im).convert('RGB')
        return im,(i,j,seed,hashlib.sha256(im.tobytes()).hexdigest())
    def __getitem__(self,i):
        i+=self.start;im,metadata=self.raw(i);seed=metadata[2]
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        return self.aug(im),metadata
