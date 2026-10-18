"""Shared D1/D2 loader; original official augmentation/collation are unchanged."""
import io,json,sqlite3
from pathlib import Path
import numpy as np
from PIL import Image
from common import R,ZIP,read,sha
from sampling import choose,read_member
from legacy_ssl_data import Images as LegacyImages,Collate

class Images(LegacyImages):
 def __init__(self,manifest,cfg,seed,start,count,**kwargs):
  super().__init__(R/'inputs/pretrain_all_mitoverse.json',cfg,seed,start,count,**kwargs)
  self.spec=read(manifest);self.pool=read(R/'mitoverse_pool.json');self.cem=read(R/'cem_pool.json');self.rows=self.pool['volumes'];self.row_index={r['volume_id']:r for r in self.rows}
  assert self.spec['mitoverse_pool_sha256']==sha(R/'mitoverse_pool.json') and self.spec['cem_pool_sha256']==sha(R/'cem_pool.json')
  assert ZIP.stat().st_size==self.cem['archive_bytes'] and ZIP.stat().st_mtime_ns==self.cem['archive_mtime_ns']
  self.archive_handle=None;self.db=None
 def raw(self,absolute_index):
  where=choose(self.pool,self.cem,self.spec['cem_fraction'],self.seed,absolute_index)
  if where['domain']=='CEM1.5M':
   if self.db is None:self.db=sqlite3.connect('file:'+self.cem['index']+'?mode=ro&immutable=1',uri=True)
   if self.archive_handle is None:self.archive_handle=ZIP.open('rb')
   group,path,offset,compressed,size,crc,method=self.db.execute('SELECT collection,path,offset,compressed,bytes,crc,method FROM images WHERE id=?',(where['image_id'],)).fetchone()
   assert group==where['source']
   value=read_member(self.archive_handle,offset,compressed,size,crc,method)
   with Image.open(io.BytesIO(value)) as im:
    assert im.mode in ['L','RGB'] and min(im.size)>0
    image=im.convert('RGB');where['native_hw']=[im.height,im.width]
   # No eager upsampling or padding: the SAME official RandomResizedCrop
   # outputs global256/local112 from native CEM (typically224) or MV512.
   where['format_adapter']='native_PIL_RGB_then_unchanged_official_RRC';return image,where
  row=self.row_index[where['volume_id']];vid=row['volume_id']
  if vid not in self.handles:
   if len(self.handles)>=2:self.handles.clear()
   self.handles[vid]=self.image_opener(row)
  array=self.handles[vid];assert list(array.shape)==row['shape']
  z,y,x=where['z'],where['y'],where['x'];crop=np.asarray(array[z,y:y+512,x:x+512])
  assert crop.ndim==2 and crop.dtype==np.uint8 and np.isfinite(crop).all()
  py,px=512-crop.shape[0],512-crop.shape[1]
  if py or px:crop=np.pad(crop,((0,py),(0,px)),mode='reflect' if min(crop.shape)>1 else 'edge')
  assert all(not (a<=z<b) for box in self.pool['heldout_regions'].get(vid,[]) for a,b in [box[0]])
  return Image.fromarray(crop).convert('RGB'),where
