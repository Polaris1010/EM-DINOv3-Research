"""Full pixel integrity check and immutable staging caches, only admitted train/dev arrays."""
from pathlib import Path
import sys,json,time,hashlib,csv
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'src'),'/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps']
from mv_data import manifest,sha,atomic
import numpy as np,zarr
def main():
 m=manifest(locked=False);results=[]
 for row in m['entries']:
  d=R/'cache'/row['id'];d.mkdir(parents=True,exist_ok=True)
  assert not (d/'QC.json').exists(),'Refuse to replace an existing audited cache'
  p=Path(row['image_path'])
  for k,h in row['metadata_sha256'].items():assert sha(p/k)==h
  group=zarr.open_group(str(p),mode='r');assert set(group.array_keys())=={'img','mito'}
  assert list(group['img'].shape)==list(group['mito'].shape)==row['shape_zyx']
  assert group['img'].dtype==np.uint8 and np.issubdtype(group['mito'].dtype,np.unsignedinteger)
  shape=tuple(b-a for a,b in row['region_zyx']);img=np.lib.format.open_memmap(d/'image.npy',mode='w+',shape=shape,dtype=np.uint8)
  lab=np.lib.format.open_memmap(d/'label.npy',mode='w+',shape=shape,dtype=np.uint8)
  hs=[hashlib.sha256(),hashlib.sha256()];ids=set();count=0;positive_slices=[];pixel_total=0;minim=255;maxim=0
  for z in range(0,shape[0],16):
   bounds=row['region_zyx'];key=(slice(bounds[0][0]+z,min(bounds[0][0]+z+16,bounds[0][1])),slice(*bounds[1]),slice(*bounds[2]))
   a=np.asarray(group['img'][key]);b=np.asarray(group['mito'][key]);assert a.shape==b.shape
   img[z:z+len(a)]=a;lab[z:z+len(a)]=b>0;hs[0].update(a.tobytes());hs[1].update(b.tobytes())
   ids.update(map(int,np.unique(b)));count+=int(np.count_nonzero(b));pixel_total+=b.size
   minim=min(minim,int(a.min()));maxim=max(maxim,int(a.max()))
   positive_slices.extend(z+i for i in np.flatnonzero(np.any(b>0,axis=(1,2))))
  img.flush();lab.flush();ids.discard(0)
  declared=group.attrs['mitoverse']['n_instances'];assert len(ids)==declared,(row['id'],len(ids),declared)
  assert count>0 and count<pixel_total and maxim>minim
  evidence={'id':row['id'],'status':'PASS','shape':shape,'source_image_sha256_raw_pixels':hs[0].hexdigest(),
   'source_instance_label_sha256_raw_pixels':hs[1].hexdigest(),'instance_count':len(ids),'declared_instance_count':declared,
   'foreground_voxels':count,'total_voxels':pixel_total,'positive_slices':len(positive_slices),'image_range':[minim,maxim],
   'image_cache_sha256':sha(d/'image.npy'),'binary_label_cache_sha256':sha(d/'label.npy'),
   'completeness_status':'curator_declared_proofread; numeric QC cannot prove exhaustive annotation',
   'test_labels_accessed':False,'original_files_written':False}
  atomic(d/'QC.json',evidence);results.append(evidence);print('QC_PASS',row['id'],len(ids),count/pixel_total,flush=True)
  del img,lab
 atomic(R/'evidence/pixel_QC.json',{'status':'PASS','volumes':results,'test_labels_accessed':False})
 m['status']='locked_development';m['training_permitted']=True;m['pixel_qc_sha256']=sha(R/'evidence/pixel_QC.json')
 for row in m['entries']:row['pixel_qc_sha256']=sha(R/'cache'/row['id']/'QC.json')
 atomic(R/'MV-Dev-v1.json',m)
 elig=list(csv.DictReader((R/'mitoverse_supervised_eligibility.csv').open()))
 for row in elig:
  if row['admission_status']=='pending_pixel_QC':row['admission_status']='admitted_development_curator_contract_QC_pass'
 with (R/'mitoverse_supervised_eligibility.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=elig[0].keys());w.writeheader();w.writerows(elig)
 print('MV_DEV_LOCKED',sha(R/'MV-Dev-v1.json'),flush=True)
if __name__=='__main__':main()
