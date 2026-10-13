"""Bounded chunked audit of only the five suspected partially annotated Han volumes."""
import os,sys,json,time,hashlib,csv
from pathlib import Path
from settings import *
sys.path.insert(0,str(ROOT/'codex/mitotopo_20260908_round2/deps'))
import numpy as np,zarr
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,default=str));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for a in iter(lambda:f.read(8*1024**2),b''):h.update(a)
 return h.hexdigest()
def main():
 out=Path(os.environ['RUN_OUTPUT']);start=time.time();rows=[];anchors={}
 for path in sorted((DATA/'data/han24').glob('*.zarr')):
  g=zarr.open_group(str(path),mode='r');m=g['mask'];lab=g['mito'];assert g['img'].shape==m.shape==lab.shape
  D,H,W=m.shape;dz,dy,dx=m.chunks;valid=fg_inside=fg_outside=0;values={};per_slice=[None]*D;mh=hashlib.sha256();lh=hashlib.sha256()
  for z in range(0,D,dz):
   for y in range(0,H,dy):
    for x in range(0,W,dx):
     sl=(slice(z,min(z+dz,D)),slice(y,min(y+dy,H)),slice(x,min(x+dx,W)));a=np.asarray(m[sl]);b=np.asarray(lab[sl]);mh.update(a.tobytes());lh.update(b.tobytes());v=a!=0;f=b>0
     valid+=int(v.sum());fg_inside+=int((v&f).sum());fg_outside+=int((~v&f).sum())
     u,n=np.unique(a,return_counts=True)
     for k,num in zip(u,n):values[str(int(k))]=values.get(str(int(k)),0)+int(num)
     for iz in range(a.shape[0]):
      if per_slice[z+iz] is None and v[iz].any():
       coords=np.argwhere(v[iz]);oy,ox=coords[len(coords)//2];per_slice[z+iz]=[z+iz,y+int(oy),x+int(ox)]
   save(out/'progress.json',dict(volume=path.name,z_completed=min(z+dz,D),z_total=D,seconds=time.time()-start))
  # Record every volume before adjudicating positive labels outside its region mask.
  vid='han24_'+path.stem;anchors[vid]=[a for a in per_slice if a is not None]
  row=dict(volume_id=vid,shape_zyx=list(m.shape),total_voxels=int(np.prod(m.shape)),mask_nonzero_voxels=valid,nonzero_fraction=valid/int(np.prod(m.shape)),foreground_inside=fg_inside,foreground_outside=fg_outside,mask_values=values,mask_chunk_stream_sha256=mh.hexdigest(),label_chunk_stream_sha256=lh.hexdigest(),hash_order='ZYX chunk iteration; decompressed C-order chunk bytes')
  rows.append(row);save(BASE/'han_mask_full_audit.json',rows);print(json.dumps(row),flush=True)
 assert len(rows)==5
 save(BASE/'han_valid_coordinates.json',anchors)
 if any(r['foreground_outside'] for r in rows):
  save(out/'result.json',dict(status='complete',volumes=5,mask_policy_gate='needs_outside_region_review',outside_foreground_voxels=sum(r['foreground_outside'] for r in rows),seconds=time.time()-start))
  print('Full audit complete; outside-region positives need review',flush=True);return
 # Policy follows the primary paper's annotated-terminal-only supervision; numerical polarity
 # is inferred from region IDs plus exhaustive foreground containment, not importer wording.
 save(BASE/'mask_policy_v2.json',dict(status='validated',valid_region='mask != 0',ignore_index=-1,applies_to='supervised han24 only; never used for new SSL image selection',primary_source='https://www.nature.com/articles/s41467-024-50411-z',paper_rule='Loss is restricted to labeled mossy-terminal regions',polarity_evidence='Nonzero mask contains terminal region IDs; all foreground labels in all five complete volumes are inside it',inference='Mapping the local array polarity to the paper rule uses local array evidence; the generic importer does not itself state polarity',original_scan_mapping='still requires original ROI linkage; mask correction alone does not pass SSL identity gate'))
 changes=[]
 for seed in range(3):
  rng=np.random.default_rng(6100+seed);changed=0;target=BASE/f'train_patch_schedule_masked_seed{seed}.jsonl'
  split=json.loads((R4/'split_mitoverse_probe_v1.json').read_text())['volumes']
  with (R4/f'train_patch_schedule_seed{seed}.jsonl').open() as old,target.open('w') as new:
   for line in old:
    q=json.loads(line);vid=q['volume_id']
    if vid in anchors and not q['force_foreground']:
     z,cy,cx=anchors[vid][int(rng.integers(len(anchors[vid])))];_,h,w=json.loads(split[vid]['shape_z_y_x']);q.update(z=z,y=min(max(cy-128,0),h-256),x=min(max(cx-128,0),w-256),mask_sampling='uniform nonempty slice, fixed valid-region anchor; supervised only');changed+=1
    new.write(json.dumps(q,separators=(',',':'))+'\n')
  changes.append(dict(seed=seed,resampled_han_nonforeground_slots=changed,sha256=sha(target),original_schedule_sha256=sha(R4/f'train_patch_schedule_seed{seed}.jsonl')))
 save(BASE/'masked_schedule_audit.json',dict(seeds=changes,source_and_volume_sequence='unchanged',foreground_slots='unchanged',augmentation_sequence='unchanged',change='Han non-foreground slots now center on valid annotation; mask-zero target pixels ignored',formal_training_launched=False))
 save(out/'result.json',dict(status='complete',volumes=5,all_foreground_inside_nonzero_mask=True,seconds=time.time()-start,formal_training_launched=False))
if __name__=='__main__':main()
