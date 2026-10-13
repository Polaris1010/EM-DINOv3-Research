"""Apply annotated-region supervision, preserving and reporting out-of-region labels."""
import os,sys,json,time,hashlib
from pathlib import Path
from settings import *
sys.path.insert(0,str(ROOT/'codex/mitotopo_20260908_round2/deps'))
import numpy as np,zarr
from scipy.ndimage import distance_transform_edt,binary_erosion
from PIL import Image
from han_mask_audit import save,sha
def main():
 out=Path(os.environ['RUN_OUTPUT']);start=time.time();audit=json.loads((BASE/'han_mask_full_audit.json').read_text());assert len(audit)==5
 valid_anchors=json.loads((BASE/'han_valid_coordinates.json').read_text());fg_anchors={};reviews=[]
 for row in audit:
  vid=row['volume_id'];g=zarr.open_group(str(DATA/'data/han24'/f'{vid.removeprefix("han24_")}.zarr'),mode='r');m=g['mask'];lab=g['mito'];D,H,W=m.shape;dz,dy,dx=m.chunks;anchors=[None]*D;examples=[];inside=outside=0
  for z in range(0,D,dz):
   for y in range(0,H,dy):
    for x in range(0,W,dx):
     sl=(slice(z,min(z+dz,D)),slice(y,min(y+dy,H)),slice(x,min(x+dx,W)));v=np.asarray(m[sl])!=0;t=np.asarray(lab[sl])>0;fg=v&t;bad=(~v)&t;inside+=int(fg.sum());outside+=int(bad.sum())
     for iz in range(fg.shape[0]):
      if anchors[z+iz] is None and fg[iz].any():
       p=np.argwhere(fg[iz]);oy,ox=p[len(p)//2];anchors[z+iz]=[z+iz,y+int(oy),x+int(ox)]
     if bad.any() and len(examples)<4:
      zz,yy,xx=np.argwhere(bad)[0];examples.append((z+int(zz),y+int(yy),x+int(xx)))
   save(out/'progress.json',dict(volume_id=vid,z_completed=min(z+dz,D),z_total=D,seconds=time.time()-start))
  assert inside==row['foreground_inside'] and outside==row['foreground_outside'],'Audit input changed'
  fg_anchors[vid]=[p for p in anchors if p is not None];assert fg_anchors[vid]
  distances=[]
  for i,(z,y,x) in enumerate(examples):
   z0,y0,x0=max(z-16,0),max(y-16,0),max(x-16,0);roi=np.asarray(m[z0:min(D,z+17),y0:min(H,y+17),x0:min(W,x+17)])!=0
   distances.append(float(distance_transform_edt(~roi,sampling=(30,8,8))[z-z0,y-y0,x-x0]) if roi.any() else None)
   if i==0:
    y0,x0=max(0,min(y-128,H-256)),max(0,min(x-128,W-256));im=np.asarray(g['img'][z,y0:y0+256,x0:x0+256]);vm=np.asarray(m[z,y0:y0+256,x0:x0+256])!=0;gt=np.asarray(lab[z,y0:y0+256,x0:x0+256])>0
    rgb=np.repeat(im[:,:,None],3,axis=2);rgb[vm^binary_erosion(vm)]=[0,255,0];rgb[gt&~vm]=[255,0,0];Image.fromarray(rgb).save(BASE/f'mask_boundary_{vid}.png')
  reviews.append(dict(volume_id=vid,outside_foreground_voxels=outside,outside_share_of_all_foreground=outside/(inside+outside),sample_outside_coords_zyx=examples,sample_distance_to_mask_nm=distances,distance_note='Bounded 33-cube; null means no valid region within inspected cube, not an infinite distance',policy='All mask-zero voxels receive ignore_index=-1, including positive labels; no dilation, union, or background relabeling'))
 save(BASE/'han_outside_region_review.json',reviews);save(BASE/'han_valid_foreground_coordinates.json',fg_anchors)
 save(BASE/'mask_policy_v2.json',dict(status='validated',valid_region='mask != 0',ignore_index=-1,primary_source='https://www.nature.com/articles/s41467-024-50411-z',paper_rule='Supervised loss restricted to annotated mossy-terminal regions',local_polarity='Region-valued nonzero mask identifies the annotated terminals; supported by primary method and overwhelming local foreground containment',outside_foreground_voxels=sum(r['foreground_outside'] for r in audit),outside_label_policy='Ignore; never convert into background or expand the valid mask to fit labels',inference='Local polarity inferred from primary method plus region IDs and exhaustive local array audit; exact exporter definition unavailable',scope='Supervised training only; not an SSL image filter',original_identity_gate='Still pending scan/ROI provenance; mask correction does not prove dataset independence'))
 split=json.loads((R4/'split_mitoverse_probe_v1.json').read_text())['volumes'];changes=[]
 for seed in range(3):
  rng=np.random.default_rng(6100+seed);n=0;target=BASE/f'train_patch_schedule_masked_seed{seed}.jsonl'
  with (R4/f'train_patch_schedule_seed{seed}.jsonl').open() as src,target.open('w') as dst:
   for line in src:
    q=json.loads(line);vid=q['volume_id']
    if vid in fg_anchors:
     pool=fg_anchors[vid] if q['force_foreground'] else valid_anchors[vid];z,cy,cx=pool[int(rng.integers(len(pool)))];_,h,w=json.loads(split[vid]['shape_z_y_x']);q.update(z=z,y=min(max(cy-128,0),h-256),x=min(max(cx-128,0),w-256),mask_sampling='uniform eligible z, audited foreground/valid-region anchor; supervised only');n+=1
    dst.write(json.dumps(q,separators=(',',':'))+'\n')
  changes.append(dict(seed=seed,resampled_han_slots=n,sha256=sha(target),original_sha256=sha(R4/f'train_patch_schedule_seed{seed}.jsonl')))
 save(BASE/'masked_schedule_audit.json',dict(seeds=changes,source_and_volume_sequence='unchanged',foreground_slot_flags='unchanged; Han coordinates resampled within valid annotation',augmentation_sequence='unchanged',all_groups_require_new_baselines=True))
 save(out/'result.json',dict(status='complete',mask_policy='mask != 0 valid; mask == 0 ignored including positive labels',all_five_volumes_reviewed=True,seconds=time.time()-start,formal_training=False))
if __name__=='__main__':main()
