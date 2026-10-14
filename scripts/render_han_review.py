"""Native GT/mask/image review panels, no predictions and no training changes.

Uses only the three non-test Han assets admitted to MV3. Existing full-volume
GT-mask checks are inherited; this script reads one deterministic slice each.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import zarr
from PIL import Image,ImageDraw,ImageFont

def font(size):
 p=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
 return ImageFont.truetype(str(p),size) if p.exists() else ImageFont.load_default()

def render(root):
 output=root/'review/Han24';output.mkdir(parents=True,exist_ok=True)
 evidence=[];panel=640;margin=20;header=135;row_height=panel+70
 for name in ['neg0','pos0','pos2']:
  vid='han24_'+name;cache=root/'cache'/vid/'CACHE.json';receipt=json.loads(cache.read_text());zs=receipt['valid_z'];z=zs[len(zs)//2]
  path=Path('/opt/data/3_117/tanghr/MitoVerse_data/data/han24')/(name+'.zarr');group=zarr.open_group(str(path),mode='r')
  image=np.asarray(group['img'][z]);label=np.asarray(group['mito'][z]);mask=np.asarray(group['mask'][z]);valid=mask!=0;positive=label>0
  assert image.shape==label.shape==mask.shape and image.dtype==np.uint8
  current=np.load(receipt['valid_path'],mmap_mode='r');assert np.array_equal(current[z].astype(bool),valid),'Live loader mask differs from native mask!=0'
  H,W=image.shape;ys,xs=np.nonzero(valid);index=len(ys)//2;cy,cx=int(ys[index]),int(xs[index]);side=min(640,H,W)
  y=max(0,min(H-side,cy-side//2));x=max(0,min(W-side,cx-side//2));bounds=(y,y+side,x,x+side)
  gt_rgb=np.zeros((*image.shape,3),np.uint8);gt_rgb[positive&valid]=[70,230,170];gt_rgb[positive&~valid]=[255,60,185]
  mask_rgb=np.zeros((*image.shape,3),np.uint8);mask_rgb[valid]=[255,210,65]
  raw=Image.fromarray(image).convert('RGB');gt=Image.fromarray(gt_rgb);region=Image.fromarray(mask_rgb)
  canvas=Image.new('RGB',(panel*3+margin*4,header+row_height*2+50),(18,22,28));draw=ImageDraw.Draw(canvas)
  title=f'{vid} | native Z={z} | shape YX={H}x{W} | mask IDs={np.unique(mask).tolist()}'
  draw.text((margin,15),title,font=font(23),fill='white')
  draw.text((margin,55),'GT: mito>0, green inside mask / magenta outside. MASK: yellow=nonzero terminal region.',font=font(17),fill=(220,225,230))
  draw.text((margin,85),f'Slice valid pixels={int(valid.sum()):,}; GT inside={int((positive&valid).sum()):,}; GT outside={int((positive&~valid).sum()):,}. No model output.',font=font(17),fill=(220,225,230))
  for col,(im,title_text) in enumerate([(raw,'ORIGINAL IMAGE'),(gt,'GROUND TRUTH (all positive labels)'),(region,'ANNOTATION MASK (original nonzero IDs)')]):
   px=margin+col*(panel+margin)
   for rownum in range(2):
    py=header+rownum*row_height
    source=im if rownum==0 else im.crop((x,y,x+side,y+side))
    resized=source.copy();resized.thumbnail((panel,panel),Image.Resampling.LANCZOS if col==0 else Image.Resampling.NEAREST)
    ix=px+(panel-resized.width)//2;iy=py+(panel-resized.height)//2
    canvas.paste(resized,(ix,iy))
    if rownum==0:
     scale=resized.width/W;draw.rectangle((ix+x*scale,iy+y*scale,ix+(x+side)*scale,iy+(y+side)*scale),outline=(90,175,255),width=2)
    draw.text((px,py+panel+8),title_text,font=font(17),fill='white')
    if rownum==1:draw.text((px,py+panel+32),f'Zoom native Y[{y}:{y+side}] X[{x}:{x+side}]',font=font(14),fill=(170,180,190))
  target=output/(vid+'_raw_gt_mask.png');canvas.save(target)
  # Preserve the three component images as requested, without any label editing.
  for suffix,im in [('original',raw),('GT_semantic',gt),('mask_nonzero',region)]:im.save(output/(vid+'_'+suffix+'.png'))
  hashes={k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in [('image',image),('label',label),('mask',mask)]}
  evidence.append({'volume_id':vid,'native_z':z,'selection':'middle of existing legal valid-Z list, no predictions consulted','native_shape_yx':[H,W],'mask_values':np.unique(mask).tolist(),'valid_pixels':int(valid.sum()),'GT_foreground_inside':int((positive&valid).sum()),'GT_foreground_outside':int((positive&~valid).sum()),'native_slice_sha256':hashes,'live_loader_equals_native_mask_nonzero':True,'zoom_bounds_yxyx':list(bounds),'composite_file':str(target),'visualization':'top whole-slice / bottom identical native crop; no image normalization, GT>0 semantic colors; original integer labels unchanged','test_pixels_read':False})
 (output/'review_receipt.json').write_text(json.dumps({'status':'PASS','model_output_used':False,'new_full_audit':False,'training_changed':False,'rows':evidence},indent=2)+'\n')
 print(json.dumps({'status':'PASS','composites':[r['composite_file'] for r in evidence],'receipt':str(output/'review_receipt.json')}))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();render(a.root)
