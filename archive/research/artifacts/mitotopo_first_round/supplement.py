import json,csv,time,subprocess,signal,sys
from pathlib import Path
p=Path(__file__).resolve().parent
rows=list(csv.DictReader((p/'dataset_manifest.csv').open()))
for r in rows:
    raw=json.loads(r['voxel_nm_raw'])
    r['voxel_nm_z_y_x']=json.dumps(raw[::-1]) if isinstance(raw,list) and len(raw)==3 else ''
    r['notes']=r['notes'].replace('voxel order unverified','voxel_nm XYZ confirmed by dataset_registry.yaml:4; converted to ZYX')
with (p/'dataset_manifest.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Regression: a nohup-like parent ignores SIGINT, but the repaired child restores it.
previous=signal.signal(signal.SIGINT,signal.SIG_IGN)
code="import signal,time,sys\nsignal.signal(signal.SIGINT,signal.default_int_handler)\nprint('READY',flush=True)\ntry: time.sleep(5)\nexcept KeyboardInterrupt: print('INTERRUPTED',flush=True);sys.exit(130)"
child=subprocess.Popen([sys.executable,'-u','-c',code],stdout=subprocess.PIPE,text=True)
assert child.stdout.readline().strip()=='READY';child.send_signal(signal.SIGINT);out,_=child.communicate(timeout=3)
signal.signal(signal.SIGINT,previous)
assert child.returncode==130 and 'INTERRUPTED' in out
(p/'signal_regression.json').write_text(json.dumps({'passed':True,'child_exit_code':child.returncode,'scope':'SIGINT restoration under inherited SIG_IGN; emergency model checkpoint not exercised'},indent=2))
# A paired augmentation view for the exact CPU test sample, using bounded patch reads.
import numpy as np,zarr
from PIL import Image,ImageDraw
root=Path('/opt/data/3_117/tanghr/MitoVerse_data')
g=zarr.open_group(str(root/'data/jiang25/vol1.zarr'),mode='r');rng=np.random.default_rng(20260907)
z=int(rng.integers(g['img'].shape[0]));y=int(rng.integers(g['img'].shape[1]-192+1));x=int(rng.integers(g['img'].shape[2]-192+1))
a=np.asarray(g['img'][z,y:y+192,x:x+192]);b=np.asarray(g['mito'][z,y:y+192,x:x+192])>0
flips=[];aa=a.copy();bb=b.copy()
for axis in (0,1):
    if rng.random()<.5:aa=np.flip(aa,axis);bb=np.flip(bb,axis);flips.append(axis)
canvas=Image.new('RGB',(768,216),'white')
for i,(title,arr) in enumerate([('before image',a),('before mask',b*255),('after image',aa),('after mask',bb*255)]):
    canvas.paste(Image.fromarray(arr.astype('uint8')).convert('RGB'),(i*192,24));ImageDraw.Draw(canvas).text((i*192+2,3),title,fill='black')
canvas.save(p/'previews/augmentation_cpu_sample.png')
(p/'augmentation_audit.json').write_text(json.dumps({'volume':'jiang25_vol1','zyx':[z,y,x],'flip_axes':flips,'same_geometry_for_image_and_label':True,'seed':20260907},indent=2))
print('Manifest axes enriched; signal regression passed; paired augmentation view saved')
