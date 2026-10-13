import os,json,hashlib
from pathlib import Path
import zarr,numpy as np
from PIL import Image
ROOT=Path('/opt/data/3_117/tanghr');BASE=Path(__file__).resolve().parent;OUT=BASE/'train_patches';OUT.mkdir(exist_ok=True)
cat=json.loads((ROOT/'MitoVerse_data/catalog.json').read_text());a=zarr.open_group(str(ROOT/'MitoVerse_data'/cat['jiang25_vol1']['zarr']),mode='r')['img'];rows=[]
for i in range(100):
 rng=np.random.default_rng(20260907+i);z=int(rng.integers(a.shape[0]));y=int(rng.integers(a.shape[1]-512+1));x=int(rng.integers(a.shape[2]-512+1))
 raw=np.asarray(a[z,y:y+512,x:x+512]);Image.fromarray(raw).save(OUT/f'{i:05}.png')
 rows.append(dict(index=i,volume='jiang25_vol1',zyx=[z,y,x],shape=[512,512],sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
 if i%10==0:print('exported',i+1,flush=True)
(OUT/'manifest.json').write_text(json.dumps(rows,indent=2));print('COMPLETE',len(rows),flush=True)
