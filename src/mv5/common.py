"""MV4 uses the completed MV3 data protocol as immutable input."""
import hashlib,json,os
from pathlib import Path
R=Path(os.environ.get('MV5_ROOT','/opt/data/3_117/tanghr/codex/mv5_mitonet_adaptation_20261015'))
M3=Path('/opt/data/3_117/tanghr/codex/mv_generalist_v2_han_fullgt_20261014')
MV2=M3.parent/'mv_dev_v2_notta_20261013'
MODEL='MitoNet-MV'
def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def config():return read(R/'MV5_protocol.json')
def rows():
 c=config();assert sha(M3/'MV-Generalist.json')==c['manifest_sha256']
 rr=[r for r in read(M3/'MV-Generalist.json')['entries'] if r['supervised_role']=='dev']
 assert len(rr)==38 and len({r['source_id'] for r in rr})==12
 return rr
def cache(row):
 j=read(M3/'cache'/row['id']/'CACHE.json')
 assert j['manifest_sha256']==config()['manifest_sha256'] and j['region_zyx']==row['region_zyx']
 return j
def image_array(row):
 """Inference has no access to GT values or valid-mask values."""
 import numpy as np
 j=cache(row);key=tuple(slice(a,b) for a,b in j['offsets'])
 im=np.load(j['paths']['image'],mmap_mode='r')[key]
 assert list(im.shape)==j['shape_zyx'] and im.dtype==np.uint8
 return im,j
def evaluation_arrays(row):
 import numpy as np
 j=cache(row);key=tuple(slice(a,b) for a,b in j['offsets'])
 gt=np.load(j['paths']['label'],mmap_mode='r')[key]
 valid=np.load(j['valid_path'],mmap_mode='r') if j['valid_path'] else None
 assert list(gt.shape)==j['shape_zyx']
 if row['source_id']=='han24':assert valid is None and row['mask_usage']=='NONE'
 return gt,valid,j
def gpu_guard():
 import torch,subprocess,csv
 allowed={int(row[0]):row[1].strip() for row in csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines())}
 uuid=os.environ.get('CUDA_VISIBLE_DEVICES');assert uuid in {allowed[i] for i in [1,2,3]},'Physical GPU0 forbidden'
 assert torch.cuda.is_available() and torch.cuda.device_count()==1
 return next(i for i in [1,2,3] if allowed[i]==uuid)
