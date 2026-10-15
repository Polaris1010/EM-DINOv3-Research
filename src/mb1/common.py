"""MB1 reads existing official TIFFs and checkpoints; writes only MB1 artifacts."""
import hashlib,json,os
from pathlib import Path
R=Path(os.environ.get('MB1_ROOT','/opt/data/3_117/tanghr/codex/mb1_official_benchmark_20261015'))
BASE=R.parent;M3=BASE/'mv_generalist_v2_han_fullgt_20261014';M4=BASE/'mv4_official_mitonet_20261014';M5=BASE/'mv5_mitonet_adaptation_20261015';OLD=BASE/'mv_generalist_stage1_20261013';MV2=BASE/'mv_dev_v2_notta_20261013'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(o,indent=2)+'\n');t.replace(p)
def config():return read(R/'MB1_protocol.json')
def manifest():return read(R/'MB1_benchmark_manifest.json')
def rows():return manifest()['entries']
def run_spec(run_id):return next(r for r in config()['runs'] if r['run_id']==run_id)
def gpu_guard():
 import torch,subprocess,csv
 uu={int(r[0]):r[1].strip() for r in csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines())};visible=os.environ.get('CUDA_VISIBLE_DEVICES')
 assert visible in {uu[g] for g in [1,2,3]} and torch.cuda.is_available() and torch.cuda.device_count()==1,'GPU0 forbidden'
 return next(g for g in [1,2,3] if uu[g]==visible)
class Planes:
 def __init__(self,path,shape):
  import tifffile
  self.path=path;self.shape=tuple(shape);self.tf=None
  try:self.array=tifffile.memmap(path,mode='r')
  except (ValueError,TypeError):self.array=None;self.tf=tifffile.TiffFile(path);assert len(self.tf.pages)==self.shape[0],'TIFF needs explicit lossless native plane access'
  if self.array is not None:
   if self.array.ndim==2:self.array=self.array[None]
   assert self.array.shape==self.shape
 def __getitem__(self,z):return self.array[z] if self.array is not None else self.tf.pages[z].asarray()
 def close(self):
  if self.tf:self.tf.close()
