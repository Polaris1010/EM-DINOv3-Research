"""Package reproducibility metadata, logs, metrics, and plots; exclude datasets and weights."""
import json,hashlib,zipfile,time,subprocess,os
from pathlib import Path
BASE=Path(__file__).resolve().parent
m=json.loads((BASE/'comparison.json').read_text());assert m['status']=='complete'
assert (BASE/'lower_lr_control_complete.json').exists()
for gpu in [1,2,3]:assert (BASE/f'phase2_gpu{gpu}_complete.json').exists()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
manifest={'created':time.time(),'script_hashes':{p.name:sha(p) for p in BASE.glob('*.py')},'encoders':{},'completed_probes':{},'gpu_snapshot':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv'],text=True)}
for k,v in m['models'].items():
 p=Path(v['encoder']);manifest['encoders'][k]={'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
 d=BASE/'jobs'/v['run'];manifest['completed_probes'][k]=json.loads((d/'guard_result.json').read_text())
root=Path('/opt/data/3_117/tanghr')
refs=[root/'DINOv3/dinov3/dinov3/train/train.py',root/'DINOv3/dinov3/dinov3/train/ssl_meta_arch.py',root/'Med_DINOv3/MedDINOv3/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/dinov3/models/primus.py',root/'Med_DINOv3/MedDINOv3/nnUNet/nnunetv2/preprocessing/normalization/default_normalization_schemes.py']
manifest['implementation_sources']={str(p):sha(p) for p in refs}
(BASE/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2))
archive=BASE/'MitoTopo_CMCC_CMCT_Gram_validation.zip'
allowed={'.py','.json','.jsonl','.md','.yaml','.yml','.txt','.csv','.log','.png'}
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as tar:
 for p in sorted(BASE.rglob('*')):
  rel=p.relative_to(BASE)
  if not p.is_file() or p.suffix not in allowed or rel.parts[0] in {'cache','guard_testdata','__pycache__'}:continue
  tar.write(p,arcname=str(Path('mitotopo_round3')/rel))
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'models':len(m['models'])}),flush=True)
