"""Materialize the exact completed30k readout inputs in a separate writable run root."""
import hashlib,json,os,shutil
from pathlib import Path

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare(root,parent):
 root=Path(root).resolve();parent=Path(parent).resolve()
 if root==parent or root.is_relative_to(parent) or parent.is_relative_to(root):
  raise ValueError('Readout output must be separate from completed experiment')
 source=parent/'downstream';target=root/'downstream';target.mkdir(parents=True,exist_ok=True)
 files=['MV3_protocol.json','MV-Generalist.json','src/data.py']
 for name in files:
  old=source/name;new=target/name
  if not old.is_file():raise FileNotFoundError(old)
  new.parent.mkdir(parents=True,exist_ok=True)
  if new.exists():
   if digest(old)!=digest(new):raise ValueError('Existing readout input differs: '+name)
  else:shutil.copyfile(old,new)
 protocol=json.loads((target/'MV3_protocol.json').read_text());manifest=json.loads((target/'MV-Generalist.json').read_text())
 if protocol['models']!={'D1':'D1','D2':'D2'} or protocol['optimizer_steps']!=40000 or protocol['tta']:
  raise ValueError('Readout training protocol drift')
 counts={role:sum(v['supervised_role']==role for v in manifest['entries']) for role in ['train','dev']}
 if counts!={'train':156,'dev':38}:raise ValueError('Unexpected supervision split')
 cache=(source/'cache').resolve(strict=True);link=target/'cache'
 if link.exists():
  if link.resolve()!=cache:raise ValueError('Existing cache points to different supervision')
 elif link.is_symlink():raise ValueError('Existing cache link is broken')
 else:os.symlink(cache,link,target_is_directory=True)
 manifest_sha=digest(target/'MV-Generalist.json')
 for row in manifest['entries']:
  receipt=json.loads((cache/row['id']/'CACHE.json').read_text())
  if receipt['manifest_sha256']!=manifest_sha:raise ValueError('Cache belongs to different manifest')
  for path in [*receipt['paths'].values(),receipt.get('valid_path')]:
   if path and not Path(path).is_file():raise FileNotFoundError(path)
  if row['source_id']=='han24' and receipt.get('valid_path') is not None:raise ValueError('Han fullGT policy drift')
 return {'status':'PASS','copied_inputs':{name:digest(target/name) for name in files},'cache_read_only_usage':'existing mmap r; no data.prepare call','cache_target':str(cache),'regions':counts,'sources':len({v['source_id'] for v in manifest['entries']}),'tta':False,'optimizer_steps':40000}
