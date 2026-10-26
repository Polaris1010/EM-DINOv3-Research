"""Reuse the fixed MV-Generalist-v2 supervision/cache in the independent new root."""
import os,shutil
from pathlib import Path
from common import read,sha,atomic

def prepare(root,parent):
 root,parent=Path(root),Path(parent)
 assert root.resolve()!=parent.resolve() and not root.resolve().is_relative_to(parent.resolve())
 source=parent/'downstream';target=root/'downstream';target.mkdir(parents=True,exist_ok=True)
 for name in ['MV-Generalist.json','src/data.py']:
  dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert sha(dest)==sha(source/name)
  else:shutil.copyfile(source/name,dest)
 protocol=read(source/'MV3_protocol.json');assert protocol['optimizer_steps']==40000 and not protocol['tta']
 protocol['models']={'Coverage-NR':'Coverage-NR'}
 dest=target/'MV3_protocol.json'
 if dest.exists():assert read(dest)==protocol
 else:atomic(dest,protocol)
 manifest=read(target/'MV-Generalist.json')
 counts={role:sum(e['supervised_role']==role for e in manifest['entries']) for role in ['train','dev']}
 assert counts=={'train':156,'dev':38}
 cache=(source/'cache').resolve(strict=True);link=target/'cache'
 if link.exists():assert link.resolve()==cache
 else:os.symlink(cache,link,target_is_directory=True)
 for row in manifest['entries']:
  receipt=read(cache/row['id']/'CACHE.json');assert receipt['manifest_sha256']==sha(target/'MV-Generalist.json')
  for p in [*receipt['paths'].values(),receipt.get('valid_path')]:
   if p:assert Path(p).is_file()
  if row['source_id']=='han24':assert receipt.get('valid_path') is None
 return {'status':'PASS','regions':counts,'manifest_sha256':sha(target/'MV-Generalist.json'),
         'protocol_sha256':sha(dest),'cache_target':str(cache),'encoder_frozen':True,'tta':False}
