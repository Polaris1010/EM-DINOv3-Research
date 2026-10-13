set -e
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import sys,json,hashlib,time
from pathlib import Path
sys.path.insert(0,'/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps')
import zarr,numpy as np
root=Path('/opt/data/3_117/tanghr/MitoVerse_data')
out=Path('/opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013/evidence');out.mkdir(parents=True,exist_ok=True)
cat=json.loads((root/'catalog.json').read_text());rows=[]
for vid,v in cat.items():
 p=root/v['zarr'];g=zarr.open_group(str(p),mode='r');arrays={k:{'shape':list(g[k].shape),'dtype':str(g[k].dtype),'chunks':list(g[k].chunks)} for k in g.array_keys()}
 row={'id':vid,'path':str(p),'catalog':v,'attrs':dict(g.attrs),'arrays':arrays,'metadata_sha256':{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p/'.zattrs',p/'img/.zarray',p/'mito/.zarray'] if f.exists()}}
 # QC on predetermined slices; never use test/unknown labels to define a split.
 if v['dataset_id'] in {'guay21','jiang25','muller21','casser20','haberl18','mekuc20','kunduri22','han24'}:
  row['array_metadata_valid']=(arrays.get('img',{}).get('shape')==arrays.get('mito',{}).get('shape')==v['shape_zyx'])
 rows.append(row)
result={'collected_at':time.time(),'catalog_sha256':hashlib.sha256((root/'catalog.json').read_bytes()).hexdigest(),'volumes':rows,'split_files':{f.name:json.loads(f.read_text()) for f in (root/'splits').glob('*.json')},'readme':(root/'README.md').read_text(),'todo':(root/'TODO.md').read_text()}
(out/'live_inventory.json').write_text(json.dumps(result,indent=2))
print('SAVED',str(out/'live_inventory.json'),'volumes',len(rows))
for s in ['guay21','jiang25','muller21','casser20','haberl18','mekuc20','kunduri22','han24']:
 print(s,[(r['id'],r['arrays'],r['attrs'].get('mitoverse',{}).get('provenance')) for r in rows if r['catalog']['dataset_id']==s])
base=Path('/opt/data/3_117/tanghr')
for p in base.iterdir():
 if p.is_dir():print('BASEDIR',p.name)
print('TODO',result['todo'])
PY
