set -euo pipefail
cd /opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import ast,json,sys,collections,numpy as np
sys.path.insert(0,'/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps')
import zarr
ref=json.loads(Path('references/paths.json').read_text());stage1=Path(ref['stage1_root'])
samples=json.loads(Path('references/sample_manifest_424.json').read_text())['samples']
print('SAMPLE_EXAMPLE',json.dumps(samples[0]))
p=stage1/'scripts/mam_sam_vit_b_smoke.py'
t=p.read_text();tree=ast.parse(t)
for n in tree.body:
    if isinstance(n,ast.FunctionDef) and n.name=='read_patch':print('READ_PATCH',ast.get_source_segment(t,n))
counts=collections.Counter();examples=[]
for s in samples:
    p=Path(s['local_path']);g=zarr.open_group(str(p),mode='r')
    counts['samples']+=1;counts['has_mito']+=int('mito' in g);counts['has_mask']+=int('mask' in g)
    if 'mito' in g:counts['shape_match']+=int(g['img'].shape==g['mito'].shape)
    if len(examples)<4 and (not examples or s['source'] not in {x['source'] for x in examples}):
        examples.append({'source':s['source'],'path':str(p),'arrays':list(g.array_keys()),'attrs':dict(g.attrs)})
print('GT_COUNTS',json.dumps(dict(counts)))
print('GT_EXAMPLES',json.dumps(examples,default=str))
PY
