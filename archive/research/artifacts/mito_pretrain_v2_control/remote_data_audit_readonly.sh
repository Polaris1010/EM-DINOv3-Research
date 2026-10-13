set -euo pipefail
base=/opt/data/3_117/tanghr
manifest="$base/codex/mitoverse_round6_20260917_044606/data_manifest_v3.csv"
sha256sum "$manifest"
PYTHONPATH="$base/codex/mitotopo_20260908_round2/deps${PYTHONPATH:+:$PYTHONPATH}" "$base/envs/meddinov3/bin/python" - "$manifest" <<'PY'
import collections,csv,json,os,sys
import numpy as np
import zarr

path=sys.argv[1]
rows=list(csv.DictReader(open(path,encoding='utf-8-sig',newline='')))
active=[r for r in rows if r['split'] in {'train','validation','legacy_dev_test'}]
splits=collections.defaultdict(set)
sources=collections.Counter()
errors=[]
for r in active:
    vid=r['volume_id']; split=r['split']; p=r['local_path']
    canonical=os.path.realpath(p)
    if canonical in splits[split]: errors.append(f'duplicate path within {split}: {vid}')
    splits[split].add(canonical)
    sources[(split,r['original_dataset_id'])]+=1
    try:
        g=zarr.open_group(p,mode='r')
        im=g['img']; lab=g['mito']
        if len(im.shape)!=3 or im.shape!=lab.shape: errors.append(f'shape mismatch: {vid} {im.shape} {lab.shape}')
        if 'mask' in g and g['mask'].shape!=im.shape: errors.append(f'mask shape mismatch: {vid}')
        z,y,x=(int(n//2) for n in im.shape)
        a=np.asarray(im[z,y:y+1,x:x+1]); b=np.asarray(lab[z,y:y+1,x:x+1])
        if a.size!=1 or b.size!=1 or not np.isfinite(a).all() or not np.isfinite(b).all(): errors.append(f'sample invalid: {vid}')
    except Exception as e:
        errors.append(f'{vid}: {type(e).__name__}: {e}')
for a,b in [('train','validation'),('train','legacy_dev_test'),('validation','legacy_dev_test')]:
    if splits[a]&splits[b]: errors.append(f'path overlap: {a} {b}')
print(json.dumps({'total_rows':len(rows),'active_rows':len(active),'split_counts':{k:len(v) for k,v in splits.items()},'source_counts':{str(k):v for k,v in sources.items()},'errors':errors[:30],'error_count':len(errors)},ensure_ascii=False))
if errors: raise SystemExit(2)
PY
