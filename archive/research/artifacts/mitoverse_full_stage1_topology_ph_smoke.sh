set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
PYTHONPATH="$run/topology_feasibility/vendor:$run/scripts" /opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json,topology_raw_ph as p
manifest=json.load(open(p.TOP/'sample_manifest.json'))
x=p.image_for_sample(manifest['samples'][0])
a=p.diagrams(x)
print('image',x.shape,x.min(),x.max(),'H0',len(a[0]),'H1',len(a[1]))
for name,y in p.augmentations(x).items():
 b=p.diagrams(y)
 print(name,p.distance(a[0],b[0]),p.distance(a[1],b[1]))
PY
