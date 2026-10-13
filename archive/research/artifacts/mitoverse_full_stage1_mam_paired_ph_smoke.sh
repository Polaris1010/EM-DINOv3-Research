set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
PYTHONPATH="$run/topology_feasibility/vendor:$run/scripts" /opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import numpy as np
from pathlib import Path
from topology_raw_ph import diagrams,distance
p=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120/topology_feasibility/mam_sam_vit_b/paired/maps/sample_000000.npz')
with np.load(p) as z: raw=z['raw']; mam=z['mam']
for name,arr in [('raw_em',raw),('mam_affinity',mam),('mam_boundary',1-mam)]:
 base=diagrams(arr[0]); flip=diagrams(arr[1]); crop=diagrams(arr[9])
 print(name,'shape',arr.shape,'range',float(arr.min()),float(arr.max()),'std',float(arr.std()),'H0/H1',len(base[0]),len(base[1]),'flip',distance(base[0],flip[0]),distance(base[1],flip[1]),'aggressive_crop',distance(base[0],crop[0]),distance(base[1],crop[1]))
PY
