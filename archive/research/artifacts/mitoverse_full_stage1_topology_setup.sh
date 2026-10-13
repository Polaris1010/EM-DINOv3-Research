set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
top="$run/topology_feasibility"
py=/opt/data/3_117/tanghr/envs/meddinov3/bin/python
"$py" -m pip install --no-index --no-deps --target "$top/vendor" "$top"/gudhi-3.13.0-*.whl
PYTHONPATH="$top/vendor" "$py" - <<'PY'
import gudhi,json,importlib.util
from pathlib import Path
p=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120/topology_feasibility/sample_manifest.json')
d=json.loads(p.read_text())
print('gudhi',gudhi.__version__)
print('sample_type',type(d).__name__,'length',len(d))
print('first',str(d[0] if isinstance(d,list) else next(iter(d.items())))[:2000])
for name in ('numpy','scipy','zarr','skimage','matplotlib'):
 print(name,bool(importlib.util.find_spec(name)))
PY
