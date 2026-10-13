set -u
base=/opt/data/3_117/tanghr
"$base/envs/meddinov3/bin/python" -V
"$base/envs/meddinov3/bin/python" - <<'PY'
import importlib.util,platform,sys
for name in ['gudhi','persim','scipy','skimage','zarr']:
 print(name, bool(importlib.util.find_spec(name)))
print('implementation',sys.implementation.name,'machine',platform.machine())
PY
