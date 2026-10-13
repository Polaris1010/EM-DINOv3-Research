set -euo pipefail
TASK=/opt/data/3_117/tanghr/codex/mv_dev_v2_notta_20261013
tail -n 95 "$TASK/logs/score.log"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - "$TASK" <<'PY'
import pathlib,sys,json,inspect
from monai.metrics.hausdorff_distance import _compute_percentile_hausdorff_distance
from monai.metrics.utils import get_surface_distance
r=pathlib.Path(sys.argv[1])
print('PERCENTILE_FUNCTION',inspect.getsource(_compute_percentile_hausdorff_distance))
print('DISTANCE_FUNCTION',inspect.getsource(get_surface_distance))
print('COUNTS',len(list(r.glob('runs/*/seed*/*_overlap.json'))),len(list(r.glob('runs/*/seed*/*_metrics.json'))))
print('SCORING_PROGRESS',(r/'scoring_progress.json').read_text())
for p in r.glob('runs/*/seed*/progress.json'):print('PROGRESS',p, p.read_text())
print('MASK_COUNTS')
for p in (r/'cache').glob('*/QC.json'):
 j=json.loads(p.read_text())
 if j.get('ignored_voxels',0):print(p.parent.name,j)
PY
systemctl show mv-dev-v2-notta-20261013.service --property=ExecMainStartTimestamp,ExecMainExitTimestamp,ExecStart
