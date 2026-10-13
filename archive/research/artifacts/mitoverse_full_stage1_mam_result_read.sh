set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
paired="$run/topology_feasibility/mam_sam_vit_b/paired"
ls -lh "$paired"/{generation_complete.json,ph_complete.json,metrics_paired.csv,summary.json,report.md}
cat "$paired/ph_complete.json"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120/topology_feasibility/mam_sam_vit_b/paired/summary.json')
d=json.loads(p.read_text())
print('rows',d['rows'],'samples',d['samples'],'sources',d['sources'])
for mode in ('raw_em_paired','mam_affinity','mam_boundary'):
 print('MODE',mode)
 for name in ('identity','flip_horizontal','flip_vertical','weak_jitter','strong_jitter','weak_blur','strong_blur','mild_crop','medium_crop','aggressive_crop'):
  v=d['modes'][mode][name]
  print(name,'d0_p50',round(v['d0']['p50'],4),'d1_p50',round(v['d1']['p50'],4),'sum_p50',round(v['d_equal']['p50'],4),'sum_p90',round(v['d_equal']['p90'],4))
 print('paired_fraction',d['modes'][mode]['pairwise_fraction'])
 for aug in ('flip_horizontal','aggressive_crop'):
  v=[x[aug] for x in d['source_medians'][mode].values()]
  print('source_median_range',aug,round(min(v),4),round(max(v),4))
PY
tail -n 5 "$paired/summary.log"
