set -e
ROOT=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2_20260930_082816
cd "$ROOT"
date -u +%FT%TZ
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
printf 'website_http=%s\n' "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:7860/)"
if test -f topology_calibration/cache_benchmark/vit_h_raw_tile_10.json; then
  /opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('topology_calibration/cache_benchmark/vit_h_raw_tile_10.json').read_text())
print('tile_benchmark', {k:v for k,v in r.items() if k!='rows'})
PY
else
  tail -4 topology_calibration/cache_benchmark/vit_h_raw_tile_10.log
fi
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('topology_calibration/sam_vit_h/paired_B_H_summary.json').read_text())
for model in ('ViT-B','ViT-H'):
 print('prior',model, {k: round(r[model][k]['median'],5) for k in ('mean','std','dynamic_range','entropy_32','edge_density_005','neighbor_autocorr','h0_count','h1_count','h0_mass','h1_mass')})
e=json.loads(Path('topology_calibration/equivariance/vit_h_equivariance_424.json').read_text())
for op in ('hflip','vflip','rot90','rot180','rot270'):
 print('online_H',op,{k:round(e['groups']['online_H'][op][k]['median'],5) for k in ('mae','d0','d1')})
PY
