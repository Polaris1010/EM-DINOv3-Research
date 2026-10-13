set -euo pipefail
date -u '+%Y-%m-%d %H:%M:%S UTC'
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json,pathlib
r=pathlib.Path('/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227');s=r/'src'
for name in ('overlap_status.json','expanded_status.json','STOP.json'):
 if (s/name).exists():print(name,json.loads((s/name).read_text()))
for name in ('formal_seed0_LT_part1','probe_N0_0_head0','probe_B_100000_head0'):
 p=s/'jobs'/name
 if (p/'progress.json').exists():
  d=json.loads((p/'progress.json').read_text());print(name,{k:d[k] for k in ('step','loss','seconds')})
 for v in sorted(p.glob('validation_step*.json')):
  d=json.loads(v.read_text());print(name,v.name,'dice',d['source_macro_dice'])
print('coverage',json.loads((r/'evidence/expanded_coverage.json').read_text()))
PY
systemctl is-active mito-pretrain-v2-seed0 mito-pretrain-v2-watcher
