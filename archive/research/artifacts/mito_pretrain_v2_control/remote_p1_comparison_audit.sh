set -euo pipefail
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import json
root=Path('/opt/data/3_117/tanghr/codex')
for r in root.glob('*round6*'):
 print('ROUND6',r)
 for name in ('stage_status.json','mitonet_comparison_summary.json','exploratory_baseline_protocol.json'):
  p=r/name
  if p.exists():print(name,p.read_text())
 for p in sorted((r/'jobs').glob('*')):
  if any(s in p.name for s in ('mitonet_M0','masked_P1','legacy')):
   print('JOB',p.name,'files',[x.name for x in p.iterdir() if x.is_file()],'pred_volumes',len(list((p/'predictions').glob('*.zarr'))))
 print('CSVS',[p.name for p in r.glob('*comparison*')])
v=root/'mito_pretrain_v2_20260926_060227'
print('CURRENT', (v/'stage_status.json').read_text())
PY
