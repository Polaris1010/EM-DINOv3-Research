set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
paired="$run/topology_feasibility/mam_sam_vit_b/paired"
systemctl show mitoverse-stage1-mam-paired-full.service -p ActiveState -p MainPID -p Result -p CPUUsageNSec -p MemoryCurrent --no-pager
if test -f "$paired/generation_progress.json"; then cat "$paired/generation_progress.json"; fi
if test -f "$paired/generation_complete.json"; then cat "$paired/generation_complete.json"; fi
if test -f "$paired/ph_progress.json"; then cat "$paired/ph_progress.json"; fi
tail -n 5 "$paired/generation.log" 2>/dev/null || true
systemctl show mitoverse-stage1-B-full-v2.service -p ActiveState -p MainPID -p Result --no-pager
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json
from pathlib import Path
run=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120')
p=json.loads((run/'runs/B_full/formal_attempt1/progress.json').read_text())
c=json.loads((run/'runs/B_full/formal_attempt1/latest_checkpoint.json').read_text())
print('B_full_step',p['step'],'loss',p['loss'],'checkpoint',c['step'])
PY
free -h | head -2
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
