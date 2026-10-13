set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
mam="$run/topology_feasibility/mam_sam_vit_b"
systemctl show mitoverse-stage1-mam-smoke.service -p ActiveState -p MainPID -p Result --no-pager
if test -f "$mam/smoke_result.json"; then cat "$mam/smoke_result.json"; else tail -n 15 "$mam/smoke.log"; fi
systemctl show mitoverse-stage1-B-full-v2.service -p ActiveState -p MainPID -p Result --no-pager
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json
p=json.load(open('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120/runs/B_full/formal_attempt1/progress.json'))
print('B_full_step',p['step'],'loss',p['loss'])
PY
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
