set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
date -u
stat -c 'heartbeat_mtime=%y heartbeat_epoch=%Y' "$run/audit/gpu_guard_heartbeat.txt"
cat "$run/audit/gpu_guard_heartbeat.txt"
cat "$run/runs/B_full/formal/result.json"
find "$run/runs/B_full/formal" -maxdepth 2 -type f -printf '%s %p\n' | sort -n | tail -20
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - "$run" <<'PY'
import json,sys,time
from pathlib import Path
run=Path(sys.argv[1]); p=run/'audit/gpu_guard_heartbeat.txt'
print('now',time.time(),'mtime',p.stat().st_mtime,'age',time.time()-p.stat().st_mtime)
for line in (run/'audit/gpu_guard.jsonl').read_text().splitlines()[-3:]:
 x=json.loads(line); print('snapshot_time',x['time_unix'],'unknown',x['unknown_apps'],'website',x['website_http_200'])
PY
