set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
systemctl show mito-pretrain-v2-seed0 -p ActiveState -p MainPID -p Result --no-pager
systemctl show mito-pretrain-v2-watcher -p ActiveState -p MainPID -p Result --no-pager
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - "$run" <<'PY'
import json,pathlib,sys,time
r=pathlib.Path(sys.argv[1]);src=r/'src'
s=json.loads((r/'stage_status.json').read_text())
print('stage_updated',s['last_updated_utc'])
print('stages',{k:v['status'] for k,v in s.items() if k[:2].isdigit() and isinstance(v,dict) and 'status' in v})
print('new_ssl_updates',s['new_ssl_updates'])
print('stop_marker',(src/'STOP.json').exists())
h=json.loads((src/'coordinator.heartbeat').read_text())
print('heartbeat_age_seconds',round(time.monotonic()-h['monotonic'],1))
PY
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
tail -n 4 "$run/src/watcher.log"
