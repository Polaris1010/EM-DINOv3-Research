set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_restart_20260928_133515
registry="$run/src/process_registry.json"
test -f "$registry"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - "$run" "$registry" <<'PY'
import json, os, psutil, signal, sys, time
from pathlib import Path
run, registry = map(Path, sys.argv[1:])
expected = {'formal_seed0_B_part0', 'formal_seed0_L_part0', 'formal_seed0_T_part0'}
jobs = json.loads(registry.read_text())
selected = {name: row for name, row in jobs.items() if name in expected}
if set(selected) != expected:
    raise SystemExit(f'Worker registry changed; found {sorted(selected)}')
for name, row in selected.items():
    process = psutil.Process(row['pid'])
    if abs(process.create_time() - row['birth']) > 0.1:
        raise SystemExit(f'Process identity changed: {name}')
    cmd = process.cmdline()
    if str(run / 'src' / 'train_ssl.py') not in cmd or name.split('_')[2] not in cmd:
        raise SystemExit(f'Unexpected command for {name}: {cmd}')
    if os.getpgid(process.pid) != process.pid:
        raise SystemExit(f'Worker is not its own process group: {name}')
print('Verified owned worker groups:', [(n, r['pid']) for n, r in selected.items()], flush=True)
marker = run / 'src' / 'STOP.json'
if not marker.exists():
    temporary = marker.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({'reason': 'Superseded by user Stage 1 B_full-only protocol', 'time': time.time()}) + '\n')
    temporary.replace(marker)
for name, row in selected.items():
    try:
        os.killpg(row['pid'], signal.SIGTERM)
        print('SIGTERM sent to', name, row['pid'], flush=True)
    except ProcessLookupError:
        print('Already exited:', name, flush=True)
PY
sleep 35
systemctl stop mitoverse-full-seed0.service
systemctl show mitoverse-full-seed0.service -p ActiveState -p MainPID -p Result --no-pager
for arm in B L T; do
  job="$run/src/jobs/formal_seed0_${arm}_part0"
  printf '%s ' "$arm"
  if test -f "$job/latest_checkpoint.json"; then cat "$job/latest_checkpoint.json"; else echo no_checkpoint_receipt; fi
  if test -f "$job/result.json"; then cat "$job/result.json"; else echo no_result; fi
done
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
