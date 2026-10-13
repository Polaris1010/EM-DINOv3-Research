set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
test ! -e "$run/runs/B_full/formal"
test "$(/opt/data/3_117/tanghr/envs/meddinov3/bin/python -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$run/audit/formal_B_full_admission.json")" = admitted
test "$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:7860/)" = 200
for spec in 1:600 2:600 3:100; do
  gpu=${spec%%:*}; limit=${spec##*:}
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
  test "$used" -lt "$limit"
done
systemctl show mitoverse-stage1-B-full.service -p ActiveState -p MainPID --no-pager || true
systemd-run --unit=mitoverse-stage1-B-full.service \
  --description='Full MitoVerse Stage 1 vanilla DINOv3 B_full 100k' \
  --property=Restart=no --property=KillMode=control-group --property=TimeoutStopSec=300 \
  /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u "$run/scripts/gpu_guard.py"
sleep 30
systemctl show mitoverse-stage1-B-full.service -p ActiveState -p MainPID -p Result --no-pager
cat "$run/STATUS.json"
if test -f "$run/runs/B_full/formal/progress.json"; then cat "$run/runs/B_full/formal/progress.json"; fi
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
