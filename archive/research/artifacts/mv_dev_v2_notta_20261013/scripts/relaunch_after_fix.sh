set -euo pipefail
TASK=/opt/data/3_117/tanghr/codex/mv_dev_v2_notta_20261013
test -f "$TASK/evidence/hd95_size_limit_fix.json"
test "$(systemctl is-active mv-dev-v2-notta-20261013.service || true)" != active
systemd-run --unit=mv-dev-v2-notta-20261013 --property=WorkingDirectory="$TASK" --property=StandardOutput=append:"$TASK/logs/runner.log" --property=StandardError=append:"$TASK/logs/runner.log" /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u "$TASK/scripts/run_validation.py"
systemctl is-active mv-dev-v2-notta-20261013.service
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - "$TASK" <<'PY'
import pathlib,sys,json,time
r=pathlib.Path(sys.argv[1])
for _ in range(25):
 if json.loads((r/'STATUS.json').read_text()).get('status')=='running':break
 time.sleep(2)
else:raise RuntimeError('Resume did not publish running state')
print('RUNNING', (r/'STATUS.json').read_text())
PY
systemd-run --unit=mv-dev-v2-notta-delivery-resume1-20261013 --property=WorkingDirectory="$TASK" --property=StandardOutput=append:"$TASK/logs/delivery.log" --property=StandardError=append:"$TASK/logs/delivery.log" /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u "$TASK/scripts/deliver.py"
