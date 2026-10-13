set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
unit=mito-pretrain-v2-seed0
test -s "$run/evidence/formal_seed0_admission.json"
test ! -e "$run/src/STOP.json"
if systemctl is-active --quiet "$unit"; then
  echo "Service is already active; refusing duplicate launch"
  exit 2
fi
systemd-run --unit="$unit" --property="WorkingDirectory=$run/src" \
  --property=KillMode=control-group --property=Restart=no \
  /bin/bash -c "exec /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u $run/src/coordinator_formal_seed0.py >> $run/src/formal_coordinator.log 2>&1"
sleep 5
systemctl show "$unit" -p ActiveState -p SubState -p MainPID -p Result --no-pager
if [ -f "$run/src/formal_seed0_status.json" ]; then
  cat "$run/src/formal_seed0_status.json"
fi
