set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
unit=mito-pretrain-v2-watcher
test -s "$run/src/watch_all_batches.py"
test -s "$run/src/coordinator_probes.py"
test -s "$run/src/coordinator_finetune.py"
test -s "$run/src/coordinator_independent.py"
test -s "$run/src/coordinator_final_eval.py"
test ! -e "$run/src/STOP.json"
if systemctl is-active --quiet "$unit"; then
  echo 'Watcher already active; refusing duplicate launch'
  exit 2
fi
systemd-run --unit="$unit" --property="WorkingDirectory=$run/src" \
  --property=Restart=no \
  /bin/bash -c "exec /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u $run/src/watch_all_batches.py >> $run/src/watcher.log 2>&1"
sleep 5
systemctl show "$unit" -p ActiveState -p SubState -p MainPID -p Result --no-pager
tail -n 15 "$run/src/watcher.log"
