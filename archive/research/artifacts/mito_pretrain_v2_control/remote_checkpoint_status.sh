set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
for arm in B L T; do
  job="$run/src/jobs/formal_seed0_${arm}_part0"
  echo "arm=$arm"
  if [ -f "$job/latest_checkpoint.json" ]; then cat "$job/latest_checkpoint.json"; fi
  if [ -f "$job/checkpoint_step1000.pth" ]; then stat -c 'checkpoint_step1000_bytes=%s' "$job/checkpoint_step1000.pth"; fi
  if [ -f "$job/progress.json" ]; then /opt/data/3_117/tanghr/envs/meddinov3/bin/python - "$job/progress.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));print('progress_step',x['step'])
PY
  fi
done
test ! -e "$run/src/STOP.json"
systemctl show mito-pretrain-v2-seed0 -p ActiveState -p MainPID --no-pager
