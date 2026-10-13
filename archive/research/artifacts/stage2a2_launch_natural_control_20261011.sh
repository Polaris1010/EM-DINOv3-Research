set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json
from pathlib import Path
assert json.loads(Path('branch_E/control_20261011/protocol_match.json').read_text())['status']=='pass'
assert json.loads(Path('branch_E/control_20261011/official_CMCC_smoke/result.json').read_text())['status']=='smoke_pass'
PY
test "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:7860/)" = 200
for GPU in 1 2; do
  test "$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 1024
done
for ITEM in '1 Dataset111_CMCC' '2 Dataset131_CMCT'; do
  set -- $ITEM
  GPU=$1; DS=$2
  UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
  OUT="$RUN/branch_E/control_20261011/runs/official_natural_${DS}"
  test ! -e "$OUT"
  mkdir -p "$OUT"
  nohup env RUN_OUTPUT="$OUT" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$UUID" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/meddinov3/bin/python src/frozen_probe_control.py --dataset "$DS" --encoder-kind official_natural --steps 3000 --seed 20260907 > "$OUT/train.log" 2>&1 < /dev/null &
  PID=$!
  echo "$PID" > "$OUT/train.pid"
  printf 'dataset=%s gpu=%s uuid=%s pid=%s\n' "$DS" "$GPU" "$UUID" "$PID"
done
tail -3 branch_S/blind_review_raw_20261011/render.log
