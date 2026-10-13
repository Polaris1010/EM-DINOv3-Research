set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
test -f branch_E/runs/B_full_CMCC_smoke/result.json
test "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:7860/)" = 200
for GPU in 1 2; do
  USED=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  test "$USED" -lt 1024
done
for ITEM in '1 Dataset111_CMCC' '2 Dataset131_CMCT'; do
  set -- $ITEM
  GPU=$1; DS=$2
  OUT="$RUN/branch_E/runs/B_full_${DS}"
  mkdir -p "$OUT"
  nohup env RUN_OUTPUT="$OUT" CUDA_VISIBLE_DEVICES="$GPU" OPENBLAS_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/meddinov3/bin/python src/frozen_probe.py --dataset "$DS" --steps 3000 --seed 20260907 > "$OUT/train.log" 2>&1 < /dev/null &
  PID=$!
  echo "$PID" > "$OUT/train.pid"
  printf 'dataset=%s gpu=%s pid=%s\n' "$DS" "$GPU" "$PID"
done
