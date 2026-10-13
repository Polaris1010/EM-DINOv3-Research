set -euo pipefail
ROOT=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2_20260930_082816
cd "$ROOT"
test "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:7860/)" = 200
test "$(find topology_calibration/sam_vit_h/paired_maps -maxdepth 1 -name 'sample_*.npz' | wc -l)" -ge 1
for GPU in 1 2 3; do
  USED=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  UTIL=$(nvidia-smi -i "$GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  test "$USED" -lt 1024
  test "$UTIL" -lt 10
done
mkdir -p topology_calibration/sam_vit_h/logs
for SHARD in 0 1 2; do
  GPU=$((SHARD+1))
  LOG="topology_calibration/sam_vit_h/logs/shard${SHARD}.log"
  nohup env CUDA_VISIBLE_DEVICES="$GPU" OPENBLAS_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/microsam/bin/python src/sam_h_paired.py --limit 424 --shard "$SHARD" --num-shards 3 > "$LOG" 2>&1 < /dev/null &
  PID=$!
  echo "$PID" > "topology_calibration/sam_vit_h/logs/shard${SHARD}.pid"
  printf 'shard=%s gpu=%s pid=%s log=%s\n' "$SHARD" "$GPU" "$PID" "$LOG"
done
