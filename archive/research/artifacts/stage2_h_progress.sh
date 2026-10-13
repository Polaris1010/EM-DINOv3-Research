set -e
ROOT=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2_20260930_082816
cd "$ROOT"
date -u +%FT%TZ
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
printf 'paired_maps=%s\n' "$(find topology_calibration/sam_vit_h/paired_maps -maxdepth 1 -name 'sample_*.npz' | wc -l)"
for SHARD in 0 1 2; do
  PID=$(cat "topology_calibration/sam_vit_h/logs/shard${SHARD}.pid")
  if kill -0 "$PID" 2>/dev/null; then STATE=running; else STATE=stopped; fi
  printf 'shard=%s pid=%s state=%s\n' "$SHARD" "$PID" "$STATE"
  tail -1 "topology_calibration/sam_vit_h/logs/shard${SHARD}.log" | cut -c1-260 || true
done
printf 'website_http=%s\n' "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:7860/)"
