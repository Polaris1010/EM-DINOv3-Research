set -e
ROOT=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2_20260930_082816
date -u +%FT%TZ
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
free -h | head -2
ls -lh "$ROOT/topology_calibration/sam_vit_h"/*.pth* 2>/dev/null || true
ls "$ROOT/topology_calibration/sam_vit_h/paired_maps"/sample_*.npz 2>/dev/null | wc -l
cat "$ROOT/topology_calibration/sam_vit_h/generation_progress.json" 2>/dev/null || true
