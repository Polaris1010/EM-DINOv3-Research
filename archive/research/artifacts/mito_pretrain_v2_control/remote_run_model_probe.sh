set -euo pipefail
base=/opt/data/3_117/tanghr
run="$base/codex/mito_pretrain_v2_20260926_060227"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$base/envs/meddinov3/bin/python" "$run/src/model_interface_probe.py" \
  --dinov3-root "$base/DINOv3/dinov3" \
  --weights "$base/DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
  --output "$run/evidence/model_interface_probe.json"
