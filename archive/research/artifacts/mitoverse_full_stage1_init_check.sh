set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
test "$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:7860/)" = 200
gpu3_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -d ' ')
test "$gpu3_used" -lt 100
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-5a68cc29-a13a-bf90-858e-079e14de807a \
PYTHONPATH=/opt/data/3_117/tanghr/DINOv3/dinov3 \
/opt/data/3_117/tanghr/dinov3_shared/bin/python "$run/scripts/validate_initialization.py" \
  --dinov3-root /opt/data/3_117/tanghr/DINOv3/dinov3 \
  --override "$run/configs/B_full.yaml" \
  --weight /opt/data/3_117/tanghr/DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --output "$run/audit/initialization_check.json" \
  --resolved "$run/configs/B_full_resolved.yaml"
