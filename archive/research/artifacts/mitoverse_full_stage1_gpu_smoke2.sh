set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
test "$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:7860/)" = 200
gpu3_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -d ' ')
test "$gpu3_used" -lt 100
test ! -e "$run/runs/B_full/smoke2"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-5a68cc29-a13a-bf90-858e-079e14de807a \
PYTHONPATH=/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps:/opt/data/3_117/tanghr/DINOv3/dinov3 \
/opt/data/3_117/tanghr/dinov3_shared/bin/python "$run/scripts/train_B_full.py" \
  --manifest "$run/data/pretrain_all_mitoverse.json" \
  --config "$run/configs/B_full.yaml" \
  --weight /opt/data/3_117/tanghr/DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --output "$run/runs/B_full/smoke2" --steps 2 --seed 0
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv,noheader
