set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
test "$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:7860/)" = 200
for spec in 1:600 2:600 3:100; do
  gpu=${spec%%:*}; limit=${spec##*:}
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
  test "$used" -lt "$limit"
done
test ! -e "$run/runs/B_full/distributed_smoke2"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-29591cdf-1498-85e8-bb8c-158082a53728,GPU-794aaaef-6c72-f6ac-b6d0-a07ba3f76364,GPU-5a68cc29-a13a-bf90-858e-079e14de807a
export PYTHONPATH=/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps:/opt/data/3_117/tanghr/DINOv3/dinov3
export OMP_NUM_THREADS=2
python=/opt/data/3_117/tanghr/dinov3_shared/bin/python
"$python" -m torch.distributed.run --standalone --nproc_per_node=3 "$run/scripts/train_B_full.py" \
  --manifest "$run/data/pretrain_all_mitoverse.json" \
  --config "$run/configs/B_full.yaml" \
  --weight /opt/data/3_117/tanghr/DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --output "$run/runs/B_full/distributed_smoke2" --steps 2 --seed 0 \
  > "$run/runs/B_full/distributed_smoke2_launcher.log" 2>&1 || { tail -n 70 "$run/runs/B_full/distributed_smoke2_launcher.log"; exit 1; }
cat "$run/runs/B_full/distributed_smoke2/result.json"
for rank in 0 1 2; do
  tail -n 1 "$run/runs/B_full/distributed_smoke2/steps_rank${rank}.jsonl"
done
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
