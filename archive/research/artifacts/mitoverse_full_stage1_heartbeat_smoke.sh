set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
heartbeat="$run/audit/heartbeat_test.txt"
test ! -e "$run/runs/B_full/distributed_heartbeat2"
date +%s > "$heartbeat"
touch -d '20 minutes ago' "$heartbeat"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-29591cdf-1498-85e8-bb8c-158082a53728,GPU-794aaaef-6c72-f6ac-b6d0-a07ba3f76364,GPU-5a68cc29-a13a-bf90-858e-079e14de807a
export PYTHONPATH=/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps:/opt/data/3_117/tanghr/DINOv3/dinov3:"$run/scripts"
export STAGE1_GUARD_HEARTBEAT="$heartbeat"
/opt/data/3_117/tanghr/dinov3_shared/bin/python - "$heartbeat" <<'PY'
import sys,time
from pathlib import Path
from train_B_full import guard_is_stale
path=Path(sys.argv[1])
assert time.time()-path.stat().st_mtime > 1000
assert guard_is_stale(str(path)) is False
path.write_text(str(time.time()-300)+'\n')
assert guard_is_stale(str(path)) is True
path.write_text(str(time.time())+'\n')
print('heartbeat_content_test=PASS')
PY
test "$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:7860/)" = 200
for spec in 1:600 2:600 3:100; do
  gpu=${spec%%:*}; limit=${spec##*:}
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
  test "$used" -lt "$limit"
done
export OMP_NUM_THREADS=2
/opt/data/3_117/tanghr/dinov3_shared/bin/python -m torch.distributed.run --standalone --nproc_per_node=3 "$run/scripts/train_B_full.py" \
  --manifest "$run/data/pretrain_all_mitoverse.json" --config "$run/configs/B_full.yaml" \
  --weight /opt/data/3_117/tanghr/DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --output "$run/runs/B_full/distributed_heartbeat2" --steps 2 --seed 0 \
  > "$run/runs/B_full/distributed_heartbeat2.log" 2>&1 || { tail -n 70 "$run/runs/B_full/distributed_heartbeat2.log"; exit 1; }
cat "$run/runs/B_full/distributed_heartbeat2/result.json"
