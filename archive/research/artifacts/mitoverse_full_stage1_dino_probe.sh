set -euo pipefail
base=/opt/data/3_117/tanghr/DINOv3/dinov3
find "$base/dinov3" -maxdepth 3 -type f \( -name 'train.py' -o -name 'ssl_meta_arch.py' -o -name '*parallelize*.py' \) -print
rg -n 'def (forward_backward|update_ema|get_params_groups|prepare_for_distributed_training)|parallelize|DistributedDataParallel|FSDP|fully_shard' "$base/dinov3/train/ssl_meta_arch.py" "$base/dinov3/train/train.py" "$base/dinov3/fsdp" | head -110
/opt/data/3_117/tanghr/dinov3_shared/bin/python - <<'PY'
import importlib.util
for name in ('zarr','omegaconf','torch','numcodecs','PIL'):
 print(name, bool(importlib.util.find_spec(name)))
PY
