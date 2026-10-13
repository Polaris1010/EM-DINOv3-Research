set -euo pipefail
base=/opt/data/3_117/tanghr
pkill -f '^find /opt/data/3_117/tanghr -maxdepth 6 -type f' || true
for dir in "$base/MedSAM" "$base/microsam" "$base/SAM3" "$base/Med_SAM3" "$base/weights" "$base/checkpoints" "$base/pretrained_weights" "$base/.cache/huggingface"; do
  if test -d "$dir"; then
    echo "DIR $dir"
    find "$dir" -maxdepth 4 -type f \( -iname '*.pth' -o -iname '*.pt' -o -iname '*.safetensors' -o -iname '*.ckpt' \) -printf '%p %s bytes\n' | head -30
  fi
done
for py in "$base/envs/microsam/bin/python" "$base/envs/medsam/bin/python" "$base/envs/sam3/bin/python"; do
 if test -x "$py"; then "$py" - <<'PY'
import importlib.util,sys
print(sys.executable,{m:bool(importlib.util.find_spec(m)) for m in ['segment_anything','micro_sam','torch','cv2','zarr']})
PY
 fi
done
getent ahostsv4 github.com | head -1 || true
getent ahostsv4 huggingface.co | head -1 || true
