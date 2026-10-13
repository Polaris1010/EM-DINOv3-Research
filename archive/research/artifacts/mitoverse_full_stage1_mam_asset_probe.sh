set -euo pipefail
base=/opt/data/3_117/tanghr
echo 'directory candidates'
find "$base" -maxdepth 5 -type d \( -iname '*orgmim*' -o -iname '*sam*' -o -iname '*segment_anything*' -o -iname '*topocl*' \) -print | head -60
echo 'checkpoint candidates'
find "$base" -maxdepth 6 -type f \( -iname '*sam*.pth' -o -iname '*sam*.pt' -o -iname '*orgmim*.pth' -o -iname '*orgmim*.pt' -o -iname '*vit_h*.pth' -o -iname '*vit_l*.pth' -o -iname '*vit_b*.pth' \) -printf '%p %s bytes\n' | head -80
echo 'python packages'
for py in "$base/dinov3_shared/bin/python" "$base/envs/meddinov3/bin/python"; do
  "$py" - <<'PY'
import importlib.util,sys
print(sys.executable, {m:bool(importlib.util.find_spec(m)) for m in ['segment_anything','sam2','transformers','timm','torch','cv2']})
PY
done
echo 'network probes'
getent ahostsv4 github.com | head -1 || true
getent ahostsv4 huggingface.co | head -1 || true
