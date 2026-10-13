set -euo pipefail
base=/opt/data/3_117/tanghr
for py in "$base/dinov3_shared/bin/python" "$base/envs/meddinov3/bin/python"; do
  "$py" - <<'PY'
import importlib.util,sys
print(sys.executable)
for name in ('gudhi','cv2','segment_anything','torch','zarr','scipy','skimage'):
 print(name,bool(importlib.util.find_spec(name)))
PY
done
find "$base" -maxdepth 3 \( -iname '*orgmim*' -o -iname '*sam_vit*' -o -iname '*sam2*' -o -iname '*topocl*' \) -print | head -50
git ls-remote https://github.com/gm3g11/TopoCL.git HEAD | head -1 || true
git ls-remote https://github.com/yanchaoz/OrgMIM.git HEAD | head -1 || true
