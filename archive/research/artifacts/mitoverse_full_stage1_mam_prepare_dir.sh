set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
mkdir -p "$run/topology_feasibility/mam_sam_vit_b/weights" "$run/topology_feasibility/mam_sam_vit_b/maps" "$run/topology_feasibility/mam_sam_vit_b/figures"
/opt/data/3_117/tanghr/envs/microsam/bin/python - <<'PY'
import importlib.util
print({m:bool(importlib.util.find_spec(m)) for m in ['sklearn','scipy','imageio','cv2','segment_anything','torch','zarr','matplotlib']})
PY
