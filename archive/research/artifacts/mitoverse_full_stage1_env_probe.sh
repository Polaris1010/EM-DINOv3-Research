set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
old_deps=/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps
du -sh "$old_deps" 2>/dev/null || true
find "$old_deps" -maxdepth 1 -iname 'zarr*' -o -iname 'numcodecs*' | head -20
PYTHONPATH="$old_deps:/opt/data/3_117/tanghr/DINOv3/dinov3" /opt/data/3_117/tanghr/dinov3_shared/bin/python - <<'PY'
import torch,zarr,numcodecs,omegaconf
print('torch',torch.__version__,'zarr',zarr.__version__,'numcodecs',numcodecs.__version__,'omegaconf',omegaconf.__version__)
PY
test -f "$run/configs/B_full_resolved.yaml"
sha256sum "$run/configs/B_full.yaml" "$run/configs/B_full_resolved.yaml" "$run/scripts/ssl_full_data.py"
