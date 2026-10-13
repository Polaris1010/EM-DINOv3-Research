set -euo pipefail
base=/opt/data/3_117/tanghr
run="$base/codex/mito_pretrain_v2_20260926_060227"
test -d "$run"
mkdir -m 750 "$run/src" "$run/data" "$run/evidence" "$run/logs"
for name in dinov3 meddinov3; do
  if [ "$name" = dinov3 ]; then repo="$base/DINOv3/dinov3"; else repo="$base/Med_DINOv3/MedDINOv3"; fi
  git -C "$repo" rev-parse HEAD > "$run/evidence/${name}_commit.txt"
  git -C "$repo" status --porcelain > "$run/evidence/${name}_status.txt"
  git -C "$repo" diff HEAD > "$run/evidence/${name}_working_tree.patch"
done
"$base/envs/meddinov3/bin/python" -m pip freeze > "$run/evidence/pip_freeze.txt"
"$base/envs/meddinov3/bin/python" - "$run/evidence/environment.json" <<'PY'
import json,platform,sys
try:
 import torch
 torch_info={'version':torch.__version__,'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available()}
except Exception as e: torch_info={'error':str(e)}
with open(sys.argv[1],'w') as f: json.dump({'python':sys.version,'platform':platform.platform(),'torch':torch_info},f,indent=2)
PY
sha256sum "$base/DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" > "$run/evidence/natural_weight_sha256.txt"
sha256sum "$base/codex/mitoverse_round6_20260917_044606/data_manifest_v3.csv" > "$run/evidence/source_manifest_sha256.txt"
printf 'workspace_prepared=%s\n' "$run"
