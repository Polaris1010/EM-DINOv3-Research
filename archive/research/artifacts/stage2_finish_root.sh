set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2_20260930_082816
test -d "$run"
git -C /opt/data/3_117/tanghr/DINOv3/dinov3 rev-parse HEAD > "$run/audit/source_state.txt"
git -C /opt/data/3_117/tanghr/DINOv3/dinov3 status --short >> "$run/audit/source_state.txt"
curl -fsS -o /dev/null http://127.0.0.1:7860/
git -C "$run" init -q -b codex/stage2
cat > "$run/.gitignore" <<'EOF'
*.pth
*.pt
*.npz
*.npy
*.log
*.png
*.tmp
topology_cache/mam/
EOF
RUN="$run" /opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json,os
from pathlib import Path
run=Path(os.environ['RUN'])
s=json.loads((run/'topology_calibration/sample_manifest.json').read_text())
m=json.loads(Path(json.loads((run/'stage1_reference/stage1_paths.json').read_text())['manifest']).read_text())
print('stage2_root',run)
print('sample_count',len(s['samples']),'full_volumes',len(m['volumes']))
print('full_first_keys',list(m['volumes'][0]))
print('sample_first',s['samples'][0])
PY
