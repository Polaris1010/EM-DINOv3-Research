set -euo pipefail
base=/opt/data/3_117/tanghr
run="$base/codex/mito_pretrain_v2_20260926_060227"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$base/envs/meddinov3/bin/python" "$run/src/probe_last_selection.py" | tee "$run/evidence/last_selection_probe_v2.json"
sha256sum "$run/src/last_selection.py" "$run/src/probe_last_selection.py"
