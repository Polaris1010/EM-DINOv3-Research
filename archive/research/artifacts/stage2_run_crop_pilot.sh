set -euo pipefail
ROOT=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2_20260930_082816
STAGE1=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
cd "$ROOT"
test -f topology_calibration/equivariance/vit_h_equivariance_424.json
PYTHONPATH="$STAGE1/topology_feasibility/vendor" OPENBLAS_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/meddinov3/bin/python src/crop_score_pilot.py
