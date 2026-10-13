set -euo pipefail
ROOT=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2_20260930_082816
STAGE1=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
cd "$ROOT"
test "$(find topology_calibration/sam_vit_h/paired_maps -maxdepth 1 -name 'sample_*.npz' | wc -l)" = 424
test -f topology_calibration/sam_vit_h/generation_complete.json
PYTHONPATH="$STAGE1/topology_feasibility/vendor" /opt/data/3_117/tanghr/envs/meddinov3/bin/python src/sam_h_analyze.py
