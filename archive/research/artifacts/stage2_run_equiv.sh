set -e
ROOT=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2_20260930_082816
STAGE1=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
cd "$ROOT"
PYTHONPATH="$STAGE1/topology_feasibility/vendor" /opt/data/3_117/tanghr/envs/meddinov3/bin/python src/equivariance_b.py
