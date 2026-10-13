set -euo pipefail
ROOT=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2_20260930_082816
cd "$ROOT"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python src/finalize_fail.py
