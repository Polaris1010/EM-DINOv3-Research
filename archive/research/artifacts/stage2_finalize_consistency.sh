set -euo pipefail
ROOT=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2_20260930_082816
cd "$ROOT"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python src/finalize_fail.py
git add STATUS.json src/finalize_fail.py reports/stage2A_calibration.md topology_calibration/stage2A_gate.json
git commit --amend --no-edit >/dev/null
printf 'commit=%s\n' "$(git rev-parse HEAD)"
printf 'working_tree_status=%s\n' "$(git status --porcelain | wc -l)"
cat STATUS.json
