set -euo pipefail
TASK=/opt/data/3_117/tanghr/codex/mv_dev_v2_notta_20261013
STATE=$(systemctl is-active mv-dev-v2-notta-20261013.service || true)
test "$STATE" = failed
mkdir -p "$TASK/evidence/quantile_failure_original"
cp -n "$TASK/src/validation.py" "$TASK/scripts/run_validation.py" "$TASK/STATUS.json" "$TASK/logs/score.log" "$TASK/scoring_progress.json" "$TASK/evidence/quantile_failure_original/"
