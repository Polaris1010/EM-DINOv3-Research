set -euo pipefail
TASK=/opt/data/3_117/tanghr/codex/mv_dev_v2_expansion_20261013
test -f "$TASK/src/validation.py"
test ! -f "$TASK/STATUS.json"
systemd-run --unit=mv-dev-v2-validation-20261013 --property=WorkingDirectory="$TASK" --property=StandardOutput=append:"$TASK/logs/runner.log" --property=StandardError=append:"$TASK/logs/runner.log" /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u "$TASK/scripts/run_validation.py"
systemctl is-active mv-dev-v2-validation-20261013.service
