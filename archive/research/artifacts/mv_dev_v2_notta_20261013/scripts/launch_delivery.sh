set -euo pipefail
TASK=/opt/data/3_117/tanghr/codex/mv_dev_v2_notta_20261013
systemd-run --unit=mv-dev-v2-notta-delivery-20261013 --property=WorkingDirectory="$TASK" --property=StandardOutput=append:"$TASK/logs/delivery.log" --property=StandardError=append:"$TASK/logs/delivery.log" /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u "$TASK/scripts/deliver.py"
systemctl is-active mv-dev-v2-notta-delivery-20261013.service
