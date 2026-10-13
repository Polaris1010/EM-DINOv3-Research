set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
top="$run/topology_feasibility"
test ! -e "$top/metrics.csv"
systemd-run --unit=mitoverse-stage1-raw-ph.service \
  --description='Stage 1 raw EM PH diagnostic on 424 label-free slices' \
  --property=Restart=no --property=Nice=15 --property=CPUQuota=100% \
  /bin/bash -lc "export PYTHONPATH='$top/vendor'; /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u '$run/scripts/topology_raw_ph.py' > '$top/raw_ph.log' 2>&1"
sleep 5
systemctl show mitoverse-stage1-raw-ph.service -p ActiveState -p MainPID -p Result --no-pager
test -f "$top/metrics.csv"
head -n 2 "$top/metrics.csv"
