set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012
cd "$RUN"
test -s audit/continuation_admission.json
test -s evaluation/smoke_compatible_probe/result.json
test -s scripts/posttrain_evaluation.py
test ! -e runs/B_full/formal_attempt1
systemd-run --unit=mitoverse-B-full-extend150k-20261012.service --description='B_full complete-state 100k to 150k continuation and matched frozen evaluation' --property=Restart=no --property=KillMode=control-group --property=TimeoutStopSec=300 /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u "$RUN/scripts/gpu_guard.py"
systemctl show mitoverse-B-full-extend150k-20261012.service -p ActiveState -p MainPID -p Result --no-pager
