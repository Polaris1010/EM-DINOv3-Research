set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012
/opt/data/3_117/tanghr/envs/meddinov3/bin/python -m py_compile "$RUN/scripts/finalize_evaluation_metrics.py"
systemd-run --unit=mitoverse-B-full-finalize150k-20261012.service --description='Finalize matched B_full 150k frozen surface metrics with original 100k initialization audit' --property=Restart=no --property=KillMode=control-group --property=TimeoutStopSec=300 /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u "$RUN/scripts/finalize_evaluation_metrics.py"
systemctl show mitoverse-B-full-finalize150k-20261012.service -p ActiveState -p MainPID -p Result --no-pager
