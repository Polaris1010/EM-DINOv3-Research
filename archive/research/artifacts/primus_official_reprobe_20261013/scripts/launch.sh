set -euo pipefail
OUT=/opt/data/3_117/tanghr/codex/primus_official_reprobe_20261013
/opt/data/3_117/tanghr/envs/meddinov3/bin/python -m py_compile "$OUT/src/frozen_probe_official.py" "$OUT/scripts/run_comparison.py"
systemd-run --unit=primus-official-reprobe-20261013.service --description='Official MedDINOv3 Primus P0 B100k B150k frozen comparison' --property=Restart=no --property=KillMode=control-group /usr/bin/env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u "$OUT/scripts/run_comparison.py"
systemctl show primus-official-reprobe-20261013.service -p ActiveState -p MainPID -p Result --no-pager
