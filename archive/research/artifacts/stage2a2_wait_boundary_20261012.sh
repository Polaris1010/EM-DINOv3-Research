set -euo pipefail
cd /opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
env OPENBLAS_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u - <<'PY'
from pathlib import Path
import time,json,os
p=Path('branch_E/boundary_metrics_20261012');last=''
for _ in range(180):
    if (p/'error.txt').exists():
        print((p/'error.txt').read_text(),flush=True);raise SystemExit(1)
    log=(p/'run.log').read_text()
    if log!=last:
        print(log[len(last):],end='',flush=True);last=log
    if (p/'results.json').exists():
        print((p/'REPORT_zh.md').read_text(),flush=True);break
    pid=int((p/'job.pid').read_text())
    if not Path('/proc') .joinpath(str(pid)).exists():
        raise RuntimeError('Boundary metric job exited without results')
    time.sleep(5)
else: raise RuntimeError('Metric job still running after 15 minutes')
PY
