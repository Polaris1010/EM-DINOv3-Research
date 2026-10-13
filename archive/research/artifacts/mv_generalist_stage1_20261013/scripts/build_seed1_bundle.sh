set -e
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import tarfile
r=Path('/opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013')
files=[r/'target_strata_seed1.csv',r/'evidence/seed1_diagnostics_receipt.json',r/'evidence/metric_memoization_verification.json',*sorted((r/'failure_panels').glob('*.png'))]
with tarfile.open(r/'seed1_diagnostics.tar.gz','w:gz') as t:
 for f in files:t.add(f,arcname=str(f.relative_to(r)))
print('DIAGNOSTICS_BUNDLE',len(files),'bytes',(r/'seed1_diagnostics.tar.gz').stat().st_size)
PY
