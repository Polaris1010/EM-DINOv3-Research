set -e
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import tarfile,json
r=Path('/opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013').resolve()
assert json.loads((r/'MV-Dev-v1.json').read_text())['status']=='locked_development'
with tarfile.open(r/'probe_bundle.tar.gz') as t:
 for m in t.getmembers():
  assert not m.issym() and not m.islnk() and (r/m.name).resolve().is_relative_to(r) and m.name.startswith(('src/','scripts/'))
 t.extractall(r)
PY
BASE=/opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013
/opt/data/3_117/tanghr/envs/meddinov3/bin/python -m py_compile "$BASE/src/mv_probe.py" "$BASE/src/mv_data.py" "$BASE/scripts/run_benchmark.py" "$BASE/scripts/analyze.py"
systemd-run --unit=mv-generalist-stage1-20261013.service --description='MitoVerse curated development frozen official Primus benchmark' --property=Restart=no --property=KillMode=control-group /usr/bin/env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u "$BASE/scripts/run_benchmark.py"
systemctl show mv-generalist-stage1-20261013.service -p ActiveState -p MainPID -p Result --no-pager
