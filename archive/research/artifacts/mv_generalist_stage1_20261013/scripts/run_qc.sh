set -e
BASE=/opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import tarfile
r=Path('/opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013').resolve()
with tarfile.open(r/'stage_A_bundle.tar.gz') as t:
 for m in t.getmembers():
  assert not m.issym() and not m.islnk() and (r/m.name).resolve().is_relative_to(r)
 t.extractall(r)
PY
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u "$BASE/scripts/pixel_qc.py"
