set -euo pipefail
OUT=/opt/data/3_117/tanghr/codex/primus_official_reprobe_20261013
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import shutil,json,hashlib,urllib.request,subprocess
r=Path('/opt/data/3_117/tanghr/codex/primus_official_reprobe_20261013')
s2=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002')
ext=Path('/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012')
for a,b in [(s2/'references/paths.json',r/'references/paths.json'),(s2/'src/frozen_probe.py',r/'src/frozen_probe.py'),(ext/'src/repository_metrics.py',r/'src/repository_metrics.py')]:
 assert not b.exists();shutil.copy2(a,b)
url='https://raw.githubusercontent.com/ricklisz/MedDINOv3/a2460c2547899283446e520b84523454f7e25608/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/dinov3/models/primus.py'
local=(r/'src/official_primus.py').read_bytes();h=hashlib.sha256(local).hexdigest()
record={'url':url,'commit':'a2460c2547899283446e520b84523454f7e25608','local_sha256':h}
try:
 raw=urllib.request.urlopen(url,timeout=20).read()
 assert hashlib.sha256(raw).hexdigest()==h
 record['live_upstream_byte_match']=True
except AssertionError:raise
except Exception as e:record.update(live_upstream_byte_match='UNVERIFIED',reason=str(e),fallback='previously captured pinned official source')
(r/'audit/official_source_provenance.json').write_text(json.dumps(record,indent=2))
assert b'last_gates' not in local and b'FrequencyGate' not in local
raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert 'GPU-5a68cc29-a13a-bf90-858e-079e14de807a' not in raw,'GPU3 not idle'
print(json.dumps(record),flush=True)
PY
export CUDA_VISIBLE_DEVICES=3
export B_FULL_CHECKPOINT=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120/runs/B_full/formal_attempt1/checkpoints/step000100000/teacher_backbone.pth
export B_FULL_STEP=100000
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONPATH=/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps
/opt/data/3_117/tanghr/envs/meddinov3/bin/python -u "$OUT/scripts/smoke_official.py" > "$OUT/audit/smoke.log" 2>&1
cat "$OUT/audit/smoke.log"
