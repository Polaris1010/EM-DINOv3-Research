set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
"/opt/data/3_117/tanghr/dinov3_shared/bin/python" -c 'import scipy; print("scipy", scipy.__version__)'
"/opt/data/3_117/tanghr/envs/meddinov3/bin/python" - "$run" <<'PY'
import json, pathlib, subprocess, sys
run=pathlib.Path(sys.argv[1]); marker=run/'src/STOP.json'
data=json.loads(marker.read_text())
assert data['reason']=='T_seed0_batch16_smoke10 exited 1', data
listing=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'], text=True)
authorized={'GPU-29591cdf-1498-85e8-bb8c-158082a53728','GPU-794aaaef-6c72-f6ac-b6d0-a07ba3f76364','GPU-5a68cc29-a13a-bf90-858e-079e14de807a'}
active={(gpu.strip(),int(pid.strip())) for gpu,pid in (line.split(',') for line in listing.splitlines()) if gpu.strip() in authorized}
assert all(pid==1971241 for gpu,pid in active), active
marker.unlink()
print('Cleared exactly the T import-failure STOP marker after checking live GPU PIDs')
PY
