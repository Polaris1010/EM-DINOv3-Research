set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import hashlib,json,shutil,platform
from pathlib import Path
import torch
root=Path.cwd();out=root/'branch_E/control_20261011/source_snapshot';out.mkdir(exist_ok=True)
match=json.loads((root/'branch_E/control_20261011/protocol_match.json').read_text())
for item in match['dependency_snapshots']:
    source=Path(item['path']);target=out/source.name
    assert hashlib.sha256(source.read_bytes()).hexdigest()==item['sha256']
    shutil.copy2(source,target)
for name in ['frozen_probe.py','frozen_probe_control.py','validate_control.py']:
    shutil.copy2(root/'src'/name,out/name)
info={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
      'decoder_head_initial_sha256':match['decoder_initial_sha256'],
      'protocol':'same existing frozen decoder probe; only encoder checkpoint changes',
      'P':'paused','SSL_long_training':'prohibited'}
(root/'branch_E/control_20261011/environment.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info))
PY
