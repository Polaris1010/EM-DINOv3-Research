set -euo pipefail
cd /opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227/src
CUDA_VISIBLE_DEVICES='' /opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import expanded_eval
from pathlib import Path
print('LEGACY_ENV_IMPORT_OK')
s=Path('guard.py').read_text()
print('\n'.join(line for line in s.splitlines() if 'cmd=list' in line))
p=Path('/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/official/postprocess.py')
s=p.read_text();i=s.index('def factor_pad');print(s[i:i+1200])
PY
