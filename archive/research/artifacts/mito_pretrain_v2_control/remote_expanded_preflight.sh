set -euo pipefail
src=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227/src
cd "$src"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python audit_expanded.py
CUDA_VISIBLE_DEVICES='' /opt/data/3_117/tanghr/dinov3_shared/bin/python - <<'PY'
import expanded_eval
print('EXPANDED_CPU_IMPORT_OK')
from pathlib import Path
p=Path('/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/official/postprocess.py')
s=p.read_text();i=s.index('def factor_pad');print(s[i:i+1300])
PY
