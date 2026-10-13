set -euo pipefail
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
curl -s -o /dev/null -w 'website_http=%{http_code}\n' --max-time 8 http://127.0.0.1:7860/
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import json
r=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002')
ref=json.loads((r/'references/paths.json').read_text())
print('REFERENCES',json.dumps(ref,indent=2))
p=Path(ref['previous_frozen_probe_source']).parent/'evaluate.py'
s=p.read_text();start=s.index('def sliding(');end=s.index('\ndef ',start+1)
print(s[start:end])
q=Path('/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012/src/repository_metrics.py')
print(q.read_text()[:13000])
out=Path('/opt/data/3_117/tanghr/codex/primus_official_reprobe_20261013')
assert not out.exists(), 'Fresh experiment output required'
for d in ['src','scripts','references','runs','audit','reports']: (out/d).mkdir(parents=True)
print('CREATED',out)
PY
