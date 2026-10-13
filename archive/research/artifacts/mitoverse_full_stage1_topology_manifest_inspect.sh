set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json
p='/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120/topology_feasibility/sample_manifest.json'
d=json.load(open(p))
print({k:(len(v) if isinstance(v,list) else v) for k,v in d.items() if k!='samples'})
print('first sample',d['samples'][0])
PY
