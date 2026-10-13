set -euo pipefail
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import hashlib,json
r=Path('/opt/data/3_117/tanghr/codex/representation_diagnosis_live_20261013')
records=[]
for name in ['decoder_architecture_audit.md','dinov3_stage_completion_audit.md','representation_diagnosis_preflight.md','model_parameter_summary.json','verification_results.json','historical_label1_metrics.csv']:
 p=r/'reports'/name;q=r/name
 assert p.is_file() and not q.exists()
 p.rename(q)
 records.append({'path':str(q),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
(r/'reports/delivery.json').write_text(json.dumps(records,indent=2))
print('Reports available at',str(r))
PY
