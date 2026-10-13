set -euo pipefail
cd /opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json,hashlib
from pathlib import Path
p=Path('branch_E/boundary_metrics_20261012')
r=json.loads((p/'results.json').read_text())
assert r['status']=='complete' and len(r['summary'])==4 and len(r['per_volume'])==10
assert r['protocol']['evaluator_ast_matches_upstream'] and r['protocol']['std_ddof']==1 and r['protocol']['nsd_threshold']==1.0
audit=json.loads((p/'input_audit.json').read_text());assert len(audit)==10
for a in audit:
    assert a['dice_matches_previous'] and a['spacing_xyz']==[1.,1.,1.]
    h=hashlib.sha256()
    with Path(a['prediction_source']).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    assert h.hexdigest()==a['prediction_sha256']
s=Path('STATUS.json');d=json.loads(s.read_text())
d['E_surface_metrics']='COMPLETE_MEDDINOV3_DEFAULT_MONAI_NSD_TAU_1_VOLUME_STD_DDOF_1'
d['E_surface_metrics_report']='branch_E/boundary_metrics_20261012/REPORT_zh.md'
assert d['long_training']=='PROHIBITED' and d['P'].startswith('PAUSED')
s.write_text(json.dumps(d,indent=2))
print('FINAL_AUDIT_PASS: 10 predictions unchanged; 4 groups complete; protocol matches upstream')
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
curl -s -o /dev/null -w 'website_http=%{http_code}\n' --max-time 8 http://127.0.0.1:7860/
