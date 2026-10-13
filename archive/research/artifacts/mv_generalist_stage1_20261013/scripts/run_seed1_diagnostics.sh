set -e
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u - <<'PY'
from pathlib import Path
import sys,json,hashlib,ast
r=Path('/opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013');sys.path.insert(0,str(r/'scripts'))
import analyze
analyze.strata_and_panels()
from mv_data import sha,atomic
node=next(n for n in ast.parse((r/'scripts/analyze.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='strata_and_panels')
atomic(r/'evidence/seed1_diagnostics_receipt.json',{'status':'PASS','manifest_sha256':sha(r/'MV-Dev-v1.json'),'logic_sha256':hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),'predictions_sha256':{str(f.relative_to(r)):sha(f) for f in (r/'runs').glob('*/seed20261013/*_prediction.npy')},'outputs_sha256':{str(f.relative_to(r)):sha(f) for f in [r/'target_strata_seed1.csv',*sorted((r/'failure_panels').glob('*.png'))]},'test_labels_accessed':False})
print('SEED1_DIAGNOSTICS_COMPLETE',flush=True)
PY
