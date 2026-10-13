set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012
cd "$RUN"
env OPENBLAS_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import ast,json
base=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002/src/frozen_probe_control.py').read_text()
new=Path('src/frozen_probe_150k.py').read_text()
a=ast.parse(base);b=ast.parse(new)
for name in ['seed','digest','head_digest','Samples','loss','validation']:
    get=lambda t:ast.dump(next(n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name),include_attributes=False)
    assert get(a)==get(b),name
getloop=lambda t:ast.dump(next(n for n in ast.walk(t) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step'),include_attributes=False)
assert getloop(a)==getloop(b)
print('FROZEN_DECODER_PROTOCOL_AST_UNCHANGED')
PY
env CUDA_VISIBLE_DEVICES=GPU-5a68cc29-a13a-bf90-858e-079e14de807a OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps:/opt/data/3_117/tanghr/DINOv3/dinov3:/opt/data/3_117/tanghr/Med_DINOv3/MedDINOv3/nnUNet B_FULL_CHECKPOINT="$RUN/runs/B_full/resume_smoke10/checkpoints/step000100010/teacher_backbone.pth" B_FULL_STEP=100010 RUN_OUTPUT="$RUN/evaluation/smoke_compatible_probe" /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u src/frozen_probe_150k.py --dataset Dataset111_CMCC --encoder-kind B_full --steps 1 --seed 20260907 --smoke
env OPENBLAS_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('evaluation/smoke_compatible_probe');r=json.loads((p/'result.json').read_text())
assert r['status']=='smoke_pass' and r['frozen_encoder_grad_absent'] and r['frozen_encoder']
assert r['decoder_initial_sha256']=='4b19c61be6451662481678ca87ffac1162812df24c31422c0d8d3f63acc358cf'
assert r['encoder_step']==100010
print('POSTTRAIN_FROZEN_EVALUATION_SMOKE_PASS',r['smoke_loss'])
PY
