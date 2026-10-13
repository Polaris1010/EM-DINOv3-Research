set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json,pickle,importlib.util
from pathlib import Path
import nibabel as nib, numpy as np
r=Path('.')
ref=json.loads((r/'references/paths.json').read_text())
print('REFERENCES',json.dumps(ref))
root=Path('/opt/data/3_117/tanghr/Med_DINOv3')
for ds in ['Dataset111_CMCC','Dataset131_CMCT']:
    split=json.loads((root/'nnUNet_preprocessed'/ds/'splits_final.json').read_text())[0]
    for case in split['val']:
        p=root/'nnUNet_raw'/ds/'labelsTr'/(case+'.nii.gz')
        im=nib.load(p)
        print('LABEL',case,im.shape,im.header.get_zooms(),im.header.get_xyzt_units(),str(p))
        for run in [r/'branch_E/runs'/('B_full_'+ds),r/'branch_E/control_20261011/runs'/('official_natural_'+ds)]:
            a=run/ds/(case+'.npy')
            arr=np.load(a,mmap_mode='r')
            print('PRED',str(a),arr.shape,arr.dtype,a.stat().st_size)
print('SURFACE_DISTANCE',importlib.util.find_spec('surface_distance'))
print('GIT')
PY
git status --short
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
curl -s -o /dev/null -w 'website_http=%{http_code}\n' --max-time 8 http://127.0.0.1:7860/
