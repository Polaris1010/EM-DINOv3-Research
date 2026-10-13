set -euo pipefail
cd /opt/data/3_117/tanghr/Med_DINOv3
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import hashlib,inspect,json,monai,torch
from monai.metrics import compute_hausdorff_distance,compute_surface_dice
p=Path('MedDINOv3/nnUNet/nnunetv2/compute_metrics.py')
if not p.exists(): p=Path('nnUNet/nnunetv2/compute_metrics.py')
print('METRIC_SOURCE',str(p),p.exists())
if p.exists():
    print('SHA',hashlib.sha256(p.read_bytes()).hexdigest())
    print(p.read_text())
print('MONAI',monai.__version__,'TORCH',torch.__version__)
print('HD_ARGS',inspect.signature(compute_hausdorff_distance))
print('NSD_ARGS',inspect.signature(compute_surface_dice))
PY
