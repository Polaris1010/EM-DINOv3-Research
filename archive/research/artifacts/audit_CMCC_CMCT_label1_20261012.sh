set -euo pipefail
/opt/data/3_117/tanghr/envs/meddinov3/bin/python -u - <<'PY'
from pathlib import Path
import json,numpy as np,SimpleITK as sitk
root=Path('/opt/data/3_117/tanghr/Med_DINOv3')
for ds in ['Dataset111_CMCC','Dataset131_CMCT']:
 raw=root/'nnUNet_raw'/ds;pp=root/'nnUNet_preprocessed'/ds
 print('DATASET',ds,'RAW_RESOLVED',str(raw.resolve()),flush=True)
 for p in [raw/'dataset.json',pp/'dataset.json',pp/'dataset_fingerprint.json']:
  if p.exists():print('METADATA',str(p),p.read_text()[:3500],flush=True)
 split=json.loads((pp/'splits_final.json').read_text())[0]
 for case in split['val']:
  p=raw/'labelsTr'/(case+'.nii.gz');img=sitk.ReadImage(str(p));arr=sitk.GetArrayFromImage(img);v,c=np.unique(arr,return_counts=True)
  print('GT_VALIDATION',case,'RESOLVED',str(p.resolve()),'COUNTS',dict(zip(map(str,v),map(int,c))),'SHAPE',arr.shape,'NONZERO_EQUALS_LABEL1',bool(np.array_equal(arr>0,arr==1)),flush=True)
 for case in split['train']+split['val']:
  p=pp/'nnUNetPlans_2d'/(case+'_seg.npy')
  if p.exists():
   a=np.load(p,mmap_mode='r');v,c=np.unique(a,return_counts=True)
   print('PREPROCESSED',case,'COUNTS',dict(zip(map(str,v),map(int,c))),flush=True)
PY
rg -n 'CMCC|CMCT|== ?1|> ?0' /opt/data/3_117/tanghr/Med_DINOv3/MedDINOv3/nnUNet/nnunetv2/dataset_conversion -g '*.py' | head -100
