"""One additional prespecified union: existing finetuned EM + fresh frozen-EM probe."""
from pathlib import Path
import json,numpy as np,nibabel as nib
from audit_cached import BASE,ROOT,DATASETS,metrics,confusion
out={'protocol':'original finetuned EM OR freshly trained frozen-EM probe; exploratory validation fusion, no Gram component','datasets':{}}
for ds in DATASETS:
 split=json.loads((ROOT/'nnUNet_preprocessed'/ds/'splits_final.json').read_text())[0];cases={}
 for case in split['val']:
  gt=nib.load(BASE/'cache'/(case+'_gt.nii'),mmap=True);a=np.load(BASE/'jobs/em_native_recheck'/(case+'.npy'),mmap_mode='r');b=np.load(BASE/'jobs/probe_em_final'/ds/(case+'.npy'),mmap_mode='r');c=np.zeros(4,dtype=np.int64)
  for z in range(a.shape[0]):c+=confusion((a[z]>0)|(b[z]>0),np.asarray(gt.dataobj[:,:,z]).T>0)
  cases[case]=metrics(c)
 out['datasets'][ds]={'cases':cases,'macro_dice':float(np.mean([v['dice'] for v in cases.values()]))};print(ds,out['datasets'][ds]['macro_dice'],flush=True)
(BASE/'em_probe_fusion_metrics.json').write_text(json.dumps(out,indent=2))
