from pathlib import Path
import json,numpy as np,nibabel as nib
from audit_cached import BASE,ROOT,DATASETS,metrics,confusion
report={'protocol':'exploratory binary OR/AND of newly inferred EM finetuned and MitoNet native masks; no retraining, original fold-0 validation; not a Gram effect','datasets':{}}
for ds in DATASETS:
 split=json.loads((ROOT/'nnUNet_preprocessed'/ds/'splits_final.json').read_text())[0]
 models={m:{'cases':{}} for m in ['union','intersection']};report['datasets'][ds]=models
 for case in split['val']:
  gt=nib.load(BASE/'cache'/(case+'_gt.nii'),mmap=True);a=np.load(BASE/'jobs/em_native_recheck'/(case+'.npy'),mmap_mode='r');b=np.load(BASE/'jobs/mitonet_full_v2'/(case+'.npy'),mmap_mode='r');counts={m:np.zeros(4,dtype=np.int64) for m in models}
  for z in range(a.shape[0]):
   t=np.asarray(gt.dataobj[:,:,z]).T>0;x=a[z]>0;y=b[z]>0
   counts['union']+=confusion(x|y,t);counts['intersection']+=confusion(x&y,t)
  for m in models:models[m]['cases'][case]=metrics(counts[m])
 for m,v in models.items():
  v['macro_dice']=float(np.mean([w['dice'] for w in v['cases'].values()]));v['micro']=metrics(np.sum([[w[k] for k in ('tp','fp','fn','tn')] for w in v['cases'].values()],axis=0));print(ds,m,v['macro_dice'],flush=True)
(BASE/'fusion_metrics.json').write_text(json.dumps(report,indent=2))
