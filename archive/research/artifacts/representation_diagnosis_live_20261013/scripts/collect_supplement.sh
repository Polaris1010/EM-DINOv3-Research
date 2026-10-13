set -euo pipefail
OUT=/opt/data/3_117/tanghr/codex/representation_diagnosis_live_20261013
cat "$OUT/smoke/run.log"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import json,hashlib,shutil,tarfile
from omegaconf import OmegaConf
r=Path('/opt/data/3_117/tanghr/codex/representation_diagnosis_live_20261013')
old=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120')
ext=Path('/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012')
s2=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=[ext/'audit/100k_decoder_initialization_reconstruction.json',ext/'STATUS.json',old/'audit/initialization.json',ext/'audit/training_completion.json',ext/'audit/resume_admission.json']
sources.extend(sorted(old.glob('**/*resolved*.yaml')))
sources.extend(sorted(ext.glob('**/*resolved*.yaml')))
sources.extend(sorted(ext.glob('configs/*.yaml')))
records=[]
for i,p in enumerate(sources):
 if p.is_file():
  q=r/'evidence'/('supplement_'+str(i)+'_'+p.name);shutil.copy2(p,q)
  records.append({'path':str(p),'sha256':sha(p),'copy':str(q)})
 else:records.append({'path':str(p),'status':'UNVERIFIED missing'})
configs=[]
for root in [old,ext,s2]:
 for p in sorted(root.glob('**/*.yaml')):
  try:
   c=OmegaConf.load(p)
   configs.append({'path':str(p),'sha256':sha(p),'gram_use_loss':OmegaConf.select(c,'gram.use_loss'),'gram_ckpt':OmegaConf.select(c,'gram.ckpt'),
   'global_crops_size':OmegaConf.select(c,'crops.global_crops_size'),'local_crops_size':OmegaConf.select(c,'crops.local_crops_size'),
   'gram_teacher_crops_size':OmegaConf.select(c,'crops.gram_teacher_crops_size')})
  except Exception as e:configs.append({'path':str(p),'error':str(e)})
datasets=[];exposure=[]
base=Path('/opt/data/3_117/tanghr/Med_DINOv3')
for ds in ['Dataset111_CMCC','Dataset131_CMCT']:
 splitp=base/'nnUNet_preprocessed'/ds/'splits_final.json';split=json.loads(splitp.read_text())[0]
 descp=base/'nnUNet_raw'/ds/'dataset.json';desc=json.loads(descp.read_text())
 datasets.append({'dataset':ds,'description':desc,'description_sha256':sha(descp),'split':split,'split_sha256':sha(splitp),
 'annotation_completeness':'UNVERIFIED','physical_spacing_calibration':'UNVERIFIED','acquisition_isolation':'UNVERIFIED'})
 for category,cases in split.items():
  for case in cases:
   for model in ['P0','B100k','B150k']:
    exposure.append({'model':model,'dataset':ds,'case_id':case,'split':category,'readout_exposure':'head_optimization' if category=='train' else 'head_selection_and_final_development_evaluation',
     'ssl_image_membership':'source included in full MitoVerse; exact raw-preprocessed mapping UNVERIFIED' if model!='P0' else 'EM continuation none; upstream corpus overlap UNVERIFIED',
     'acquisition_id':'UNVERIFIED','final_independent_test':False})
(r/'evidence/supplement.json').write_text(json.dumps({'sources':records,'config_scan':configs,'datasets':datasets},indent=2))
(r/'evidence/data_exposure_records.json').write_text(json.dumps({'formal_ssl_manifest_sha256':sha(old/'data/pretrain_all_mitoverse.json'),'rows':exposure,
 'legacy_split_membership':'Formal SSL selects all 212 volumes, including legacy train/validation/test/excluded pools; these names do not isolate B_full SSL.'},indent=2))
with tarfile.open(r/'evidence_bundle.tar.gz','w:gz') as tar:
 for folder in ['sources','evidence','smoke']:
  tar.add(r/folder,arcname=folder)
print('BUNDLE READY',flush=True)
PY
