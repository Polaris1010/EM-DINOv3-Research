"""Independent arithmetic and protocol reconciliation of completed Pilot B artifacts."""
import csv,json,math,statistics,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'experiments/Pilot-B';MODEL='Trajectory-Lerp05'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def table(n):
 with (R/n).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def close(a,b):
 if a in ('',None) or b in ('',None):assert a in ('',None) and b in ('',None);return
 assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-10),(a,b)
def avg(a):
 a=[float(x) for x in a if x not in ('',None)];return statistics.mean(a) if a else None
def check_counts(row):
 tp,fp,fn,tn=[int(row[k]) for k in ['tp','fp','fn','tn']]
 close(row['dice'],2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1)
 close(row['precision'],tp/(tp+fp) if tp+fp else None);close(row['recall'],tp/(tp+fn) if tp+fn else None)
 return tp,fp,fn,tn
def main():
 status=read(R/'STATUS.json');assert status['status']=='complete' and status['Primus_steps']==40000 and status['Linear_steps']==500 and status['dev_regions']==38 and status['MB1_groups']==7
 assert not status['GPU0_used'] and not status['new_SSL_steps'] and not status['Pilot_C_started'] and not status['further_experiments_started']
 enc=read(R/'encoder.json');assert enc['alpha']==.5 and enc['keys']==368 and enc['training_step'] is None
 fit=read(R/'linear/fit_complete.json');assert fit['steps']==500 and fit['seed']==20261027 and not fit['evaluation_GT_read'] and fit['Primus_gate'] is False
 assert fit['coordinate_batch_hash']=='52112acd1aca39ebbf17bd943efa09cfe9bef677dda93f6f3328d79c02b2ba54'
 done=read(R/'downstream/runs/Trajectory-Lerp05/seed20261014/training_complete.json')
 assert done['completed_steps']==40000 and done['seed']==20261014 and done['encoder_frozen'] and done['encoder_unchanged'] and not done['tta']
 assert done['encoder_sha256']==enc['sha256'] and done['interaction_indices']==[5,11,17,23]
 assert done['decoder_initial_sha256']=='2ec39fb096d1a0eceb3124a9e08549d0c6ac82665dd71f8e2bc7b5a8d56f48ec'
 assert done['sampling_hash']=='45b8763f399df51c4455e4a791e6c32b2e9d6dfd9e5ba9ec988bbebb8b883d64' and sum(done['source_draws'].values())==40000
 dev=table('PilotB_dev_per_region.csv');sources=table('PilotB_dev_per_source.csv');assert len(dev)==114 and len(sources)==36
 for row in dev:check_counts(row)
 for region in {x['region_id'] for x in dev}:
  rows=[x for x in dev if x['region_id']==region];assert len(rows)==3 and len({x['model'] for x in rows})==3
  assert len({(x['valid_voxels'],x['gt_foreground_voxels'],x['source_id']) for x in rows})==1
 for row in sources:
  group=[x for x in dev if x['model']==row['model'] and x['source_id']==row['source_id']]
  assert int(row['regions'])==len(group)
  for k in ['dice','precision','recall','hd95_nm','nsd_16nm','predicted_foreground_fraction']:close(row[k],avg([x[k] for x in group]))
 pairs=table('PilotB_dev_paired_vs_100k.csv');assert len(pairs)==38
 for row in pairs:close(row['delta'],float(row['midpoint_dice'])-float(row['D2_100k_dice']))
 images=table('PilotB_new_MB1_per_image.csv');datasets=table('PilotB_MB1_per_dataset.csv');muscle=table('PilotB_muscle_Primus_per_image.csv')
 assert len(images)==2595 and len(datasets)==21 and len(muscle)==906
 for row in images:check_counts(row)
 for row in [x for x in datasets if x['model']==MODEL]:
  group=[x for x in images if x['dataset_id']==row['dataset_id']];assert len(group)==int(row['planes'])
  for k in ['tp','fp','fn','tn']:assert sum(int(x[k]) for x in group)==int(row[k])
  for k in ['dice','iou','precision','recall','hd95_nm','nsd_16nm','predicted_foreground_fraction','pixel_AUROC','pixel_AP']:close(row[k],avg([x[k] for x in group]))
 for z in {x['z'] for x in muscle}:
  group=[x for x in muscle if x['z']==z];assert len(group)==3
  assert len({int(x['tp'])+int(x['fn']) for x in group})==1 and len({x['far_background_pixels'] for x in group})==1
 for model in ['D2-30k','D2-100k',MODEL]:
  group=[x for x in muscle if x['model']==model];row=next(x for x in datasets if x['model']==model and x['dataset']=='glycolytic_muscle')
  close(row['pixel_AP'],avg([x['pixel_AP'] for x in group]));close(row['far_FP_rate'],sum(int(x['far_FP_pixels']) for x in group)/sum(int(x['far_background_pixels']) for x in group))
  if model!=MODEL:assert read(R/'benchmark'/model/'metrics.json')['old_mask_disagreement_pixels']==0
 summary=read(R/'PilotB_summary.json');assert summary['status']=='complete' and summary['Linear_performance_never_gated_Primus'] and not summary['new_SSL_steps']
 for model,v in summary['models'].items():
  close(v['MV3_source_equal_Dice'],avg([x['dice'] for x in sources if x['model']==model]));close(v['MB1_dataset_equal_Dice'],avg([x['dice'] for x in datasets if x['model']==model]))
 spectra=table('linear/spectrum.csv');distribution=table('linear/spectrum_distribution.csv');assert len(spectra)==len(distribution)==15
 for row in distribution:
  a=json.loads(row['variance_fractions_descending']);close(sum(a),1);assert len(a)==512 and min(a)>=0
  s=next(x for x in spectra if x['source']==row['source'] and x['repeat']==row['repeat']);close(s['effective_rank'],math.exp(-sum(p*math.log(p) for p in a if p>0)))
 old=json.loads(subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'show','6aadf409e734601ff8ddd85fc1eaf30875997125:experiments/registry.yaml'],cwd=ROOT));new=read(ROOT/'experiments/registry.yaml');idx={x['experiment_id']:x for x in new['experiments']}
 for x in old['experiments']:assert idx[x['experiment_id']]==x,x['experiment_id']
 assert set(idx)-{x['experiment_id'] for x in old['experiments']}=={'Pilot-B-Trajectory-Lerp05'}
 result={'status':'PASS','dev_region_rows':114,'dev_source_rows':36,'paired_dev_rows':38,'new_MB1_images':2595,'muscle_AP_images':906,'spectrum_repeats':15,'historical_registry_entries_unchanged':61,'head_initialization_and_sampling_match':True,'parent_muscle_masks_equal_history':True,'Linear_AP_gate':False,'SSL_training':False}
 (ROOT/'evidence/Pilot-B/independent_validation.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
if __name__=='__main__':main()
