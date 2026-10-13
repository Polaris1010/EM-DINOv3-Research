"""Stage A: explicit development overlay; never alter upstream splits or clean governance."""
from pathlib import Path
import json,csv,hashlib,shutil
R=Path(__file__).resolve().parents[1]
OLD=R.parent/'data_governance_stage0_20261013'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def csvout(name,rows):
 with (R/name).open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 live=json.loads((R/'evidence/live_inventory.json').read_text());original=json.loads((OLD/'sources/mv_catalog.json').read_text())
 exposed=json.loads((R.parent/'representation_diagnosis_live_20261013/evidence/formal_pretrain_manifest.json').read_text())
 assert set(original)=={r['id'] for r in live['volumes']}=={v['volume_id'] for v in exposed['volumes']}
 for r in live['volumes']:
  v=r['catalog'];assert v==original[r['id']]
 # Official train/test overlays are preserved; Beta dev comes only from official TRAIN cells.
 roles={'guay21_vol0':'train','guay21_vol1':'dev','guay21_vol2':'test',
 'jiang25_vol1':'train','jiang25_vol2':'dev','jiang25_vol0':'test',
 'muller21_high_c1':'train','muller21_low_c1':'train','muller21_high_c3':'dev','muller21_low_c2':'dev',
 'muller21_high_c2':'test','muller21_high_c4':'test','muller21_low_c3':'test'}
 # Haberl has no source split overlay. Keep it pending instead of inventing a formal holdout.
 eligible_sources={'guay21','jiang25','muller21'}
 entries=[];elig=[];reg=[]
 for row in live['volumes']:
  vid=row['id'];c=row['catalog'];m=row['attrs'].get('mitoverse',{});s=c['dataset_id'];arrays=row['arrays']
  p=roles.get(vid,'excluded');known=s in eligible_sources
  compatible=arrays.get('img',{}).get('shape')==arrays.get('mito',{}).get('shape')==m.get('shape_zyx')==c['shape_zyx']
  curated=m.get('provenance') in {'instance_proofread','binary_conversion_proofread'}
  if s=='han24':reason='UNVERIFIED ignore-mask polarity; excluded'
  elif s=='openorganelle':reason='CellMap official split pending or ambiguous parent ROI/annotation provenance'
  elif s=='wei20':reason='within-volume official train/dev coordinates absent; excluded'
  elif s=='turner22':reason='parent tile registration and Pinky voxel metadata UNVERIFIED'
  elif s=='kunduri22':reason='adjacent stores, parent offset and annotation-version discrepancies unresolved'
  elif s=='conrad23':reason='MitoNet reference benchmark reserved; do not tune on these labels'
  elif s not in eligible_sources:reason='no sufficiently verified development split in this release'
  else:reason='curated original-volume semantic contract; runtime pixel QC pending; acquisition independence not claimed'
  admitted=known and compatible and curated and 'mask' not in arrays
  assert not known or admitted, (vid,'Selected source metadata not eligible')
  roi=[[0,n] for n in c['shape_zyx']]
  spacing=list(reversed(c['voxel_nm']))
  source_key=m.get('source',c.get('source',{}));im=source_key.get('im',source_key.get('img'))
  # A source-file identity is NOT a physical acquisition id. Never invent the latter.
  acquisition=None
  if s=='openorganelle':acquisition=c.get('imaging',{}).get('sample')
  reg.append({'volume_id':vid,'source_id':s,'original_image_file':im or '',
   'original_label_file':source_key.get('mito',''),'acquisition_id':acquisition or '',
   'acquisition_identity_status':'official_sample_name' if acquisition else 'UNVERIFIED',
   'roi_zyx_stored_array':json.dumps(roi),'parent_roi_offset_zyx':'UNVERIFIED',
   'axes':'ZYX','spacing_zyx_nm':json.dumps(spacing),'spacing_evidence':'fixed official ingest.py --voxel X,Y,Z; stored shape Z,Y,X',
   'ssl_exposed_Bfull':True,'supervised_role':p,'modality':c['modality'],
   'claim':'transductive source-volume development; acquisition generalization UNVERIFIED'})
  elig.append({'volume_id':vid,'source_id':s,'supervised_role':p,'admission_status':'pending_pixel_QC' if admitted and p!='test' else ('reserved_test_no_label_read' if admitted else 'excluded'),
   'annotation_status':'curator_declared_proofread' if curated else 'UNVERIFIED','valid_mask_policy':'complete_without_ignore_declared' if admitted else 'UNVERIFIED',
   'instance_semantics':'mito>0; 0=background','mask_present':'mask' in arrays,'shape_compatible':compatible,
   'evidence':'fixed MitoVerse DESIGN/README + live .zattrs + image/mito metadata' if admitted else 'live metadata / official TODO',
   'completeness_limit':'no independent re-annotation; label omission cannot be disproven by numerical QC',
   'reason':reason,'ssl_exposed':True})
  if admitted and p in {'train','dev'}:
   entries.append({'id':vid,'source_id':s,'supervised_role':p,'ssl_exposed':True,'image_path':row['path'],
    'arrays':{'image':'img','instance_label':'mito'},'shape_zyx':c['shape_zyx'],'region_zyx':roi,'axes':'ZYX',
    'coordinate_frame':'stored_array','source_volume_identity':im,'acquisition_id':None,'acquisition_status':'UNVERIFIED',
    'annotation_status':'curator_declared_proofread','valid_mask_policy':'complete_without_ignore_declared',
    'provenance':m['provenance'],'metadata_sha256':row['metadata_sha256'],'spacing_zyx_nm':spacing,
    'modality':c['modality'],'tissue':c['tissue'],'species':c['species'],
    'split_evidence':'guay21 official overlay' if s=='guay21' else ('mitoem2.0 official train/val overlay' if s=='jiang25' else 'project development holdout of two official TRAIN cells; official TEST cells untouched'),
    'sampling_weight_rule':'source-balanced then uniform legal slice within source'})
 assert len(entries)==8 and len({e['source_id'] for e in entries})==3
 manifest={'name':'MV-Dev-v1','schema_version':'1.0','status':'stage_A_pending_runtime_QC',
 'claim':'transductive_source_volume_development','unseen_acquisition_claim_permitted':False,
 'training_permitted':False,'data_revision':'d0af9a6fd11faaf16ce0822233b7bbf72553ce7d',
 'source_code_revision':'df884f6bf353b3cd6b4b6a5ee3620daabd8a79be',
 'ssl_manifest_sha256':sha(R.parent/'representation_diagnosis_live_20261013/evidence/formal_pretrain_manifest.json'),
 'selected_sources':sorted(eligible_sources),'entries':entries,
 'reserved_test_ids':[k for k,v in roles.items() if v=='test'],
 'source_macro_min_dev_volumes':2,'source_min_samples_status':'only muller21 has 2 dev volumes; other sources descriptive only',
 'excluded_ids':[r['id'] for r in live['volumes'] if r['id'] not in roles],
 'limits':['Original volume identity and published role are known; physical acquisition independence is UNVERIFIED.',
 'No final-test labels may be read for training, selection, budget calibration, or failure panels.',
 'Fixed curvature/quantitative QC cannot independently establish exhaustiveness of curated labels.',
 'Official overlays unchanged; clean governance and B_clean remain separate and not unlocked by this development manifest.']}
 dump(R/'MV-Dev-v1.json',manifest);csvout('mitoverse_supervised_eligibility.csv',elig);csvout('mitoverse_acquisition_registry.csv',reg)
 old=json.loads((R.parent/'primus_official_reprobe_20261013/comparison_plan.json').read_text())
 models=[]
 for v in old['models']:
  m={k:v[k] for k in ('id','checkpoint_path','sha256','resolved_config_path','resolved_config_sha256','dinov3_root','nnunet_root','ssl_manifest_sha256')}
  m['primus_source_path']='/opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013/src/official_primus.py'
  m['primus_source_sha256']=old['decoder_source_sha256'];models.append(m)
 dump(R/'model_registry.json',{'models':models,'decoder':'official_meddinov3_primus_multiscale','gates':False})
 shutil.copyfile(R.parent/'primus_official_reprobe_20261013/src/official_primus.py',R/'src/official_primus.py')
 shutil.copyfile(R.parent/'primus_official_reprobe_20261013/src/repository_metrics.py',R/'src/repository_metrics.py')
 (R/'original_primus_config.yaml').write_text('''decoder: official.Primus_Multiscale
upstream_commit: a2460c2547899283446e520b84523454f7e25608
source_sha256: b5398dc48bf5cc14068b4fed38c5f0a8c30bcbbe242d59475016e70912f248d3
embed_dim: 1024
patch_embed_size: 16
num_classes: 2
interaction_indices: [5, 11, 17, 23]
layer_index_convention: zero_based
decoder_trainable_parameters: 44576514
encoder_frozen: true
frequency_gates: false
''')
 protocol={'stage':'MV-1','status':'budget_unlocked_until_pilot','samplers_implemented':['slice_uniform','source_balanced'],
 'sampler':'source_balanced','source_selection':'uniform source then uniform legal Z slice across all training volumes in that source',
 'crop_size':256,'batch_size':1,'input_normalization':'uint8 -> (x/255 - 0.5)/0.25; repeat grayscale to RGB; no physical resampling',
 'spatial_augmentations':['horizontal_flip_p0.5','vertical_flip_p0.5'],
 'foreground_oversample_probability':0.5,'foreground_policy':'on selected TRAIN slice, random foreground pixel centers crop if present; otherwise spatial uniform',
 'optimizer':'AdamW','lr':0.0001,'weight_decay':0.05,'betas':[0.9,0.98],
 'loss':'cross_entropy+soft_dice; mito>0 and valid pixels only','encoder_feature_precision':'bfloat16','decoder_precision':'float32',
 'seed_sequence':[20261013,20261014,20261015],'budget_pilot_model':'P0 only; avoids choosing budget to favor B100/B150',
 'budget_pilot_checkpoints':[2000,5000,10000,20000],'budget_pilot_steps':20000,
 'budget_rule':'plateau: source-macro sparse development Dice improves <=0.005 over two consecutive checkpoints and training median loss improves <=5%; choose first qualifying budget or 20000, flag no plateau; only dev labels allowed',
 'locked_steps':None,'formal_seed_runs_may_start':False,'selection':'fixed locked step; never individual best steps across models',
 'monitoring':'dev deterministic evenly spaced 8 Z slices per volume; whole original XY; no TTA',
 'final_validation':'every dev Z slice; native XY sliding window 256, stride128, mean softmax probabilities, no TTA; argmax; no postprocessing',
 'metrics':{'dice':'volume-level binary overlap','hd95_nm':'official MONAI symmetric max directed 95% using ZYX nm',
 'nsd_primary':'NSD@16nm; engineering tolerance preset, not biologically calibrated','nsd_sensitivity_nm':[8,16,32],
 'legacy_voxel_metrics':['HD95_voxel','NSD@1voxel'],'precision_recall':'all annotated pixels; zero division explicitly reported',
 'std_volume':'sample SD ddof=1; n1 -> undefined, never zero','std_seed':'SD of per-seed source means, separate from volume SD',
 'source_macro_worst':'only sources with >=2 distinct dev source volumes; insufficient sources shown separately'},
 'test_label_access':False,'physical_gpus':[1,2,3],'gpu0_forbidden':True,'ssl_long_training':False,
 'gt_strata':['instance size nm^3 (partial touching ROI objects flagged)','per-object intensity contrast defined as absolute foreground vs 1-pixel outside ring difference','spacing_nm','modality'],
 'other_membrane_adjacency':'UNVERIFIED: mito-only labels cannot identify ER/other membrane types'}
 # JSON is valid YAML 1.2; avoid local dependency changes.
 dump(R/'MV_frozen_probe_protocol.yaml',protocol)
 print({'entries':len(entries),'train':sum(e['supervised_role']=='train' for e in entries),'dev':sum(e['supervised_role']=='dev' for e in entries),'sources':3,'status':manifest['status']})
if __name__=='__main__':main()
