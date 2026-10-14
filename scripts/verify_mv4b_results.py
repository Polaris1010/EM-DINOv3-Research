"""CPU-only verification of completed four-region path comparison artifacts."""
import csv,hashlib,json,math,statistics,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def verify():
 c=read(ROOT/'configs/evaluation/MV-4B_protocol.json');folder=ROOT/'experiments/MV-4B'
 assert c['confidence_thr']==0.5 and c['fixed']['scale']==1 and c['fixed']['tta'] is False
 expected=set(c['pilot_regions']);pilot=read(folder/'MV4B_pilot_summary.json')['regions'];assert {r['region_id'] for r in pilot}==expected and len(pilot)==4
 for r in pilot:
  assert r['protocol_sha256']==hashlib.sha256((ROOT/'configs/evaluation/MV-4B_protocol.json').read_bytes()).hexdigest()
  assert r['checkpoint_sha256']==c['official_weight_sha256']
  assert r['semantic_foreground_lost_in_panoptic']==0 and r['semantic_true_positive_lost_in_panoptic']==0
  for mode in ['semantic','raw_threshold','panoptic']:
   q=r[mode];tp,fp,fn,tn=[q[k] for k in ['tp','fp','fn','tn']]
   assert tp+fp+fn+tn==q['valid_voxels'] and tp+fn==q['gt_foreground_voxels']
   assert math.isclose(q['dice'],2*tp/(2*tp+fp+fn),abs_tol=1e-12)
   assert math.isclose(q['precision'],tp/(tp+fp),abs_tol=1e-12) and math.isclose(q['recall'],tp/(tp+fn),abs_tol=1e-12)
   assert math.isclose(q['predicted_foreground_fraction'],(tp+fp)/q['valid_voxels'],abs_tol=1e-12)
  assert r['raw_threshold']==r['panoptic']
  assert abs(r['semantic_minus_panoptic_dice'])<1e-10
  j=read(folder/'runs/Official-MitoNet-Semantic-v1'/(r['region_id']+'_inference.json'))
  assert j['all_native_z_inferred'] and j['gt_used_for_inference'] is False and j['physical_gpu'] in [1,2,3]
  assert j['official_semantic_output_equals_probability_threshold_after_stuff_area']
  assert j['threshold']==0.5 and j['probability_dtype']=='float32'
 rows=list(csv.DictReader((folder/'MV4B_per_region_metrics.csv').open(encoding='utf-8')));assert len(rows)==32
 old=list(csv.DictReader((ROOT/'experiments/MV-4/MV4_per_region_metrics.csv').open(encoding='utf-8')))
 oldidx={(r['model'],r.get('seed',''),r['region_id']):r for r in old}
 for r in rows:
  assert r['region_id'] in expected
  if r['model']=='MitoNet-Semantic-v1':continue
  original=oldidx[r.get('historical_model') or r['model'],r.get('seed',''),r['region_id']]
  for k in ['dice','precision','recall','hd95_nm','nsd_16nm','predicted_foreground_fraction']:
   assert (not r[k] and not original[k]) or math.isclose(float(r[k]),float(original[k]),abs_tol=1e-12)
 status=read(folder/'STATUS.json');assert status['status']=='pilot_complete' and status['historical_mv3_mv4_unchanged']
 decision=read(ROOT/'evidence/MV-4B/extension_decision.json');assert decision['extend_to_38'] is False
 before=read(ROOT/'evidence/MV-4B/pilot_inputs_before.json');after=read(ROOT/'evidence/MV-4B/pilot_inputs_after.json');assert before==after
 runtime=read(ROOT/'evidence/MV-4B/run_outputs_verification.json');assert runtime['status']=='PASS' and runtime['raw_threshold_masks_equal_old_panoptic_voxelwise']
 for asset in runtime['assets']:assert asset['raw_threshold_vs_old_panoptic_xor_voxels']==0
 previous=json.loads(subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'show','ff19c034ebcc9f879ff1347e3d75f3325d210de8:experiments/registry.yaml'],cwd=ROOT,text=True))['experiments']
 current=read(ROOT/'experiments/registry.yaml')['experiments'];assert len(previous)==39 and current[:39]==previous and len(current)==40
 oldreport=subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'show','ff19c034ebcc9f879ff1347e3d75f3325d210de8:docs/MV4_final_report.md'],cwd=ROOT)
 assert oldreport==(ROOT/'docs/MV4_final_report.md').read_bytes()
 result={'status':'PASS','four_entire_native_regions':True,'semantic_regions':4,'comparison_rows':32,'no_full38_extension':True,'panoptic_raw_probability_threshold_xor':0,'no_semantic_true_positive_loss':True,'old_39_registry_entries_unchanged':True,'historical_mv4_report_unchanged':True,'historical_dinov3_and_panoptic_metrics_unchanged':True,'gt_parameter_tuning':False}
 (folder/'MV4B_results_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');return result
if __name__=='__main__':print(json.dumps(verify(),indent=2))
