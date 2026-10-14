"""CPU-only checks of the two-source audit, preserving unresolved original data."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def verify():
 folder=ROOT/'experiments/MV-4C';s=read(folder/'MV4C_summary.json');assert len(s['regions'])==2
 c=read(ROOT/'configs/evaluation/MV-4C_protocol.json');assert not c['inference_performed'] and not c['threshold_tuned'] and not c['scale_searched']
 assert [q['region_id'] for q in s['regions']]==['guay21_vol1','muller21_high_c1']
 for q in s['regions']:
  assert q['zarr_to_input_cache_mismatch_voxels']==0 and q['zarr_GT_foreground_to_cache_mismatch_voxels']==0
  assert q['cached_input_sha256']==q['zarr_image_values_sha256']==q['probability_original_input_sha256']
  assert sum(q['image_histogram'])==q['image_stats']['n']
  assert q['image_stats']['n']==__import__('math').prod(q['shape_zyx'])
  assert not q['inference_performed'] and not q['threshold_tuned'] and not q['scale_searched']
 g,m=s['regions'];assert g['original']['normalization_mismatch_voxels']==0 and g['original']['max_abs_mapped_error']==0
 assert g['original']['photometric']=='MINISBLACK' and g['original']['raw_to_zarr_pearson_same_coordinates']>0
 assert m['original']['status']=='UNVERIFIED_MISSING_PRE_INGEST_FILE' and m['original']['exists'] is False
 index=read(folder/'audit_index.json');assert index['status']=='UNVERIFIED' and index['requested_scope_complete'] is False
 assert index['guay_original_mapping_verified'] and not index['muller_original_mapping_verified']
 before=read(ROOT/'evidence/MV-4C/protected_before.json');after=read(ROOT/'evidence/MV-4C/protected_after.json');assert before==after
 runtime=read(ROOT/'evidence/MV-4C/execution_receipt.json')
 assert runtime['source_sha256']==hashlib.sha256((ROOT/'src/mv4c/audit.py').read_bytes()).hexdigest()
 assert runtime['protocol_sha256']==hashlib.sha256((ROOT/'configs/evaluation/MV-4C_protocol.json').read_bytes()).hexdigest()
 assert runtime['gpu_used'] is False and runtime['new_model_inference'] is False
 previous=json.loads(subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'show','8df29e29c63f6732d9f93e417d890e132966f567:experiments/registry.yaml'],cwd=ROOT,text=True))
 assert previous==read(ROOT/'experiments/registry.yaml') and len(previous['experiments'])==40
 for rel in ['docs/MV3_final_report.md','docs/MV4_final_report.md','docs/MV4B_final_report.md','experiments/MV-4/MV4_per_region_metrics.csv','experiments/MV-4B/MV4B_path_comparison.csv']:
  original=subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'show','8df29e29c63f6732d9f93e417d890e132966f567:'+rel],cwd=ROOT);assert original==(ROOT/rel).read_bytes()
 result={'status':'PASS_AVAILABLE_EVIDENCE','requested_scope_complete':False,'guay_raw_to_zarr_verified':True,'muller_raw_to_zarr':'UNVERIFIED_MISSING_ORIGINAL','both_downstream_alignment_verified':True,'reused_probability_input_sha_verified':True,'historical_inputs_unchanged':True,'all40_model_registry_entries_unchanged':True,'historical_reports_and_metrics_unchanged':True,'new_inference':False,'gpu_used':False}
 (folder/'MV4C_results_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');return result
if __name__=='__main__':print(json.dumps(verify(),indent=2))
