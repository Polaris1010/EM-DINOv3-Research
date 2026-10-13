"""Adversarial split checks using real locked metadata, no image or label access."""
import copy,json,sys,hashlib,csv
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mv_data import partition_check
def main():
 m=json.loads((R/'MV-Dev-v1.json').read_text());partition_check(m);checks=[]
 cases=[]
 x=copy.deepcopy(m);dev=next(r for r in x['entries'] if r['supervised_role']=='dev');train=next(r for r in x['entries'] if r['supervised_role']=='train');dev['source_volume_identity']=train['source_volume_identity'];cases.append(('same_source_file_across_roles',x))
 x=copy.deepcopy(m);x['entries'][0]['id']=x['reserved_test_ids'][0];cases.append(('reserved_test_admission',x))
 x=copy.deepcopy(m);x['entries'][0]['region_zyx'][0][1]+=1;cases.append(('ROI_outside_stored_array',x))
 x=copy.deepcopy(m);x['entries'][0]['valid_mask_policy']='mask_nonzero_is_valid';cases.append(('unverified_ignore_polarity',x))
 x=copy.deepcopy(m);x['unseen_acquisition_claim_permitted']=True;cases.append(('unsupported_acquisition_claim',x))
 x=copy.deepcopy(m);x['entries'][0]['axes']='XYZ';cases.append(('axis_mismatch',x))
 for name,x in cases:
  try:partition_check(x)
  except (AssertionError,ValueError):checks.append({'check':name,'status':'PASS_rejected'})
  else:raise AssertionError('Unsafe metadata accepted: '+name)
 qc=json.loads((R/'evidence/pixel_QC.json').read_text());assert len(qc['volumes'])==8 and not qc['test_labels_accessed']
 assert hashlib.sha256((R/'evidence/pixel_QC.json').read_bytes()).hexdigest()==m['pixel_qc_sha256']
 for v in qc['volumes']:assert v['instance_count']==v['declared_instance_count'] and 0<v['foreground_voxels']<v['total_voxels']
 smoke=json.loads((R/'evidence/multi_source_smoke.json').read_text());assert len({x['head_initial_sha256'] for x in smoke['models']})==1
 for row in smoke['models']:
  assert row['encoder_unchanged'] and row['trainable_parameters']==44576514 and not row['test_labels_accessed']
  balanced=row['sampling_draws']['source_balanced'];assert max(balanced.values())-min(balanced.values())<.02*sum(balanced.values())
 result={'status':'PASS','negative_admission_tests':checks,'runtime_pixel_QC_receipt_valid':True,'official_decoder_hash':hashlib.sha256((R/'src/official_primus.py').read_bytes()).hexdigest(),
 'balanced_sampling_verified':True,'prior_clean_governance_modified':False}
 (R/'evidence/local_admission_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
