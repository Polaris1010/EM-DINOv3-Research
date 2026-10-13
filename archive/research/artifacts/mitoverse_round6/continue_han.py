"""One coordinator for the bounded mask correction; no formal SSL launch is permitted."""
import os,json,time,hashlib,psutil,difflib
from settings import *
from guard import Guard,save
def main():
 prior=json.loads((BASE/'coordinator_pid.json').read_text());pid=prior['pid']
 assert not psutil.pid_exists(pid) or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE,'Previous coordinator still active'
 assert json.loads((BASE/'jobs/metric_audit_v2/result.json').read_text())['rows']==288
 save(BASE/'coordinator_before_han_fix.json',prior);save(BASE/'coordinator_pid.json',dict(pid=os.getpid(),birth=psutil.Process().create_time(),command=['continue_han.py'],started=time.time()))
 save(BASE/'mask_fix_source_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob('*.py')})
 stage=json.loads((BASE/'stage_status.json').read_text());stage['mask_correction']=dict(status='running',primary_source='Han et al. 2024 DOI 10.1038/s41467-024-50411-z');save(BASE/'stage_status.json',stage)
 g=Guard();g.start()
 try:
  audit=g.job('han_mask_full_audit_v2',None,'han_mask_audit.py')
  if json.loads((audit/'result.json').read_text()).get('mask_policy_gate')=='needs_outside_region_review':
   stage['mask_correction']=dict(status='blocked',reason='Complete mask audit found positive labels outside region; inspect before enabling corrected loader');save(BASE/'stage_status.json',stage);return
  g.job('masked_pipeline_check',2,'check_masked_pipeline.py')
  stage['mask_correction']=dict(status='completed',loss_policy='mask != 0 valid; zero ignored',checks='full five-volume foreground containment; loss/gradient invariance; real-data two-step smoke',formal_training=False)
  stage['S0']['reason']='Mask correction verified; original scan/ROI linkage for SSL candidates remains unresolved'
  stage['S1']['formal_mask_semantics']='corrected and checked; any formal comparison requires freshly trained controls'
  save(BASE/'stage_status.json',stage)
  (BASE/'masked_pipeline_diff.patch').write_text(''.join(difflib.unified_diff((R4/'data.py').read_text().splitlines(True),(BASE/'masked_data.py').read_text().splitlines(True),fromfile='round4/data.py',tofile='round6/masked_data.py')))
  save(BASE/'mask_fix_summary.json',dict(status='complete',old_loader_ignored_region_mask=True,corrected_loader='masked_data.py',corrected_schedules='train_patch_schedule_masked_seed{0,1,2}.jsonl',old_results_read_only=True,old_T0_reuse=False,formal_SSL_started=False,remaining_gate='Original specimen/scan and ROI linkage for candidate SSL pool'))
 except Exception as e:
  g.stop_all(repr(e));stage['mask_correction']=dict(status='blocked',reason=repr(e));save(BASE/'stage_status.json',stage);raise
 finally:g.close()
if __name__=='__main__':main()
