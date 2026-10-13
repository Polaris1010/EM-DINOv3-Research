"""Single guarded continuation after reviewing full-mask counts against the primary method."""
import os,json,time,psutil,hashlib,difflib
from settings import *
from guard import Guard,save
def main():
 prior=json.loads((BASE/'coordinator_pid.json').read_text());pid=prior['pid'];assert not psutil.pid_exists(pid) or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE
 save(BASE/'coordinator_before_mask_adjudication.json',prior);save(BASE/'coordinator_pid.json',dict(pid=os.getpid(),birth=psutil.Process().create_time(),command=['apply_han_correction.py'],started=time.time()))
 save(BASE/'mask_correction_executed_sources.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob('*.py')})
 stage=json.loads((BASE/'stage_status.json').read_text());stage['mask_correction']=dict(status='running',interpretation='Annotated region defines valid supervision; positive labels outside it are ignored, not background');save(BASE/'stage_status.json',stage)
 g=Guard();g.start()
 try:
  g.job('prepare_masked_supervision',None,'prepare_masked_supervision.py');g.job('masked_pipeline_check',2,'check_masked_pipeline.py')
  stage['mask_correction']=dict(status='completed',loss_policy='mask != 0 valid; all zero-mask pixels ignored',formal_training=False)
  stage['S0']['reason']='Mask handling corrected; original specimen/scan/ROI associations still pending for SSL candidates';stage['S1']['formal_mask_semantics']='corrected loader and masked loss checked; old baselines not reusable';save(BASE/'stage_status.json',stage)
  (BASE/'masked_pipeline_diff.patch').write_text(''.join(difflib.unified_diff((R4/'data.py').read_text().splitlines(True),(BASE/'masked_data.py').read_text().splitlines(True),fromfile='round4/data.py',tofile='round6/masked_data.py')))
  save(BASE/'mask_fix_summary.json',dict(status='complete',old_loader_ignored_region_mask=True,corrected_loader='masked_data.py',corrected_schedules='train_patch_schedule_masked_seed{0,1,2}.jsonl',original_labels_and_masks_modified=False,old_results_preserved=True,old_T0_reuse=False,formal_SSL_started=False,remaining_gate='Original specimen/scan/ROI provenance'))
 except Exception as e:g.stop_all(repr(e));stage['mask_correction']=dict(status='blocked',reason=repr(e));save(BASE/'stage_status.json',stage);raise
 finally:g.close()
if __name__=='__main__':main()
