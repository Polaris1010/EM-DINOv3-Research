"""One bounded Round6 coordinator. Formal SSL cannot bypass S0 or S1 gates."""
import os,sys,json,time,traceback,hashlib,difflib
from settings import *
from guard import Guard,save
def update(stage,status,**detail):
 p=BASE/'stage_status.json';d=json.loads(p.read_text()) if p.exists() else {f'S{i}':dict(status='pending') for i in range(5)}
 d[stage]=dict(status=status,time=time.time(),**detail);save(p,d)
def report():
 def load(p):return json.loads(p.read_text()) if p.exists() else None
 frozen=load(BASE/'jobs/diagnostic_frozen/result.json');x1=load(BASE/'jobs/diagnostic_x1/result.json');metric=load(BASE/'jobs/metric_audit_v2/result.json') or load(BASE/'jobs/metric_audit/result.json')
 (BASE/'diagnostics.md').write_text('# Round6 diagnostics\n\nMask polarity for han24 remains unresolved: importer calls it ignore mask, while sampled nonzero region IDs contain all observed positive annotations. Formal supervision is blocked pending an authoritative valid-region definition. The fixed diagnostic uses other fully annotated training sources, without changing the formal split.\n\nFrozen diagnostic:\n```json\n'+json.dumps(frozen,indent=2)+'\n```\n\nX1 fallback:\n```json\n'+json.dumps(x1,indent=2)+'\n```\n\nMetric audit:\n```json\n'+json.dumps(metric,indent=2)+'\n```\n\nSame-context inference and overlap weighting are recorded in inference_consistency.json. Probabilities are softmax channel 1, threshold 0.5; labels are never interpolated. Historical float16 probabilities must not be rethresholded to reproduce the float32-derived saved binary masks.\n\nNo diagnostic weight is eligible for formal initialization. Old result tables remain unchanged. A corrected mask pipeline would require all three new controls to be rebuilt; historical Dice cannot isolate SSL benefit after such a change.\n')
 (BASE/'next_decision.md').write_text('# Decision: retain P1; evidence insufficient for new target-data SSL\n\nS0 original specimen/scan linkage is incomplete for the proposed MitoVerse training pool. The instruction explicitly requires quarantine rather than guessed identity; therefore TJ/TM/T0 formal comparison is not launched.\n\nRequired input to unblock: an authoritative mapping of the 31 training exports to original scans/specimens and ROI origins/axes, plus han24 annotation-mask polarity or producer code. This is data evidence, not another approval request.\n\nIndependent fixed-patch engineering diagnostics and historical prediction recounts are reported separately. They do not demonstrate target-domain adaptation or superiority over MitoNet instance segmentation. No extra Gram search or SSL extension is authorized by this round.\n')
 (BASE/'README.md').write_text('# Round6 MitoVerse audit and diagnostics\n\nSee stage_status.json for machine-readable completion and blocking reasons. New SSL is gated on reliable original specimen/scan identity and the segmentation engineering checks. No historical asset has been modified.\n\nReproduction: archived inputs are indexed in archive_index.json; source hashes in source_sha256.json. Run with the recorded meddinov3 Python interpreter from this directory: `python coordinator.py`. Completed or previously launched jobs are not automatically rerun; inspect checkpoint/result/hash evidence before any explicit resumed run.\n\nData evidence: data_manifest.csv, overlap_audit.csv, exposure_by_checkpoint.csv, quarantine.csv, mask_audit.json, split_audit.md. Empty SSL allowlists reflect a failed identity gate, not a completed training run.\n\nDiagnostic evidence: jobs/diagnostic_frozen and optional jobs/diagnostic_x1, fixed_patches.json, inference_consistency.json, diagnostics.md. Metric evidence: metric_audit.csv, comparison_volume_seed.csv, comparison_source_seed.csv, paired_deltas.csv and prediction_index.json. These comparisons describe historical P1/G/C/X1 models, not unrun Round6 controls.\n\nResource protection: one coordinator, UUID-mapped GPU 1–3 only; GPU process checks every 5 seconds, resource/HTTP every 10 seconds, stale heartbeat cutoff 30 seconds, emergency process-group cleanup within 25 seconds. No blanket user/process termination.\n\nDecision and unblocking evidence: next_decision.md.\n')
 save(BASE/'resume_check.json',dict(status='not_applicable',reason='No new SSL launched because S0 identity gate is blocked; diagnostic weights never enter formal training'))
 save(BASE/'protocol.lock.json',dict(status='not_locked',reason='S0 identity gate blocked; a formal protocol lock must not imply approval of unresolved data',formal_training_launched=False))
 for mode in ['frozen','x1']:
  p=BASE/f'jobs/diagnostic_{mode}/overfit_curves.csv'
  if p.exists():
   import shutil
   shutil.copyfile(p,BASE/f'overfit_curves_{mode}.csv')
 files={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.iterdir() if p.is_file() and p.suffix in ['.csv','.json','.md','.py','.patch'] and p.name!='artifact_hashes.json'}
 save(BASE/'artifact_hashes.json',files)
def main():
 (BASE/'jobs').mkdir(exist_ok=True)
 if (BASE/'coordinator_pid.json').exists():raise RuntimeError('Prior coordinator registration exists; verify hashes/process status before explicit recovery')
 import psutil
 save(BASE/'coordinator_pid.json',dict(pid=os.getpid(),birth=psutil.Process().create_time(),python=sys.executable,command=sys.argv,started=time.time()))
 save(BASE/'source_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob('*.py')})
 # Preserve implementation differences, including resource changes, without editing Round5.
 (BASE/'pipeline_diff.patch').write_text(''.join(difflib.unified_diff((R5/'guard.py').read_text().splitlines(True),(BASE/'guard.py').read_text().splitlines(True),fromfile='round5/guard.py',tofile='round6/guard.py'))+'\n# Supervised formal loader unchanged pending authoritative mask semantics.\n')
 guard=Guard();guard.start()
 try:
  update('S0','running');guard.job('data_audit',None,'audit.py');update('S0','blocked',audit_execution='completed',reason='Original specimen/scan mapping and mask semantics unresolved; empty allowed target SSL pool')
  update('S1','running',scope='Independent pipeline diagnostics and saved-prediction metric audit')
  p=guard.job('diagnostic_frozen',2,'diagnostic.py');d=json.loads((p/'result.json').read_text())
  if not d['engineering_gate']:
   p=guard.job('diagnostic_x1',2,'diagnostic.py',['--x1']);d=json.loads((p/'result.json').read_text())
  guard.job('metric_audit',None,'metric_audit.py')
  update('S1','completed' if d['engineering_gate'] else 'blocked',engineering_gate=d['engineering_gate'],metric_recount='completed',formal_mask_semantics='blocked in S0')
  for i in [2,3,4]:update(f'S{i}','blocked',reason='S0 identity gate failed; no formal adaptation experiment or new-model evaluation')
  report();print('Round6 independent diagnostics finished; new SSL blocked by documented data evidence.',flush=True)
 except Exception as e:
  guard.stop_all(repr(e));save(BASE/'execution_error.json',dict(error=repr(e),traceback=traceback.format_exc()))
  states=json.loads((BASE/'stage_status.json').read_text())
  for stage,x in states.items():
   if x['status']=='running':update(stage,'blocked',reason=repr(e))
  report();raise
 finally:guard.close()
if __name__=='__main__':main()
