"""Explicit one-time recovery: keep completed audit/fitting, resume verified volume prefix."""
import os,sys,time,json,hashlib,psutil
from settings import *
from guard import Guard,save
from coordinator import update,report
def main():
 prior=json.loads((BASE/'coordinator_pid.json').read_text())
 if psutil.pid_exists(prior['pid']) and psutil.Process(prior['pid']).status()!=psutil.STATUS_ZOMBIE:raise RuntimeError('Prior coordinator still alive; no second guard may run')
 assert json.loads((BASE/'jobs/data_audit/result.json').read_text())['P1_hash_match']
 d=json.loads((BASE/'jobs/diagnostic_frozen/result.json').read_text());assert d['steps']==1000 and d['engineering_gate'] and d['frozen_weights_unchanged']
 assert json.loads((BASE/'inference_consistency.json').read_text())['status']=='pass'
 for n in ['coordinator_pid.json','STOP.json','execution_error.json','source_sha256.json']:
  p=BASE/n
  if p.exists():p.rename(BASE/(p.stem+'_before_s1_resume'+p.suffix))
 save(BASE/'coordinator_pid.json',dict(pid=os.getpid(),birth=psutil.Process().create_time(),python=sys.executable,command=sys.argv,started=time.time()))
 save(BASE/'source_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob('*.py')})
 g=Guard();g.start()
 try:
  update('S1','running',scope='Resume completed-volume prefix; add fixed validation patch audit; no new training')
  g.job('metric_audit_v2',None,'metric_audit.py')
  update('S1','completed',engineering_gate=True,metric_recount='completed',validation_patch_audit='completed',formal_mask_semantics='blocked in S0')
  for i in [2,3,4]:update(f'S{i}','blocked',reason='S0 original specimen/scan identity gate failed; no new SSL')
  report();print('Round6 finished: independent diagnostics complete, formal adaptation blocked by data evidence.',flush=True)
 except Exception as e:
  g.stop_all(repr(e));update('S1','blocked',reason=repr(e));report();raise
 finally:g.close()
if __name__=='__main__':main()
