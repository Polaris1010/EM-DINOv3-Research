"""Finish seed0 SSL while free GPUs run the final-checkpoint frozen probes."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json,time
from guard import Guard,save
from settings import BASE
from coordinator_formal_seed0 import run_arm,file_sha,STATUS
from coordinator_probes import checked_job

def probe(guard,arm,gpu):
    step=0 if arm=='N0' else 100000
    checked_job(guard,f'probe_smoke_{arm}',gpu,arm,step,0,10)
    path=checked_job(guard,f'probe_{arm}_{step}_head0',gpu,arm,step,0,10000)
    result=json.loads((path/'result.json').read_text())
    if result['completed_steps']!=10000 or result['best_step'] is None:
        raise RuntimeError(f'Invalid completed probe: {path}')
    return {'arm':arm,'gpu':gpu,'job':path.name,'best_validation_dice':result['best_validation_dice'],'best_step':result['best_step']}

def main():
    if (BASE/'STOP.json').exists():raise RuntimeError('STOP marker blocks dispatch')
    admission=json.loads((BASE.parent/'evidence/formal_seed0_admission.json').read_text())
    for name,expected in admission['method_hashes'].items():
        path=(BASE.parent/'topology_calibration'/name if name=='topology_selected_locked.json'
              else BASE.parent/'data'/name if name=='pretrain_train.json' else BASE/name)
        if file_sha(path)!=expected:raise RuntimeError(f'Changed SSL source: {name}')
    completed=[]
    for arm,gpu in [('B',1),('L',2),('T',3)]:
        paths=list((BASE/'jobs').glob(f'formal_seed0_{arm}_part*/result.json'))
        results=[json.loads(p.read_text()) for p in paths]
        if not any(r.get('status')=='complete' and r['completed_steps']==100000 for r in results):
            raise RuntimeError(f'{arm} is not complete')
        completed.append({'arm':arm,'gpu':gpu,'status':'completed','completed_steps':100000})
    guard=Guard();pool=ThreadPoolExecutor(max_workers=3);done=[];pending=['N0','B','L','T','LT']
    try:
        guard.start()
        save(STATUS,{'status':'running','completed':completed,'active_assignments':[['LT',1]],'scheduler':'overlap','started':time.time()})
        active={pool.submit(run_arm,guard,'LT',1):('ssl','LT',1)}
        free=[2,3];ssl_done=False
        while active or pending:
            for gpu in list(free):
                ready_arm=next((a for a in pending if a!='LT' or ssl_done),None)
                if ready_arm is None:continue
                pending.remove(ready_arm);free.remove(gpu)
                active[pool.submit(probe,guard,ready_arm,gpu)]=('probe',ready_arm,gpu)
            save(BASE/'overlap_status.json',{'status':'running','completed_probes':done,'pending':pending,
                 'active':[list(x) for x in active.values()],'ssl_complete':ssl_done,'updated':time.time()})
            if not active:break
            ready,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
            for future in ready:
                kind,arm,gpu=active.pop(future);row=future.result();free.append(gpu)
                if kind=='ssl':
                    completed.append(row);ssl_done=True
                    save(STATUS,{'status':'completed','completed':completed,'ended':time.time(),
                        'note':'Service continues final-checkpoint probes; watcher waits for service exit.'})
                else:done.append(row)
        save(BASE/'overlap_status.json',{'status':'completed','completed_probes':done,'ssl_complete':ssl_done,'ended':time.time()})
    except Exception as exc:
        guard.stop_all(repr(exc))
        save(BASE/'overlap_status.json',{'status':'blocked','reason':repr(exc),'completed_probes':done})
        raise
    finally:
        pool.shutdown(wait=True)
        guard.close()

if __name__=='__main__':main()
