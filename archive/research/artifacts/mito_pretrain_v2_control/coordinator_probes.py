"""Guarded frozen-probe queue after all four SSL seed-0 arms complete."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import time
from pathlib import Path

from guard import Guard, save
from settings import BASE

RUN=BASE.parent
NATURAL=Path('/opt/data/3_117/tanghr/DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
ARMS=('B','L','T','LT')
MILESTONES=(10000,30000,60000,100000)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def encoder_path(arm, step):
    if arm=='N0':
        return NATURAL
    for job in (BASE/'jobs').glob(f'formal_seed0_{arm}_part*'):
        receipt=job/f'ema_step{step}.json'
        if receipt.exists():
            entry=json.loads(receipt.read_text())
            path=Path(entry['path'])
            if not path.exists():
                raise RuntimeError(f'EMA receipt missing weight: {path}')
            if sha(path)!=entry['sha256']:
                raise RuntimeError(f'EMA receipt hash mismatch: {path}')
            return path
    raise RuntimeError(f'Missing seed0 EMA {arm}@{step}')


def checked_job(guard,name,gpu,arm,step,seed,steps):
    output=BASE/'jobs'/name
    if output.exists():
        result=output/'result.json'
        if not result.exists() or json.loads(result.read_text()).get('status')!='complete':
            raise RuntimeError(f'Incomplete existing probe job: {output}')
        return output
    return guard.job(name,gpu,'train_seg.py',[
        '--encoder',str(encoder_path(arm,step)),'--seed',str(seed),
        '--steps',str(steps),'--mode','2d'])


def run_queue(guard,tasks,phase,gpus=(1,2,3)):
    pending=iter(tasks)
    completed=[]
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures={}
        for gpu in gpus:
            item=next(pending,None)
            if item:
                arm,step,seed=item
                name=f'probe_{arm}_{step}_head{seed}'
                futures[pool.submit(checked_job,guard,name,gpu,arm,step,seed,10000)]=(item,gpu)
        while futures:
            ready,_=wait(futures,return_when=FIRST_COMPLETED)
            for future in ready:
                item,gpu=futures.pop(future)
                path=future.result()
                result=json.loads((path/'result.json').read_text())
                if result['completed_steps']!=10000 or result['best_step'] is None:
                    guard.stop_all(f'Probe validation-selection failure: {path}')
                    raise RuntimeError(f'Probe did not complete validation selection: {path}')
                completed.append({'task':item,'gpu':gpu,'best_validation_dice':result['best_validation_dice'],
                                  'best_step':result['best_step'],'job':path.name})
                save(BASE/f'{phase}_status.json',{'status':'running','completed':completed,
                                                'remaining':len(tasks)-len(completed),'updated':time.time()})
                next_item=next(pending,None)
                if next_item:
                    arm,step,seed=next_item
                    name=f'probe_{arm}_{step}_head{seed}'
                    futures[pool.submit(checked_job,guard,name,gpu,arm,step,seed,10000)]=(next_item,gpu)
    save(BASE/f'{phase}_status.json',{'status':'completed','completed':completed,'ended':time.time()})
    return completed


def select_w():
    scores={};sources={}
    for arm in ('N0',*ARMS):
        values=[];source_values={}
        for seed in (0,1,2):
            path=BASE/f'jobs/probe_{arm}_{100000 if arm!="N0" else 0}_head{seed}'
            result=json.loads((path/'result.json').read_text())
            report=json.loads((path/f"validation_step{result['best_step']}.json").read_text())
            values.append(report['source_macro_dice'])
            source_values[seed]={row['source']:row['mean_dice'] for row in report['source_rows']}
        scores[arm]=values;sources[arm]=source_values
    admission=json.loads((RUN/'evidence/formal_seed0_admission.json').read_text())
    cost={r['arm']:r['mean_step_seconds'] for r in admission['arms']}
    mean=lambda values:sum(values)/len(values)
    winner=sorted(('L','T','LT'),key=lambda arm:(-mean(scores[arm]),cost[arm],arm))[0]
    deltas=[x-y for x,y in zip(scores[winner],scores['B'])]
    source_deltas={name:mean([sources[winner][seed][name]-sources['B'][seed][name] for seed in (0,1,2)])
                   for name in sources['B'][0]}
    passed=mean(deltas)>=.005 and sum(d>0 for d in deltas)>=2 and all(d>=0 for d in source_deltas.values())
    result={'status':'complete','W':winner,'scores_by_arm_and_head_seed':scores,
            'W_minus_B_by_seed':deltas,'W_minus_B_source_mean':source_deltas,
            'frozen_probe_independent_repeat_gate':passed,
            'tie_rule':'lower measured 100-step SSL time, then arm name'}
    save(RUN/'evidence/probe_selection.json',result)
    return result


def main():
    formal=json.loads((BASE/'formal_seed0_status.json').read_text())
    if formal.get('status')!='completed' or {x['arm'] for x in formal['completed']}!=set(ARMS):
        raise SystemExit('All four seed0 SSL arms must complete first')
    if (BASE/'STOP.json').exists():
        raise SystemExit('STOP marker blocks probe dispatch')
    guard=Guard()
    try:
        guard.start()
        save(BASE/'probes_status.json',{'status':'running','started':time.time()})
        for index,arm in enumerate(('N0',*ARMS)):
            step=0 if arm=='N0' else 100000
            checked_job(guard,f'probe_smoke_{arm}',(1,2,3)[index%3],arm,step,0,10)
        seed0=[('N0',0,0)]+[(arm,step,0) for step in MILESTONES for arm in ARMS]
        extra=[(arm,0 if arm=='N0' else 100000,seed)
               for seed in (1,2) for arm in ('N0',*ARMS)]
        if (BASE.parent/'evidence/expanded_coverage.json').exists():
            from expanded_queue import run_expanded
            with ThreadPoolExecutor(max_workers=1) as expansion_pool:
                expansion=expansion_pool.submit(run_expanded,guard,3)
                run_queue(guard,seed0,'probes_seed0',gpus=(1,2))
                run_queue(guard,extra,'probes_extra_heads',gpus=(1,2))
                expansion.result()
        else:
            run_queue(guard,seed0,'probes_seed0')
            run_queue(guard,extra,'probes_extra_heads')
        selected=select_w()
        save(BASE/'probes_status.json',{'status':'completed','selection':selected,'ended':time.time()})
        print(json.dumps(selected))
        return 0
    except Exception as exc:
        guard.stop_all(repr(exc))
        save(BASE/'probes_status.json',{'status':'blocked','reason':repr(exc),'ended':time.time()})
        raise
    finally:
        guard.close()


if __name__=='__main__':
    raise SystemExit(main())
