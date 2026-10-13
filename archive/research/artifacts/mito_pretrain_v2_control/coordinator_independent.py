"""Conditional independent SSL seed1, paired heads, and optional 2.5D validation."""

from concurrent.futures import FIRST_COMPLETED,ThreadPoolExecutor,wait
import hashlib
import json
import time
from pathlib import Path

from guard import Guard,save
from settings import BASE
from coordinator_probes import encoder_path

RUN=BASE.parent


def seed1_encoder(arm):
    receipt=BASE/f'jobs/independent_{arm}_seed1/ema_step100000.json'
    x=json.loads(receipt.read_text())
    path=Path(x['path'])
    if not path.exists():
        raise RuntimeError(f'Missing seed1 weight {path}')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    if h.hexdigest()!=x['sha256']:
        raise RuntimeError(f'Seed1 weight hash mismatch: {path}')
    return path


def checked(guard,name,gpu,script,args,target):
    out=BASE/'jobs'/name
    if out.exists():
        path=out/'result.json'
        if not path.exists():
            raise RuntimeError(f'Incomplete existing job {out}')
        result=json.loads(path.read_text())
        if result.get('status')!='complete' or result.get('completed_steps')!=target:
            raise RuntimeError(f'Existing result incomplete: {out}')
        return out
    path=guard.job(name,gpu,script,args)
    result=json.loads((path/'result.json').read_text())
    if result.get('status')!='complete' or result.get('completed_steps')!=target:
        guard.stop_all(f'Incomplete result: {path}')
        raise RuntimeError(f'Incomplete result: {path}')
    return path


def downstream_args(weight,seed,steps,mode='2d',fine=False):
    args=['--encoder',str(weight),'--seed',str(seed),'--steps',str(steps),'--mode',mode]
    if fine:
        args.append('--fine-tune-last4')
    return args


def run_queue(guard,tasks,status_name):
    pending=iter(tasks)
    results=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={}
        for gpu in (1,2,3):
            task=next(pending,None)
            if task:
                futures[pool.submit(checked,guard,task['name'],gpu,'train_seg.py',
                                    task['args'],task['steps'])]=(task,gpu)
        while futures:
            ready,_=wait(futures,return_when=FIRST_COMPLETED)
            for future in ready:
                task,gpu=futures.pop(future)
                path=future.result()
                result=json.loads((path/'result.json').read_text())
                if result.get('best_step') is None:
                    guard.stop_all(f'Missing validation selection: {path}')
                    raise RuntimeError(f'Missing validation selection: {path}')
                results.append({'name':task['name'],'gpu':gpu,
                                'best_step':result['best_step'],
                                'best_validation_dice':result['best_validation_dice']})
                save(BASE/status_name,{'status':'running','completed':results,'updated':time.time()})
                nxt=next(pending,None)
                if nxt:
                    futures[pool.submit(checked,guard,nxt['name'],gpu,'train_seg.py',
                                        nxt['args'],nxt['steps'])]=(nxt,gpu)
    save(BASE/status_name,{'status':'completed','completed':results,'ended':time.time()})
    return results


def read_best(name):
    path=BASE/'jobs'/name
    result=json.loads((path/'result.json').read_text())
    report=json.loads((path/f"validation_step{result['best_step']}.json").read_text())
    return report['source_macro_dice'],{x['source']:x['mean_dice'] for x in report['source_rows']}


def independent_gate(winner):
    frozen0=json.loads((RUN/'evidence/probe_selection.json').read_text())
    finetune0=json.loads((RUN/'evidence/finetune_selection.json').read_text())
    output={'status':'complete','W':winner,'ssl_seeds':[0,1]}
    for kind,seed0 in (('frozen',frozen0),('finetune',finetune0)):
        values={arm:[] for arm in ('B',winner)}
        sources={arm:[] for arm in ('B',winner)}
        for arm in ('B',winner):
            for seed in (0,1,2):
                name=f'{"probe" if kind=="frozen" else "finetune"}_seed1_{arm}_head{seed}'
                score,by_source=read_best(name)
                values[arm].append(score);sources[arm].append(by_source)
        seed0_delta=sum(seed0['scores_by_arm_and_head_seed'][winner])/3-sum(seed0['scores_by_arm_and_head_seed']['B'])/3
        seed1_delta=sum(values[winner])/3-sum(values['B'])/3
        source0=seed0['W_minus_B_source_mean']
        source1={source:sum(sources[winner][i][source]-sources['B'][i][source] for i in range(3))/3
                 for source in sources['B'][0]}
        combined_source={source:(source0[source]+source1[source])/2 for source in source0}
        output[kind]={'by_ssl_seed':[seed0_delta,seed1_delta],
                      'aggregate_delta':(seed0_delta+seed1_delta)/2,
                      'source_delta_aggregate':combined_source,
                      'head_seed1_values':values,
                      'gate':(seed0_delta+seed1_delta)/2>=.005 and all(v>=0 for v in combined_source.values())}
    output['run_2p5d']=output['frozen']['gate'] and output['finetune']['gate']
    save(RUN/'evidence/independent_gate.json',output)
    return output


def main():
    decision=json.loads((RUN/'evidence/finetune_selection.json').read_text())
    if decision['status']!='complete' or not decision['confirm_independent_ssl']:
        raise SystemExit('Independent SSL branch was not triggered')
    if (BASE/'STOP.json').exists():
        raise SystemExit('STOP marker blocks independent SSL')
    winner=decision['W']
    guard=Guard()
    try:
        guard.start()
        save(BASE/'independent_status.json',{'status':'running','W':winner,'started':time.time()})
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(checked,guard,f'independent_{arm}_seed1',gpu,'train_ssl.py',
                                 ['--arm',arm,'--batch','16','--steps','100000','--seed','1'],100000):arm
                     for arm,gpu in (('B',1),(winner,2))}
            ssl_jobs={futures[future]:future.result() for future in futures}
        for arm,path in ssl_jobs.items():
            for step in (10000,30000,60000,100000):
                if not (path/f'ema_step{step}.json').exists():
                    raise RuntimeError(f'{arm} seed1 missing EMA {step}')
        save(BASE/'independent_status.json',{'status':'ssl_complete','W':winner,'updated':time.time()})
        for arm in ('B',winner):
            weight=seed1_encoder(arm)
            for kind in ('frozen','finetune'):
                fine=kind=='finetune'
                checked(guard,f'{kind}_seed1_smoke_{arm}',3,'train_seg.py',
                        downstream_args(weight,0,10,fine=fine),10)
        for kind in ('frozen','finetune'):
            fine=kind=='finetune';steps=20000 if fine else 10000
            tasks=[{'name':f'{"probe" if not fine else "finetune"}_seed1_{arm}_head{seed}',
                    'args':downstream_args(seed1_encoder(arm),seed,steps,fine=fine),'steps':steps}
                   for seed in (0,1,2) for arm in ('B',winner)]
            run_queue(guard,tasks,f'independent_{kind}_status.json')
        gate=independent_gate(winner)
        if gate['run_2p5d']:
            for arm in ('B',winner):
                checked(guard,f'context_smoke_{arm}',3,'train_seg.py',
                        downstream_args(encoder_path(arm,100000),0,10,'2p5d',True),10)
            tasks=[{'name':f'context_{arm}_head{seed}',
                    'args':downstream_args(encoder_path(arm,100000),seed,20000,'2p5d',True),
                    'steps':20000}
                   for seed in (0,1,2) for arm in ('B',winner)]
            run_queue(guard,tasks,'context_status.json')
        else:
            save(BASE/'context_status.json',{'status':'skipped','reason':'Independent frozen and fine-tune gate not both passed'})
        save(BASE/'independent_status.json',{'status':'completed','W':winner,'gate':gate,'ended':time.time()})
        print(json.dumps(gate))
        return 0
    except Exception as exc:
        guard.stop_all(repr(exc))
        save(BASE/'independent_status.json',{'status':'blocked','reason':repr(exc),'ended':time.time()})
        raise
    finally:
        guard.close()


if __name__=='__main__':
    raise SystemExit(main())
