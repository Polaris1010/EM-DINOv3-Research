"""Guarded 2D last-four-block N0/B/W fine-tune and independent-SSL gate."""

from concurrent.futures import FIRST_COMPLETED,ThreadPoolExecutor,wait
import json
import time

from guard import Guard,save
from settings import BASE
from coordinator_probes import encoder_path

RUN=BASE.parent


def checked_job(guard,name,gpu,arm,seed,steps):
    out=BASE/'jobs'/name
    if out.exists():
        result=out/'result.json'
        if not result.exists() or json.loads(result.read_text()).get('status')!='complete':
            raise RuntimeError(f'Incomplete existing fine-tune job: {out}')
        return out
    return guard.job(name,gpu,'train_seg.py',[
        '--encoder',str(encoder_path(arm,100000)),
        '--seed',str(seed),'--steps',str(steps),'--mode','2d','--fine-tune-last4'])


def main():
    probes=json.loads((BASE/'probes_status.json').read_text())
    if probes['status']!='completed':
        raise SystemExit('Frozen probes incomplete')
    selection=json.loads((RUN/'evidence/probe_selection.json').read_text())
    winner=selection['W']
    arms=('N0','B',winner)
    if (BASE/'STOP.json').exists():
        raise SystemExit('STOP marker blocks fine-tune')
    guard=Guard()
    completed=[]
    try:
        guard.start()
        save(BASE/'finetune_status.json',{'status':'running','W':winner,'started':time.time()})
        for index,arm in enumerate(arms):
            checked_job(guard,f'finetune_smoke_{arm}',(1,2,3)[index],arm,0,10)
        tasks=[(arm,seed) for seed in (0,1,2) for arm in arms]
        pending=iter(tasks)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={}
            for gpu in (1,2,3):
                arm,seed=next(pending)
                name=f'finetune_{arm}_head{seed}'
                futures[pool.submit(checked_job,guard,name,gpu,arm,seed,20000)]=((arm,seed),gpu)
            while futures:
                ready,_=wait(futures,return_when=FIRST_COMPLETED)
                for future in ready:
                    (arm,seed),gpu=futures.pop(future)
                    path=future.result()
                    result=json.loads((path/'result.json').read_text())
                    if result['completed_steps']!=20000 or result['best_step'] is None:
                        guard.stop_all(f'Fine-tune validation-selection failure: {path}')
                        raise RuntimeError(f'Fine-tune incomplete: {path}')
                    completed.append({'arm':arm,'seed':seed,'gpu':gpu,
                                      'best_step':result['best_step'],
                                      'best_validation_dice':result['best_validation_dice']})
                    save(BASE/'finetune_status.json',{'status':'running','W':winner,
                                                     'completed':completed,'updated':time.time()})
                    next_item=next(pending,None)
                    if next_item:
                        a,s=next_item;name=f'finetune_{a}_head{s}'
                        futures[pool.submit(checked_job,guard,name,gpu,a,s,20000)]=(next_item,gpu)
        scores={};sources={}
        for arm in arms:
            scores[arm]=[];sources[arm]={}
            for seed in (0,1,2):
                path=BASE/f'jobs/finetune_{arm}_head{seed}'
                result=json.loads((path/'result.json').read_text())
                report=json.loads((path/f"validation_step{result['best_step']}.json").read_text())
                scores[arm].append(report['source_macro_dice'])
                sources[arm][seed]={r['source']:r['mean_dice'] for r in report['source_rows']}
        mean=lambda seq:sum(seq)/len(seq)
        deltas=[x-y for x,y in zip(scores[winner],scores['B'])]
        source_deltas={name:mean([sources[winner][seed][name]-sources['B'][seed][name]
                                  for seed in (0,1,2)]) for name in sources['B'][0]}
        finetune_gate=mean(deltas)>=.005 and sum(d>0 for d in deltas)>=2 and all(d>=0 for d in source_deltas.values())
        decision={'status':'complete','W':winner,'scores_by_arm_and_head_seed':scores,
                  'W_minus_B_by_seed':deltas,'W_minus_B_source_mean':source_deltas,
                  'finetune_independent_repeat_gate':finetune_gate,
                  'frozen_probe_independent_repeat_gate':selection['frozen_probe_independent_repeat_gate'],
                  'confirm_independent_ssl':finetune_gate or selection['frozen_probe_independent_repeat_gate']}
        save(RUN/'evidence/finetune_selection.json',decision)
        save(BASE/'finetune_status.json',{'status':'completed','W':winner,
                                        'completed':completed,'decision':decision,'ended':time.time()})
        print(json.dumps(decision))
        return 0
    except Exception as exc:
        guard.stop_all(repr(exc))
        save(BASE/'finetune_status.json',{'status':'blocked','reason':repr(exc),'ended':time.time()})
        raise
    finally:
        guard.close()


if __name__=='__main__':
    raise SystemExit(main())
