"""Freeze validation-selected heads, evaluate legacy development volumes once, report semantic results."""

from concurrent.futures import FIRST_COMPLETED,ThreadPoolExecutor,wait
import hashlib
import json
import time

from guard import Guard,save
from settings import BASE

RUN=BASE.parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def selected_jobs(winner,independent,context):
    jobs=[]
    for seed in (0,1,2):
        for arm in ('N0','B','L','T','LT'):
            jobs.append(('frozen_seed0',arm,seed,f'probe_{arm}_{0 if arm=="N0" else 100000}_head{seed}'))
        for arm in ('N0','B',winner):
            jobs.append(('finetune_seed0',arm,seed,f'finetune_{arm}_head{seed}'))
        if independent:
            for arm in ('B',winner):
                jobs.append(('frozen_seed1',arm,seed,f'probe_seed1_{arm}_head{seed}'))
                jobs.append(('finetune_seed1',arm,seed,f'finetune_seed1_{arm}_head{seed}'))
        if context:
            for arm in ('B',winner):
                jobs.append(('context_seed0',arm,seed,f'context_{arm}_head{seed}'))
    return jobs


def run_eval(guard,task,gpu):
    category,arm,seed,source=task
    name=f'eval_{category}_{arm}_head{seed}'
    out=BASE/'jobs'/name
    if out.exists():
        result=out/'result.json'
        if not result.exists() or json.loads(result.read_text()).get('status')!='complete':
            raise RuntimeError(f'Incomplete existing evaluation job: {out}')
        return out
    return guard.job(name,gpu,'evaluate.py',[
        '--checkpoint',str(BASE/'jobs'/source/'head_best.pth'),
        '--split','legacy_dev_test','--output',str(out/'metrics.json')])


def main():
    probes=json.loads((BASE/'probes_status.json').read_text())
    fine=json.loads((BASE/'finetune_status.json').read_text())
    if probes['status']!='completed' or fine['status']!='completed':
        raise SystemExit('Validation selection must complete before final evaluation')
    decision=json.loads((RUN/'evidence/finetune_selection.json').read_text())
    winner=decision['W']
    independent=decision['confirm_independent_ssl']
    context=False
    if independent:
        branch=json.loads((BASE/'independent_status.json').read_text())
        if branch['status']!='completed':
            raise SystemExit('Independent SSL branch incomplete')
        context=json.loads((RUN/'evidence/independent_gate.json').read_text())['run_2p5d']
        if context and json.loads((BASE/'context_status.json').read_text())['status']!='completed':
            raise SystemExit('Triggered 2.5D branch incomplete')
    tasks=selected_jobs(winner,independent,context)
    entries=[]
    for category,arm,seed,job in tasks:
        source=BASE/'jobs'/job
        result=json.loads((source/'result.json').read_text())
        checkpoint=source/'head_best.pth'
        if result['status']!='complete' or not checkpoint.exists():
            raise RuntimeError(f'Invalid validation-selected head: {source}')
        entries.append({'category':category,'arm':arm,'head_seed':seed,'source_job':job,
                        'selected_step':result['best_step'],
                        'validation_dice':result['best_validation_dice'],
                        'checkpoint':str(checkpoint),'sha256':sha(checkpoint)})
    lock=RUN/'evidence/validation_selection.lock.json'
    if lock.exists():
        old=json.loads(lock.read_text())
        if old['entries']!=entries:
            raise RuntimeError('Validation lock differs from existing immutable selection')
    else:
        save(lock,{'status':'locked','entries':entries,
                   'allowed_checkpoint_sha256':[row['sha256'] for row in entries],
                   'legacy_dev_test_is_development_data':True})
    if (BASE/'STOP.json').exists():
        raise SystemExit('STOP marker blocks final evaluation')
    guard=Guard()
    completed=[]
    try:
        guard.start()
        save(BASE/'final_eval_status.json',{'status':'running','total_jobs':len(tasks),'started':time.time()})
        pending=iter(tasks)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={}
            for gpu in (1,2,3):
                task=next(pending,None)
                if task:
                    futures[pool.submit(run_eval,guard,task,gpu)]=(task,gpu)
            while futures:
                ready,_=wait(futures,return_when=FIRST_COMPLETED)
                for future in ready:
                    task,gpu=futures.pop(future)
                    path=future.result()
                    result=json.loads((path/'result.json').read_text())
                    if result['volumes']!=16:
                        guard.stop_all(f'Incomplete development-test volume coverage: {path}')
                        raise RuntimeError(f'Incomplete volume coverage: {path}')
                    completed.append({'task':task,'gpu':gpu,
                                      'source_macro_dice':result['source_macro_dice']})
                    save(BASE/'final_eval_status.json',{'status':'running','completed':completed,
                                                       'total_jobs':len(tasks),'updated':time.time()})
                    nxt=next(pending,None)
                    if nxt:
                        futures[pool.submit(run_eval,guard,nxt,gpu)]=(nxt,gpu)
        groups={}
        for category,arm,seed,_ in tasks:
            name=f'eval_{category}_{arm}_head{seed}'
            report=json.loads((BASE/'jobs'/name/'metrics.json').read_text())
            key=f'{category}:{arm}'
            groups.setdefault(key,[]).append({'dice':report['source_macro_dice'],
                'sources':{row['source']:row['mean_dice'] for row in report['source_rows']}})
        summary={name:{'head_seed_dice':[row['dice'] for row in values],
                       'mean_dice':sum(row['dice'] for row in values)/len(values),
                       'mean_dice_by_source':{source:sum(row['sources'][source] for row in values)/len(values)
                                              for source in values[0]['sources']}}
                 for name,values in sorted(groups.items())}
        mitonet=json.loads((RUN/'evidence/mitonet_comparison_summary_archived.json').read_text())
        baseline=next(row for row in mitonet['rows'] if row['model']=='MitoNet_M0_semantic' and
                      row['split']=='legacy_dev_test' and row['complete'])
        report={'status':'complete','legacy_dev_test_only':True,'W':winner,
                'total_models':len(tasks),'groups':summary,
                'mitonet_zero_shot_semantic_source_macro_dice':baseline['source_macro_dice'],
                'mitonet_zero_shot_semantic_source_dice':baseline['source_dice'],
                'mitonet_protocol':'archived fixed official TorchScript M0; zero-shot application comparison',
                'interpretation':'Previously viewed development test, not a sealed independent holdout; supervised and zero-shot models have different target-data exposure',
                'instance_metrics':'not evaluated; no fabricated 2D PQ/AP claim'}
        save(RUN/'evidence/final_semantic_report.json',report)
        save(BASE/'final_eval_status.json',{'status':'completed','completed':completed,'report':report,
                                           'ended':time.time()})
        (RUN/'next_decision.md').write_text(
            '# MitoVerse v2 下一步\n\n完整 semantic 评估已按验证集锁定选择执行。'
            '固定开发测试集不是密封外部测试；MitoNet 是零样本应用对照。'
            '请查看 `evidence/final_semantic_report.json`、逐作业 `metrics.json` 和条件门槛收据。'
            '未开展 2D 实例指标时，不得声称实例性能。\n')
        print(json.dumps(report))
        return 0
    except Exception as exc:
        guard.stop_all(repr(exc))
        save(BASE/'final_eval_status.json',{'status':'blocked','completed':completed,
                                           'reason':repr(exc),'ended':time.time()})
        raise
    finally:
        guard.close()


if __name__=='__main__':
    raise SystemExit(main())
