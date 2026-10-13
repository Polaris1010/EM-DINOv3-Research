"""Persistent dependency watcher for guarded probes, fine-tune and final evaluation."""

import json
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path

RUN=Path(__file__).resolve().parent.parent
SRC=RUN/'src'
PY='/opt/data/3_117/tanghr/envs/meddinov3/bin/python'
SERVICES={
    'formal':('mito-pretrain-v2-seed0','formal_seed0_status.json',None),
    'probes':('mito-pretrain-v2-probes','probes_status.json','coordinator_probes.py'),
    'finetune':('mito-pretrain-v2-finetune','finetune_status.json','coordinator_finetune.py'),
    'independent':('mito-pretrain-v2-independent','independent_status.json','coordinator_independent.py'),
    'final':('mito-pretrain-v2-final','final_eval_status.json','coordinator_final_eval.py'),
}


def active(unit):
    return subprocess.run(['systemctl','is-active','--quiet',unit],check=False).returncode==0


def status(name):
    path=SRC/SERVICES[name][1]
    return json.loads(path.read_text())['status'] if path.exists() else 'pending'


def launch(name):
    unit,_,script=SERVICES[name]
    if active(unit):
        return
    logfile=SRC/f'{name}_coordinator.log'
    command=f'exec {PY} -u {SRC/script} >> {logfile} 2>&1'
    subprocess.run(['systemd-run',f'--unit={unit}',f'--property=WorkingDirectory={SRC}',
                    '--property=KillMode=control-group','--property=Restart=no',
                    '/bin/bash','-c',command],check=True)
    time.sleep(5)
    if not active(unit):
        raise RuntimeError(f'{name} service did not become active')


def next_stage():
    if status('formal')!='completed':
        return None
    if status('probes')!='completed':
        return 'probes'
    if status('finetune')!='completed':
        return 'finetune'
    decision=json.loads((RUN/'evidence/finetune_selection.json').read_text())
    if decision['confirm_independent_ssl'] and status('independent')!='completed':
        return 'independent'
    if status('final')!='completed':
        return 'final'
    return 'done'


def update_stage_status():
    path=RUN/'stage_status.json'
    if not path.exists():
        return
    data=json.loads(path.read_text())
    data['last_updated_utc']=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    mapping={'formal':'03_ssl_seed0','probes':'04_frozen_probes',
             'finetune':'05_matched_2d_finetune','independent':'06_independent_ssl_confirmation',
             'final':'08_final_evaluation'}
    for name,key in mapping.items():
        state=status(name)
        if state!='pending':
            data.setdefault(key,{})['status']=state
    progress={}
    for arm in ('B','L','T','LT'):
        paths=sorted((SRC/'jobs').glob(f'formal_seed0_{arm}_part*/progress.json'))
        if paths:
            progress[arm]=json.loads(paths[-1].read_text())['step']
    data['new_ssl_updates']=progress
    context=SRC/'context_status.json'
    if context.exists():
        data.setdefault('07_2p5d_validation',{})['status']=json.loads(context.read_text())['status']
    fine=RUN/'evidence/finetune_selection.json'
    if fine.exists() and not json.loads(fine.read_text())['confirm_independent_ssl']:
        data.setdefault('06_independent_ssl_confirmation',{})['status']='skipped'
        data.setdefault('07_2p5d_validation',{})['status']='skipped'
    temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n')
    temp.replace(path)


def main():
    while True:
        update_stage_status()
        if (SRC/'STOP.json').exists():
            raise SystemExit('Resource STOP marker blocks all dependent batches')
        for name in SERVICES:
            if status(name)=='blocked':
                raise SystemExit(f'{name} batch blocked; no automatic retry')
        if status('formal')=='running' and not active(SERVICES['formal'][0]):
            raise SystemExit('Formal SSL service stopped before completion; inspect checkpoint and STOP')
        stage=next_stage()
        if stage=='done':
            return
        if stage is not None:
            previous={'probes':'formal','finetune':'probes',
                      'independent':'finetune','final':'finetune'}[stage]
            if stage=='final':
                decision=json.loads((RUN/'evidence/finetune_selection.json').read_text())
                if decision['confirm_independent_ssl']:
                    previous='independent'
            if not active(SERVICES[previous][0]):
                if status(stage)=='pending':
                    launch(stage)
                elif status(stage)=='running' and not active(SERVICES[stage][0]):
                    raise SystemExit(f'{stage} service stopped before completion; inspect checkpoint and STOP')
        time.sleep(60)


if __name__=='__main__':
    main()
