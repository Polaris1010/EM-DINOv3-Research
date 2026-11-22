"""Fit and score the existing C0 encoder with the original seed-14 crop caches."""
from pathlib import Path
import csv
import importlib
import os
import socket
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

BASE = Path(__file__).resolve().parent
RESULT = BASE/'batch4/c0_matched'

def encoder(name):
    from mixture import B, ROOT, read, sha
    from fit import primus
    import torch
    assert name == 'C0_Vanilla_100k'
    net, p, _ = primus('P0')
    entry = read(B/'pilot_c100_decoder_seeds_20261102/protocol.json')['models'][name]
    path = ROOT/entry['encoder_relative']
    digest = sha(path)
    assert digest == entry['encoder_sha256']
    net.dino_encoder.load_state_dict(torch.load(path,map_location='cpu',mmap=True,weights_only=True)['backbone'],strict=True)
    return net,p,digest

def arm_run(arm):
    import evaluate
    from mixture import read
    name='C0_Vanilla_100k'
    trainer=importlib.import_module(f'primus_{arm}_batch4')
    trainer.primus=encoder
    trainer.ROOT=RESULT/arm
    out=trainer.ROOT/name
    if not (out/'trained.json').exists():
        trainer.train(name)
    old=read(BASE/f'batch4/primus_{arm}/Barrier100/trained.json')
    new=read(out/'trained.json')
    for key in ('draws_sha','init_head_sha','batch_size','steps','seed'):
        assert old[key]==new[key], key
    evaluate.primus=encoder
    evaluate.X=trainer.ROOT
    if not (out/'evaluation.json').exists():
        evaluate.run(name)

def compare():
    from report import truth,write
    from mixture import read
    out=[]
    for arm in ('cem','mix'):
        grouped=defaultdict(lambda: defaultdict(list))
        expected=None
        for model in ('B100','C0','C1'):
            path=RESULT/arm/'C0_Vanilla_100k' if model=='C0' else BASE/f'batch4/primus_{arm}'/('Barrier100' if model=='C1' else model)
            rows=read(path/'evaluation.json')
            if expected is None: expected=truth(rows)
            assert len(rows)==len(expected)==4057 and truth(rows)==expected
            for r in rows:
                for k in ('dice','iou','precision','recall'):
                    if r[k] is not None: grouped[r['split'],r['group'],k][model].append(r[k])
        summaries=defaultdict(lambda:defaultdict(list))
        def record(s,g,k,v):
            out.append(dict(supervision=arm,split=s,group=g,metric=k,**v,
                            C1_minus_C0=v['C1']-v['C0'],C0_minus_B100=v['C0']-v['B100']))
        for (s,g,k),a in sorted(grouped.items()):
            v={m:statistics.mean(vals) for m,vals in a.items()}
            for m,val in v.items(): summaries[s,k][m].append(val)
            if len(v)==3: record(s,g,k,v)
        for (s,k),a in summaries.items(): record(s,'ALL',k,{m:statistics.mean(vals) for m,vals in a.items()})
    write(RESULT/'paired_metrics.csv',out)

def worker(arm,gpu):
    import time
    def query(field):
        return subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu='+field,'--format=csv,noheader,nounits'],text=True).strip()
    while int(query('memory.used'))>=150: time.sleep(30)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=query('uuid'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    py=BASE.parent.parent/'envs/meddinov3/bin/python'
    with (RESULT/f'{arm}.log').open('ab') as log:
        subprocess.run([py,'-u',__file__,arm],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':
    assert socket.gethostname().split('.')[0]=='mira-4090-8'
    if len(sys.argv)==2:
        arm_run(sys.argv[1])
    else:
        RESULT.mkdir(parents=True,exist_ok=True)
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(worker,('cem','mix'),(1,2)))
        compare()
        subprocess.run([sys.executable,BASE/'multiseed_driver.py'],check=True)
