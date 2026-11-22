"""Fresh MB-1/MV3 native-grid inference with three saved supervised head seeds."""
import csv
import json
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

BASE=Path(__file__).resolve().parent
OUT=BASE/'zeroshot_repeat_20261120'
MODELS=('P0','B100','Barrier100')
OFF='Official-MitoNet-v1'
SEEDS=(20261014,20261015,20261016)
METRICS=('dice','iou','precision','recall')

def source(seed):
    return BASE if seed==SEEDS[0] else BASE.with_name(BASE.name+f'_b4_s{seed}')

def evaluate(model,seed):
    import numpy as np
    import torch
    root=source(seed)
    sys.path.insert(0,str(root))
    from mixture import MB,M,read,sha,atomic,arrays
    from evaluate import metrics
    sys.path.insert(0,str(MB/'code'))
    from common import gpu_guard,config,Planes
    gpu=gpu_guard()
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if model==OFF:
        from infer import mitonet
        import runpy; ref_records=runpy.run_path(str(BASE/'report.py'))['ref_records']
        spec=next(r for r in config()['runs'] if r['model']==OFF)
        ck=Path(spec['checkpoint_path'])
        digest=sha(ck)
        assert digest==spec['checkpoint_sha256']
        engine=mitonet(spec)
        predict=lambda a: np.asarray(engine(a))>0
        old=ref_records()
    else:
        from fit import primus
        folder=root/'batch4/primus_cem'/model
        ck=folder/'model_40000.pt'
        meta=read(folder/'trained.json')
        digest=sha(ck)
        assert meta['seed']==seed and meta['steps']==40000 and digest==meta['checkpoint_sha']
        net,p,enc=primus(model)
        assert enc==meta['init_encoder_sha']
        missing,extra=net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=False)
        assert not extra and all(k.startswith('dino_encoder.') for k in missing)
        net.eval()
        predict=lambda a:p.sliding(net,a,tile_batch_size=16)
        old=read(folder/'evaluation.json')
    def counts(a,b,v=None):
        pred=torch.as_tensor(predict(a),device='cuda')
        gt=torch.as_tensor(np.asarray(b)>0,device='cuda')
        assert gt.shape==pred.shape
        if v is not None:
            valid=torch.as_tensor(np.asarray(v)>0,device='cuda')
            gt,pred=gt[valid],pred[valid]
        tp,n_gt,n_pred=torch.stack([(pred&gt).sum(),gt.sum(),pred.sum()]).tolist()
        return np.array([tp,n_pred-tp,n_gt-tp,pred.numel()-n_gt-n_pred+tp],np.int64)
    rows=[]
    with torch.inference_mode():
        for r in read(MB/'MB1_benchmark_manifest.json')['entries']:
            a,b=[Planes(r[key],r['shape_zyx']) for key in ('image_path','label_path')]
            for z in range(r['shape_zyx'][0]):
                rows.append(dict(split='mb1',group=r['dataset'],name=r['id'],z=z,**metrics(counts(a[z],b[z]))))
            a.close();b.close()
        for r in read(M/'MV-Generalist.json')['entries']:
            if r['supervised_role']!='dev':continue
            a,b,v,j=arrays(r['id'])
            total=sum((counts(a[z],b[z],v[z] if v is not None else None) for z in j['valid_z']),np.zeros(4,np.int64))
            rows.append(dict(split='mv3',group=r['source_id'],name=r['id'],z=0,**metrics(total)))
    def indexed(rs):
        return {(r['split'],r['group'],r['name'],r['z']):tuple(r[k] for k in ('tp','fp','fn','tn')) for r in rs if r['split'] in ('mv3','mb1')}
    a,b=indexed(old),indexed(rows)
    assert a.keys()==b.keys() and len(b)==len(rows)==2633
    for k in a:
        assert a[k][0]+a[k][2]==b[k][0]+b[k][2] and sum(a[k])==sum(b[k])
    atomic(OUT/f'{model}_{seed}.json',dict(model=model,seed=None if model==OFF else seed,checkpoint_sha=digest,gpu=gpu,
            differing_units=sum(a[k]!=b[k] for k in a),rows=rows))

def summarize():
    records=[]
    for path in sorted(OUT.glob('*.json')):
        job=json.loads(path.read_text())
        groups=defaultdict(list)
        for r in job['rows']:groups[r['split'],r['group']].append(r)
        assert len(groups)==19
        for (s,g),rs in groups.items():
            for metric in METRICS:
                vals=[r[metric] for r in rs if r[metric] is not None]
                if vals:records.append((job['model'],job['seed'],s,g,metric,st.mean(vals)))
    aggregate=defaultdict(list)
    for m,seed,s,g,k,v in records:aggregate[m,seed,s,k].append(v)
    records.extend((m,seed,s,'ALL',k,st.mean(vs)) for (m,seed,s,k),vs in aggregate.items())
    assert len(aggregate)==80
    pooled=defaultdict(dict)
    for m,seed,s,g,k,v in records:pooled[m,s,g,k][seed]=v
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['model','split','group','metric',*SEEDS,'n','mean','sample_sd'])
        for key,d in sorted(pooled.items()):
            w.writerow([*key,*[d.get(i) for i in SEEDS],len(d),st.mean(d.values()),st.stdev(d.values()) if len(d)>1 else None])

def launch():
    OUT.mkdir(exist_ok=False)
    def worker(gpu,jobs):
        uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        for name,seed in jobs:
            py=(BASE.parent/'mv4_official_mitonet_20261014/venv/bin/python' if name==OFF else BASE.parent.parent/'envs/meddinov3/bin/python')
            with (OUT/f'{name}_{seed}.log').open('w') as f:
                subprocess.run([py,'-u',__file__,name,str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(3) as pool:
        tasks=[pool.submit(worker,i+1,([(OFF,SEEDS[0])] if i==0 else [])+[(m,s) for s in SEEDS]) for i,m in enumerate(MODELS)]
        for t in tasks:t.result()
    summarize()

if __name__=='__main__':
    if len(sys.argv)==1:launch()
    else:evaluate(sys.argv[1],int(sys.argv[2]))
