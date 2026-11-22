"""Same fixed 2D transfer runs, scored on instance IDs."""
import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import runpy
from pathlib import Path
import json
import os
import statistics
import subprocess
import sys
import numpy as np
from zeroshot_repeat import BASE,MODELS,OFF,SEEDS,source

OUT=BASE/'instance_2d_repeat_20261120'
KEYS=('IoU','F1_50','F1_75','AP_50','AP_75','PQ')

@lru_cache(None)
def official_metrics():
    root=BASE.parent/'mitonet_paper_20261119/empanada-main/empanada/evaluation'
    instance=runpy.run_path(str(root/'instance_metrics.py'))
    return instance['ap'],instance['f1'],runpy.run_path(str(root/'panoptic_metrics.py'))['panoptic_quality']

def score(gt,pred):
    import torch
    from scipy.optimize import linear_sum_assignment
    ap,f1,panoptic_quality=official_metrics()
    def dense(a):
        ids,inv,size=torch.unique(torch.as_tensor(np.ascontiguousarray(a),device='cuda',dtype=torch.long),
                                  return_inverse=True,return_counts=True)
        if ids[0]!=0:return inv+1,size
        return inv,size[1:]
    g,ga=dense(gt)
    p,pa=dense(pred)
    n,m=len(ga),len(pa)
    overlap=(g>0)&(p>0)
    union=int(((g>0)|(p>0)).sum())
    inter=int(overlap.sum())
    hits=np.empty(0)
    if inter:
        counts=torch.bincount((g[overlap]-1)*m+(p[overlap]-1),minlength=n*m).reshape(n,m)
        matrix=(counts.double()/(ga[:,None]+pa[None,:]-counts)).cpu().numpy()
        ri,ci=linear_sum_assignment(matrix,maximize=True)
        hits=matrix[ri,ci]
        hits=hits[hits>=.5]
    args=dict(gt_matched=range(len(hits)),pred_matched=range(len(hits)),gt_unmatched=range(n-len(hits)),
              pred_unmatched=range(m-len(hits)),matched_ious=hits)
    return dict(IoU=inter/union if union else 1.,F1_50=f1(**args,iou_thr=.5),F1_75=f1(**args,iou_thr=.75),
                AP_50=ap(**args,iou_thr=.5),AP_75=ap(**args,iou_thr=.75),PQ=panoptic_quality(**args),
                n_gt=n,n_pred=m)

def evaluate(model,seed):
    import torch
    from scipy.ndimage import label
    sys.path.insert(0,str(source(seed)))
    from mixture import MB,M,read,sha,arrays,instance,atomic
    sys.path.insert(0,str(MB/'code'))
    from common import Planes,gpu_guard,config
    gpu_guard()
    torch.set_num_threads(2)
    if model==OFF:
        from infer import mitonet
        spec=next(r for r in config()['runs'] if r['model']==OFF)
        assert sha(Path(spec['checkpoint_path']))==spec['checkpoint_sha256']
        predict=mitonet(spec)
    else:
        from fit import primus
        d=source(seed)/'batch4/primus_cem'/model
        receipt=read(d/'trained.json')
        ck=d/'model_40000.pt'
        assert sha(ck)==receipt['checkpoint_sha'] and receipt['seed']==seed
        net,p,enc=primus(model)
        assert enc==receipt['init_encoder_sha']
        missing,extra=net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=False)
        assert not extra and all(k.startswith('dino_encoder.') for k in missing)
        net.eval()
        predict=lambda a:p.sliding(net,a,tile_batch_size=16)
    d=OUT/f'{model}_{seed}'
    d.mkdir(exist_ok=True)
    def plane(a,gt,valid=None):
        pred=np.asarray(predict(a))
        assert pred.shape==gt.shape
        if valid is not None:
            pred=np.where(valid,pred,0)
            gt=np.where(valid,gt,0)
        if model!=OFF:
            pred=label(pred,structure=np.ones((3,3),int))[0]
        return score(gt,pred)
    with torch.inference_mode():
        for r in read(MB/'MB1_benchmark_manifest.json')['entries']:
            dest=d/('mb1_'+r['id']+'.json')
            if dest.exists():continue
            a,b=[Planes(r[k],r['shape_zyx']) for k in ('image_path','label_path')]
            rs=[plane(a[z],b[z]) for z in range(r['shape_zyx'][0])]
            a.close();b.close()
            atomic(dest,dict(split='mb1',group=r['dataset'],id=r['id'],planes=rs))
        for r in read(M/'MV-Generalist.json')['entries']:
            if r['supervised_role']!='dev':continue
            dest=d/('mv3_'+r['id']+'.json')
            if dest.exists():continue
            a,b,v,j=arrays(r['id'])
            oz,oy,ox=(x[0] for x in r['region_zyx'])
            ids=instance(r['image_path'])
            rs=[]
            for z in j['valid_z']:
                h,w=a[z].shape
                gt=np.asarray(ids[z+oz,oy:oy+h,ox:ox+w])
                valid=v[z]>0 if v is not None else None
                assert np.all(((gt>0)==(b[z]>0)) if valid is None else ((gt>0)==(b[z]>0))[valid])
                rs.append(plane(a[z],gt,valid))
            atomic(dest,dict(split='mv3',group=r['source_id'],id=r['id'],planes=rs))

def summarize():
    gs=defaultdict(list)
    for m in (*MODELS,OFF):
        for s in (SEEDS[:1] if m==OFF else SEEDS):
            data=[json.loads(p.read_text()) for p in (OUT/f'{m}_{s}').glob('*.json')]
            assert sum(len(r['planes']) for r in data if r['split']=='mb1')==2595
            assert sum(r['split']=='mv3' for r in data)==38
            for r in data:
                for k in KEYS:
                    v=[p[k] for p in r['planes']]
                    gs[m,s,r['split'],r['group'],k].extend(v if r['split']=='mb1' else [statistics.mean(v)])
    grouped=defaultdict(dict)
    total=defaultdict(list)
    for (m,s,sp,g,k),v in gs.items():
        value=statistics.mean(v)
        grouped[m,sp,g,k][s]=value
        total[m,s,sp,k].append(value)
    for (m,s,sp,k),v in total.items():grouped[m,sp,'ALL',k][s]=statistics.mean(v)
    with (OUT/'instance_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['model','split','source','metric','fixed_release',*SEEDS,'mean','sample_sd'])
        for key,vs in sorted(grouped.items()):
            vals=list(vs.values())
            w.writerow([*key,vals[0] if key[0]==OFF else '',*[vs.get(s) if key[0]!=OFF else '' for s in SEEDS],
                        statistics.mean(vals),statistics.stdev(vals) if len(vals)>1 else ''])

def worker(card,jobs):
    uuid=subprocess.check_output(['nvidia-smi','-i',str(card),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    for m,s in jobs:
        exe=BASE.parent/('mv4_official_mitonet_20261014/venv/bin/python' if m==OFF else '../envs/meddinov3/bin/python')
        with (OUT/f'{m}_{s}.log').open('a') as f:
            subprocess.run([exe,'-u',__file__,m,str(s)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':
    if len(sys.argv)==3:evaluate(sys.argv[1],int(sys.argv[2]))
    else:
        OUT.mkdir(exist_ok=True)
        with ThreadPoolExecutor(3) as pool:
            tasks=[pool.submit(worker,i+1,([(OFF,SEEDS[0])] if i==0 else [])+[(m,s) for s in SEEDS]) for i,m in enumerate(MODELS)]
            for t in tasks:t.result()
        summarize()
