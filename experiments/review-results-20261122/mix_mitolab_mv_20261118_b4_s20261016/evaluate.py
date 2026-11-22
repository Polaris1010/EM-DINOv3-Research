from pathlib import Path
import sys,os
import numpy as np
import torch
from mixture import X,B,ROOT,C,M,MB,SEED,read,sha,atomic,arrays
from fit import primus

def metrics(c):
    tp,fp,fn,tn=map(int,c);union=tp+fp+fn
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,dice=2*tp/(union+tp) if union else 1.,
                iou=tp/union if union else 1.,precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None)
@torch.inference_mode()
def run(name):
    out=X/name;trained=read(out/'trained.json');ck=out/'model_40000.pt'
    assert sha(ck)==trained['checkpoint_sha']
    sys.path.insert(0,str(MB/'code'))
    from common import Planes,config,gpu_guard
    gpu_guard();torch.set_num_threads(2);torch.manual_seed(SEED);np.random.seed(SEED)
    if name=='MitoNet-Mix':
        from infer import mitonet
        spec,=[r.copy() for r in config()['runs'] if r['model']=='Official-MitoNet-v1']
        spec['checkpoint_path']=str(ck)
        engine=mitonet(spec)
        predict=lambda a: np.asarray(engine(a))>0
    else:
        net,p,_=primus(name)
        missing,extra=net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=False)
        assert not extra and all(k.startswith('dino_encoder.') for k in missing)
        net.eval()
        predict=lambda a:p.sliding(net,a,tile_batch_size=16)
    def counts(a,b,v=None):
        pred=torch.as_tensor(predict(a),device='cuda')
        gt=torch.as_tensor(np.asarray(b)>0,device='cuda')
        assert gt.shape==pred.shape
        if v is not None:
            valid=torch.as_tensor(np.asarray(v)>0,device='cuda');gt=gt[valid];pred=pred[valid]
        tp,pos,k=torch.stack([(pred & gt).sum(),gt.sum(),pred.sum()]).tolist()
        return np.array([tp,k-tp,pos-tp,pred.numel()-pos-k+tp],np.int64)
    def block(tag,make):
        d=out/'scores';d.mkdir(exist_ok=True);path=d/(tag+'.json')
        if not path.exists():atomic(path,make())
        return read(path)
    records=[]
    cem=read(C/'manifest.json')
    for g in cem['validation']:
        def get_cem():
            rows=[]
            for name,_,_ in cem['groups'][g]:
                a,b=np.load(Path('/opt/cem_mitolab_20261118')/g/(name+'.npy'))
                rows.append(dict(split='cem_validation',group=g,name=name,z=0,**metrics(counts(a,b))))
            return rows
        records+=block('cem_'+g,get_cem)
    for row in read(MB/'MB1_benchmark_manifest.json')['entries']:
        def get_mb():
            a=Planes(row['image_path'],row['shape_zyx']);b=Planes(row['label_path'],row['shape_zyx'])
            rs=[dict(split='mb1',group=row['dataset'],name=row['id'],z=z,**metrics(counts(a[z],b[z]))) for z in range(row['shape_zyx'][0])]
            a.close();b.close();return rs
        records+=block('mb_'+row['id'],get_mb)
    for row in read(M/'MV-Generalist.json')['entries']:
        if row['supervised_role']!='dev':continue
        def get_mv():
            a,b,v,j=arrays(row['id']);total=np.zeros(4,np.int64)
            for z in j['valid_z']:total+=counts(a[z],b[z],v[z] if v is not None else None)
            return [dict(split='mv3',group=row['source_id'],name=row['id'],z=0,**metrics(total))]
        records+=block('mv_'+row['id'],get_mv)
    assert len(records)==1424+2595+38
    atomic(out/'evaluation.json',records)
if __name__=='__main__':run(sys.argv[1])
