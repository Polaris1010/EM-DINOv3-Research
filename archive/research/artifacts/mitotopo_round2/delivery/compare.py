import os,sys,json,time,signal,gc,hashlib
from pathlib import Path
import numpy as np,torch,zarr
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
import runtime as rt
signal.signal(signal.SIGINT,signal.default_int_handler)
BASE=Path(__file__).resolve().parent;OUT=Path(os.environ.get("RUN_OUTPUT",BASE));OUT.mkdir(exist_ok=True,parents=True)
rt.OUT=BASE

def instance_score(t,p):
    _,ti=np.unique(t.ravel(),return_inverse=True);_,pi=np.unique(p.ravel(),return_inverse=True)
    # Background is explicitly mapped even for patches without a zero pixel.
    tv=np.unique(t);pv=np.unique(p)
    tab=np.bincount(ti*len(pv)+pi,minlength=len(tv)*len(pv)).reshape(len(tv),len(pv))
    a=tab.sum(1);b=tab.sum(0);sel_t=tv!=0;sel_p=pv!=0
    inter=tab[np.ix_(sel_t,sel_p)];union=a[sel_t,None]+b[None,sel_p]-inter
    iou=inter/np.maximum(union,1);nt,npred=iou.shape
    if nt and npred:
        r,c=linear_sum_assignment(-(iou>.5).astype(float)-iou/(2*max(nt,npred)))
        matched=iou[r,c];matched=matched[matched>.5]
    else:matched=np.array([])
    tp=len(matched);den=tp+.5*(nt+npred-2*tp)
    return dict(instance_tp=tp,instance_fp=npred-tp,instance_fn=nt-tp,iou_sum=float(matched.sum()),pq=float(matched.sum()/den) if den else 1.,f1_50=tp/den if den else 1.)

def tests():
    t=np.array([[0,1,1,0,2,2],[0,1,1,0,2,2]])
    assert instance_score(t,t)['pq']==1
    assert instance_score(t,np.zeros_like(t))['pq']==0
    merge=(t>0).astype(int);assert instance_score(t,merge)['pq']==0
    assert instance_score(np.zeros_like(t),np.zeros_like(t))['pq']==1

def samples():
    ds=rt.PatchDataset('val',20,192,False)
    for i in range(len(ds)):
        s=ds[i];vid=s['volume_id'];z,y,x=map(int,s['zyx'])
        cat=ds.cat[vid];g=zarr.open_group(str(rt.ROOT/'MitoVerse_data'/cat['zarr']),mode='r')
        raw=np.asarray(g['img'][z,y:y+192,x:x+192]);lab=np.asarray(g['mito'][z,y:y+192,x:x+192])
        yield i,vid,[z,y,x],raw,lab

def save_rows(name,rows):
    c=np.sum([[r[k] for k in ['tp','fp','fn','tn']] for r in rows],0)
    result=rt.metrics(c);result['mean_patch_iou']=float(np.mean([r['iou'] for r in rows]));result['mean_patch_pq']=float(np.mean([r['pq'] for r in rows]));result['rows']=rows
    (OUT/(name+'.json')).write_text(json.dumps(result,indent=2));print(name,json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

def main():
    tests();rt.seed();device='cpu' if '--cpu' in sys.argv else 'cuda'
    mitopath=BASE/'official/MitoNet_v1.pth'
    net=torch.jit.load(str(mitopath),map_location='cpu').eval()
    print('MITONET_SCHEMA',net.forward.schema,flush=True)
    with torch.no_grad():
        result=net(torch.zeros(1,1,64,64))
        print('MITONET_CPU',{k:list(v.shape) for k,v in result.items()},flush=True)
    if device=='cpu':return
    sys.path.insert(0,str(BASE/'official'));from postprocess import get_panoptic_segmentation
    net=net.to(device);rows=[]
    for i,vid,zyx,raw,lab in samples():
        x=torch.from_numpy((raw.astype(np.float32)/255.-.57571)/.12765)[None,None].to(device)
        with torch.no_grad():
            r=net(x);logits=r['sem_logits'];prob=logits.sigmoid() if logits.shape[1]==1 else logits.softmax(1)[:,1:2]
            sem=(prob>=.5).long()
            pred,_=get_panoptic_segmentation(sem,r['ctr_hmp'],r['offsets'],[1],1000,64,0,.1,7)
        p=pred.squeeze().cpu().numpy();row=dict(index=i,volume=vid,zyx=zyx,**rt.metrics(rt.confusion(p>0,lab>0)),**instance_score(lab,p));rows.append(row)
        np.savez_compressed(OUT/f'mitonet_{i:02}.npz',probability=prob.squeeze().cpu().numpy(),instance=p,truth=lab,raw=raw)
        save_rows('mitonet',rows)
    del net;gc.collect();torch.cuda.empty_cache()
    paths={'current_epoch1':rt.DOWN,'current_epoch266':rt.ROOT/'Med_DINOv3/nnUNet_results/Dataset131_CMCT/meddinov3_base_primus_multiscale_Trainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth'}
    for name,path in paths.items():
        net=rt.build_model();ck=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        sd={k.removeprefix('_orig_mod.').removeprefix('module.'):v for k,v in ck['network_weights'].items()}
        net.load_state_dict(sd,strict=True)
        (OUT/(name+'_provenance.json')).write_text(json.dumps(dict(path=str(path),epoch=ck.get('current_epoch'),best_ema=str(ck.get('_best_ema')),init_args=ck.get('init_args'),mirroring=ck.get('inference_allowed_mirroring_axes'),strict_match=True),default=str,indent=2))
        del ck,sd;net=net.eval().to(device);rows=[]
        for i,vid,zyx,raw,lab in samples():
            x=torch.from_numpy((raw.astype(np.float32)/255.-.485)/.229)[None,None].to(device)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):prob=net(x).float().softmax(1)[0,1]
            p=ndimage.label(prob.cpu().numpy()>=.5)[0]
            rows.append(dict(index=i,volume=vid,zyx=zyx,**rt.metrics(rt.confusion(p>0,lab>0)),**instance_score(lab,p)))
            np.savez_compressed(OUT/f'{name}_{i:02}.npz',probability=prob.cpu().numpy(),instance=p)
            save_rows(name,rows)
        del net;gc.collect();torch.cuda.empty_cache()
    (OUT/'complete.json').write_text(json.dumps(dict(status='completed',samples=20,crop=192,protocol='nonofficial 2D patch diagnostic; MitoNet official normalization and postprocess; current model CC instances; no TTA or scale tuning; not published 3D benchmark')))
if __name__=='__main__':main()
