import os,sys,json,time,signal,argparse,gc,pickle,itertools
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from audit_cached import BASE,ROOT,DATASETS,unpack,metrics,confusion
ROUND2=BASE.parent/'mitotopo_20260908_round2'
NNROOT=ROOT/'MedDINOv3/nnUNet'
sys.path.insert(0,str(NNROOT));sys.path.insert(0,str(NNROOT/'nnunetv2/training/nnUNetTrainer/dinov3'));sys.path.insert(0,str(ROUND2/'official'))
signal.signal(signal.SIGINT,signal.default_int_handler)
def build(ds,swap=None):
    from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.vision_transformer import vit_large
    from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import Primus_Multiscale
    path=ROOT/'nnUNet_results'/ds/'meddinov3_base_primus_multiscale_Trainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth'
    with torch.device('meta'):
        encoder=vit_large(patch_size=16,drop_path_rate=.3,layerscale_init=1e-5,n_storage_tokens=0,qkv_bias=True,mask_k_bias=False)
        net=Primus_Multiscale(embed_dim=1024,patch_embed_size=16,num_classes=2,dino_encoder=encoder,interaction_indices=[5,11,17,23])
    ck=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    sd={k.removeprefix('_orig_mod.').removeprefix('module.'):v for k,v in ck['network_weights'].items()}
    net.load_state_dict(sd,strict=True,assign=True)
    if swap:
        pre=torch.load(swap,map_location='cpu',mmap=True,weights_only=False)
        sd={k.removeprefix('backbone.'):v for k,v in pre['teacher'].items() if k.startswith('backbone.')}
        net.dino_encoder.load_state_dict(sd,strict=True,assign=True)
    net.eval().cuda();del ck,sd;gc.collect()
    return net
def starts(n):
    count=int(np.ceil((n-224)/112))+1
    return [int(round(i*(n-224)/(count-1))) for i in range(count)] if count>1 else [0]
@torch.inference_mode()
def sliding(net,arr,tta=True):
    h,w=arr.shape;ph=max(0,224-h);pw=max(0,224-w)
    x=torch.from_numpy(np.array(arr,dtype=np.float32,copy=True))[None,None].cuda()
    x=F.pad(x,(pw//2,pw-pw//2,ph//2,ph-ph//2));H,W=x.shape[-2:]
    g=np.zeros((224,224),np.float32);g[112,112]=1;g=gaussian_filter(g,28);g/=g.max();g[g==0]=g[g>0].min();weight=torch.as_tensor(g,device='cuda')
    result=torch.zeros((2,H,W),device='cuda');norm=torch.zeros((H,W),device='cuda')
    flips=[(),(2,),(3,),(2,3)] if tta else [()]
    for y,z in itertools.product(starts(H),starts(W)):
        patch=x[:,:,y:y+224,z:z+224];p=torch.zeros((1,2,224,224),device='cuda')
        for axes in flips:
            with torch.autocast('cuda',dtype=torch.float16):q=net(torch.flip(patch,axes) if axes else patch)
            p+=(torch.flip(q,axes) if axes else q).float()/len(flips)
        assert torch.isfinite(p).all()
        result[:,y:y+224,z:z+224]+=p[0]*weight;norm[y:y+224,z:z+224]+=weight
    return (result/norm)[...,ph//2:ph//2+h,pw//2:pw//2+w].cpu().numpy()
def restore_slice(logits,shape):
    if tuple(logits.shape[-2:])==tuple(shape):return logits.argmax(0)
    from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg
    resized=resample_data_or_seg(logits[:,None],(1,*shape),False,order=1,do_separate_z=False)
    return resized[:,0].argmax(0)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['mitonet','swap','native_em'],required=True);ap.add_argument('--stride',type=int,default=1);a=ap.parse_args()
    out=Path(os.environ['RUN_OUTPUT']);torch.set_num_threads(2)
    assert (BASE/'cache_ready.json').exists()
    result={'mode':a.mode,'slice_stride':a.stride,'status':'running','datasets':{},'protocol':'original-grid fold_0 binary Dice; MitoNet native slice inference, DINO 224 Gaussian half-stride 4-flip inference','interpretation':'encoder swap is compatibility diagnostic, not matched downstream retraining or causal Gram effect'}
    mito=None
    if a.mode=='mitonet':
        from postprocess import factor_pad,get_panoptic_segmentation
        mito=torch.jit.load(str(ROUND2/'official/MitoNet_v1.pth'),map_location='cpu').eval().cuda()
    for ds in DATASETS:
        split=json.loads((ROOT/'nnUNet_preprocessed'/ds/'splits_final.json').read_text())[0]
        net=None if mito else build(ds,ROUND2/'gram_full_01/teacher_latest.pth' if a.mode=='swap' else None)
        d={'cases':{}};result['datasets'][ds]=d
        for case in split['val']:
            gt=unpack(ROOT/'nnUNet_raw'/ds/'labelsTr'/(case+'.nii.gz'),case+'_gt');shape=(gt.shape[2],gt.shape[1],gt.shape[0])
            if mito:raw=unpack(ROOT/'nnUNet_raw'/ds/'imagesTr'/(case+'_0000.nii.gz'),case+'_raw')
            else:
                pre=ROOT/'nnUNet_preprocessed'/ds/'nnUNetPlans_2d';raw=np.load(pre/(case+'.npy'),mmap_mode='r')
                meta=pickle.load(open(pre/(case+'.pkl'),'rb'))
                assert tuple(meta['shape_before_cropping'])==shape
                assert meta['bbox_used_for_cropping']==[[0,n] for n in shape], 'Nontrivial crop requires explicit volume restoration'
                assert raw.shape[1]==shape[0]
            pred=np.lib.format.open_memmap(out/(case+'.npy'),mode='w+',dtype=np.uint8,shape=shape);c=np.zeros(4,dtype=np.int64);semantic=np.zeros(4,dtype=np.int64);t0=time.time()
            for z in range(0,shape[0],a.stride):
                true=np.asarray(gt.dataobj[:,:,z]).T>0
                if mito:
                    image=np.array(raw.dataobj[:,:,z],dtype=np.float32).T.copy();assert image.min()>=0 and image.max()<=255
                    x=torch.from_numpy((image/255.-.57571)/.12765)[None,None].cuda()
                    with torch.inference_mode():
                        q=mito(factor_pad(x));sem=(q['sem_logits'].sigmoid()>.5).long()
                        p,_=get_panoptic_segmentation(sem,q['ctr_hmp'],q['offsets'],[1],1000,64,0,.1,7)
                    assert int(np.prod(p.shape[:-2]))==1
                    yp=p.reshape(-1,*p.shape[-2:])[0,:shape[1],:shape[2]].cpu().numpy()>0
                    semantic+=confusion(sem[0,0,:shape[1],:shape[2]].cpu().numpy()>0,true)
                else:yp=restore_slice(sliding(net,raw[0,z]),shape[1:])>0
                assert yp.shape==true.shape
                pred[z]=yp;c+=confusion(yp,true)
                if z%20==0:
                    print(ds,case,z,metrics(c)['dice'],flush=True)
                    (out/'progress.json').write_text(json.dumps({'dataset':ds,'case':case,'z':z,'depth':shape[0],'elapsed':time.time()-t0}))
            pred.flush();d['cases'][case]={**metrics(c),'elapsed_seconds':time.time()-t0,'evaluated_slices':len(range(0,shape[0],a.stride))}
            if mito:d['cases'][case]['semantic_head_only']=metrics(semantic)
            (out/'metrics.json').write_text(json.dumps(result,indent=2))
        d['macro_dice']=float(np.mean([v['dice'] for v in d['cases'].values()]));d['micro']=metrics(np.sum([[v[k] for k in ('tp','fp','fn','tn')] for v in d['cases'].values()],axis=0))
        del net;gc.collect();torch.cuda.empty_cache()
    result['status']='complete';(out/'metrics.json').write_text(json.dumps(result,indent=2));(out/'result.json').write_text(json.dumps({'status':'complete'}))
if __name__=='__main__':main()
