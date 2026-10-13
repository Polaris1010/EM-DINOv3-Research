"""Matched frozen-backbone Primus probes, trained only on original fold-0 training ROIs."""
import os,json,time,argparse,pickle,random,gc,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from evaluate import BASE,ROOT,ROUND2,DATASETS,sliding,restore_slice,unpack,metrics,confusion
EM=Path('/opt/data/3_117/tanghr/DINOv3/med_checkpoints_continual_v2/eval/training_124999/teacher_checkpoint.pth')
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.set_num_threads(2)
def model(checkpoint,s):
    from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.vision_transformer import vit_large
    from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import Primus_Multiscale
    seed(s)
    with torch.device('meta'):enc=vit_large(patch_size=16,drop_path_rate=.3,layerscale_init=1e-5,n_storage_tokens=0,qkv_bias=True,mask_k_bias=False)
    ck=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    sd={k.removeprefix('backbone.'):v for k,v in ck['teacher'].items() if k.startswith('backbone.')}
    enc.load_state_dict(sd,strict=True,assign=True);enc.requires_grad_(False)
    # Head randomness is independent of checkpoint contents and encoder construction.
    seed(s);net=Primus_Multiscale(embed_dim=1024,patch_embed_size=16,num_classes=2,dino_encoder=enc,interaction_indices=[5,11,17,23])
    net.cuda();del ck,sd;gc.collect();return net
class Samples:
    def __init__(self,ds,s):
        self.p=ROOT/'nnUNet_preprocessed'/ds/'nnUNetPlans_2d';self.split=json.loads((self.p.parent/'splits_final.json').read_text())[0];self.s=s
        assert not set(self.split['train'])&set(self.split['val']);self.handles={};self.positive_cases=[]
        for case in self.split['train']:
            meta=pickle.load(open(self.p/(case+'.pkl'),'rb'))
            if any(len(v)>0 and (k==1 or isinstance(k,tuple) and 1 in k and 0 not in k) for k,v in meta['class_locations'].items()):self.positive_cases.append(case)
        assert self.positive_cases
    def sample(self,i,device='cuda'):
        r=np.random.default_rng(self.s+i);force_foreground=r.random()<.5;pool=self.positive_cases if force_foreground else self.split['train'];case=pool[int(r.integers(len(pool)))]
        if case not in self.handles:
            if len(self.handles)>=2:self.handles.clear()
            self.handles[case]=(np.load(self.p/(case+'.npy'),mmap_mode='r'),np.load(self.p/(case+'_seg.npy'),mmap_mode='r'),pickle.load(open(self.p/(case+'.pkl'),'rb')))
        raw,seg,meta=self.handles[case];_,D,H,W=raw.shape
        z=int(r.integers(D));cy=int(r.integers(H));cx=int(r.integers(W))
        if force_foreground:
            locs=[v for k,v in meta['class_locations'].items() if len(v)>0 and (k==1 or isinstance(k,tuple) and 1 in k and 0 not in k)]
            if locs:
                loc=locs[int(r.integers(len(locs)))];z,cy,cx=map(int,loc[int(r.integers(len(loc)))][-3:])
        y=min(max(0,cy-112),max(0,H-224));x=min(max(0,cx-112),max(0,W-224))
        a=np.array(raw[0,z,y:y+224,x:x+224],copy=True);b=np.array(seg[0,z,y:y+224,x:x+224],dtype=np.int64,copy=True)
        pad=((0,224-a.shape[0]),(0,224-a.shape[1]));a=np.pad(a,pad);b=np.pad(b,pad,constant_values=-1)
        if r.random()<.5:a=np.flip(a,0);b=np.flip(b,0)
        if r.random()<.5:a=np.flip(a,1);b=np.flip(b,1)
        return torch.from_numpy(a.copy())[None,None].to(device),torch.from_numpy(b.copy())[None].to(device),case
def loss(logits,y):
    valid=y>=0;t=(y==1).float();p=logits.float().softmax(1)[:,1]
    ce=F.cross_entropy(logits.float(),y,ignore_index=-1)
    dice=1-(2*(p*t*valid).sum()+1e-5)/((p*valid).sum()+(t*valid).sum()+1e-5)
    assert valid.any();return ce+dice
def validation(net,ds,out,stride,save=False):
    net.eval();split=json.loads((ROOT/'nnUNet_preprocessed'/ds/'splits_final.json').read_text())[0];cases={}
    for case in split['val']:
        pp=ROOT/'nnUNet_preprocessed'/ds/'nnUNetPlans_2d';raw=np.load(pp/(case+'.npy'),mmap_mode='r');meta=pickle.load(open(pp/(case+'.pkl'),'rb'))
        gt=unpack(ROOT/'nnUNet_raw'/ds/'labelsTr'/(case+'.nii.gz'),case+'_gt');shape=(gt.shape[2],gt.shape[1],gt.shape[0]);assert meta['bbox_used_for_cropping']==[[0,n] for n in shape] and raw.shape[1]==shape[0]
        prediction=np.lib.format.open_memmap(out/(case+'.npy'),mode='w+',dtype=np.uint8,shape=shape) if save else None
        c=np.zeros(4,dtype=np.int64)
        for z in range(0,shape[0],stride):
            p=restore_slice(sliding(net,raw[0,z],tta=save),shape[1:])>0;t=np.asarray(gt.dataobj[:,:,z]).T>0;c+=confusion(p,t)
            if prediction is not None:prediction[z]=p
        if prediction is not None:prediction.flush()
        cases[case]=metrics(c);print('VAL',ds,case,stride,cases[case]['dice'],flush=True)
    return {'cases':cases,'macro_dice':float(np.mean([v['dice'] for v in cases.values()])),'micro':metrics(np.sum([[v[k] for k in ('tp','fp','fn','tn')] for v in cases.values()],axis=0)),'slice_stride':stride,'tta':save}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--encoder',required=True);ap.add_argument('--eval-from',type=Path);ap.add_argument('--steps',type=int,default=3000);ap.add_argument('--seed',type=int,default=20260907);a=ap.parse_args()
    out=Path(os.environ['RUN_OUTPUT']);encoder=EM if a.encoder=='em' else ROUND2/'gram_full_01/teacher_latest.pth' if a.encoder=='gram200k' else Path(a.encoder)
    result={'evaluation_head_source':str(a.eval_from) if a.eval_from else None,'status':'running','encoder':str(encoder),'seed':a.seed,'steps':a.steps,'protocol':'frozen ViT-L BF16 features, FP32 decoder training (no loss scaling), identical fresh Primus decoder and gates; batch 1; CE+Dice, ignore preprocessed -1 during training only; 50% sample from foreground-positive training cases and coordinates; remaining 50% uniformly from all training cases; fold-0 train only; select head by stride-20 validation Dice; final full-volume 4-flip validation','datasets':{}}
    (out/'protocol.json').write_text(json.dumps(result,indent=2))
    for ds in DATASETS:
        d=out/ds;d.mkdir(exist_ok=True);net=model(encoder,a.seed);samples=Samples(ds,a.seed)
        if not a.eval_from:
            params=[p for p in net.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.05,betas=(.9,.98));best=-1.;curve=[]
            t0=time.time()
            for step in range(a.steps):
                net.train();net.dino_encoder.eval();x,y,case=samples.sample(step)
                lr=1e-4*min(1.,(step+1)/50)*(.1+.9*.5*(1+np.cos(np.pi*step/a.steps)))
                for g in opt.param_groups:g['lr']=lr
                opt.zero_grad(set_to_none=True)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    features=net.dino_encoder.get_intermediate_layers(x.repeat(1,3,1,1),n=net.interaction_indices,reshape=True)
                logits=net.up_projection(torch.cat([gate(f.float()) for gate,f in zip(net.last_gates,features)],dim=1))
                v=loss(logits,y);assert torch.isfinite(v)
                v.backward();torch.nn.utils.clip_grad_norm_(params,12,error_if_nonfinite=True);opt.step()
                if (step+1)%50==0:
                    row={'dataset':ds,'step':step+1,'loss':float(v.detach()),'lr':lr,'case':case,'seconds':time.time()-t0};print(json.dumps(row),flush=True)
                    (out/'progress.json').write_text(json.dumps(row))
                if (step+1)%1000==0 or step+1==a.steps:
                    val=validation(net,ds,d,20);curve.append({'step':step+1,**val})
                    if val['macro_dice']>best:
                        best=val['macro_dice'];torch.save({'head':{k:v.detach().cpu() for k,v in net.state_dict().items() if not k.startswith('dino_encoder.')},'step':step+1,'encoder':str(encoder),'seed':a.seed},d/'head_best.pth')
                    (d/'curve.json').write_text(json.dumps(curve,indent=2))
        head_dir=a.eval_from/ds if a.eval_from else d
        ck=torch.load(head_dir/'head_best.pth',map_location='cpu',weights_only=False);missing,unexpected=net.load_state_dict(ck['head'],strict=False)
        assert not unexpected and all(k.startswith('dino_encoder.') for k in missing)
        assert ck['encoder']==str(encoder) and ck['seed']==a.seed
        if a.eval_from:(d/'curve.json').write_text((head_dir/'curve.json').read_text())
        result['datasets'][ds]={'selected_step':ck['step'],**validation(net,ds,d,1,True)}
        (out/'metrics.json').write_text(json.dumps(result,indent=2));del net,ck,samples
        if not a.eval_from:del opt,params
        gc.collect();torch.cuda.empty_cache()
    result['status']='complete';(out/'metrics.json').write_text(json.dumps(result,indent=2));(out/'result.json').write_text(json.dumps({'status':'complete'}))
if __name__=='__main__':main()
