"""Frozen official EMA features and train-only 1x1 logistic probes; no Primus/SSL training."""
import argparse
import gc
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from common import R, read, atomic, sha, npy


def run(model_id):
    started=time.time();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.28)
    protocol=read(R/'protocol.json');items=read(R/'samples.json');spec=next(v for v in read(R/'verified_models.json') if v['id']==model_id)
    path=Path(spec['checkpoint_path']);assert sha(path)==spec['sha256']
    out=R/'models'/model_id;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'probe_complete.json').exists(),'Do not retrain completed probes'
    sys.path.insert(0,spec['dinov3_root'])
    from omegaconf import OmegaConf
    from dinov3.models import build_model_from_cfg
    encoder,dim=build_model_from_cfg(OmegaConf.load(spec['resolved_config_path']),only_teacher=True)
    assert dim==1024 and encoder.n_storage_tokens==4 and encoder.patch_size==16
    saved=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
    if spec['kind']=='official_natural':weights=saved
    else:
        assert saved['step']==(30000 if model_id=='D2-30k' else 100000)
        weights=saved['backbone']
    assert len(weights)==368
    encoder.load_state_dict(weights,strict=True,assign=True);encoder.requires_grad_(False).eval().cuda()
    assert all(not p.requires_grad for p in encoder.parameters())
    del saved,weights;gc.collect()
    n=len(items);layers=protocol['layers'];images=npy(R/'images.npy')
    features={layer:np.lib.format.open_memmap(out/f'layer{layer}.npy',mode='w+',dtype=np.float16,shape=(n,256,1024)) for layer in layers}
    cls=np.lib.format.open_memmap(out/'CLS.npy',mode='w+',dtype=np.float16,shape=(n,4,1024))
    registers=np.lib.format.open_memmap(out/'registers.npy',mode='w+',dtype=np.float16,shape=(n,4,4,1024))
    with torch.inference_mode():
        for first in range(0,n,8):
            x=torch.tensor(np.array(images[first:first+8],copy=True),device='cuda',dtype=torch.float32)
            x=((x/255-.5)/.25)[:,None].repeat(1,3,1,1)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                outputs=encoder.get_intermediate_layers(x,n=layers,reshape=False,return_class_token=True,return_extra_tokens=True,norm=True)
            for li,(patches,c,reg) in enumerate(outputs):
                assert tuple(patches.shape[1:])==(256,1024) and tuple(reg.shape[1:])==(4,1024)
                assert torch.isfinite(patches).all()
                features[layers[li]][first:first+len(x)]=patches.float().cpu().numpy()
                cls[first:first+len(x),li]=c.float().cpu().numpy();registers[first:first+len(x),li]=reg.float().cpu().numpy()
            atomic(out/'progress.json',{'stage':'frozen_features','completed_crops':min(n,first+8),'total':n,'seconds':time.time()-started})
    for a in features.values():a.flush()
    cls.flush();registers.flush();del encoder;gc.collect();torch.cuda.empty_cache()
    atomic(out/'feature_receipt.json',{'status':'complete','encoder_sha256':spec['sha256'],'samples_sha256':sha(R/'samples.json'),
        'layers':layers,'patch_shape_per_layer':[n,256,1024],'CLS_shape':[n,4,1024],'register_shape':[n,4,4,1024],
        'registers_excluded_from_patch_statistics':True,'normalization':'official post-layer norm=True','input_grid':'native256/16 row-major',
        'autocast':'bfloat16','cache_dtype':'float16 (exactly represents bfloat16 outputs)','frozen_encoder':True})
    labels=np.load(R/'train_labels.npz');train=np.array([i['role']=='train' for i in items]);sources=sorted({i['source'] for i in items if i['role']=='train'})
    valid=labels['valid'][train].reshape(-1);target=labels['fraction'][train].reshape(-1)
    source_ids=np.repeat([sources.index(i['source']) for i in items if i['role']=='train'],256)
    usable=valid&((target==0)|(target>=.5));binary=target>=.5
    bins={(si,c):np.flatnonzero(usable&(source_ids==si)&(binary==c)) for si in range(len(sources)) for c in [False,True]}
    available=[si for si in range(len(sources)) if len(bins[si,False]) and len(bins[si,True])]
    assert available,'No supervised foreground/background training tokens'
    rng=np.random.default_rng(protocol['seed']);batches=[]
    # One identical coordinate minibatch plan for all models and all four layers.
    for _ in range(protocol['probe']['steps']):
        si=rng.choice(available,protocol['probe']['batch']);cl=rng.integers(2,size=len(si));batch=np.empty(len(si),np.int64)
        for s in available:
            for c in [0,1]:
                pick=(si==s)&(cl==c);pool=bins[s,bool(c)];batch[pick]=pool[rng.integers(len(pool),size=pick.sum())]
        batches.append(batch)
    sequence=sha(R/'samples.json');batch_hash=__import__('hashlib').sha256(np.stack(batches).tobytes()).hexdigest()
    states={};predictions=np.lib.format.open_memmap(out/'linear_probability.npy',mode='w+',dtype=np.float32,shape=(n,4,256))
    prototypes=np.lib.format.open_memmap(out/'prototype_score.npy',mode='w+',dtype=np.float32,shape=(n,4,256))
    class_info={str(s):{'background':len(bins[s,False]),'foreground':len(bins[s,True])} for s in range(len(sources))}
    for li,layer in enumerate(layers):
        raw=features[layer]
        x=torch.tensor(np.array(raw[train],copy=True).reshape(-1,1024),device='cuda',dtype=torch.float32)
        w=torch.tensor(valid,dtype=torch.float32,device='cuda');source=torch.tensor(source_ids,device='cuda')
        weights=torch.zeros_like(w)
        for si in range(len(sources)):
            mask=(source==si)&(w>0);weights[mask]=1/max(1,int(mask.sum()))/len(sources)
        mean=(x*weights[:,None]).sum(0);var=((x-mean).square()*weights[:,None]).sum(0);scale=var.sqrt().clamp_min(1e-4)
        fg=torch.tensor(usable&binary,device='cuda');bg=torch.tensor(usable&(~binary),device='cuda')
        centered=x-mean;unit=F.normalize(centered,dim=1)
        fgproto=F.normalize((unit[fg]*weights[fg,None]).sum(0),dim=0);bgproto=F.normalize((unit[bg]*weights[bg,None]).sum(0),dim=0)
        x=(x-mean)/scale;head=torch.nn.Linear(1024,1,device='cuda');torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
        opt=torch.optim.AdamW(head.parameters(),lr=protocol['probe']['lr'],weight_decay=protocol['probe']['weight_decay'])
        y=torch.tensor(binary.astype(np.float32),device='cuda')
        for step,indices in enumerate(batches):
            idx=torch.as_tensor(indices,device='cuda');opt.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(head(x[idx]).squeeze(-1),y[idx]);assert torch.isfinite(loss)
            loss.backward();opt.step()
        assert all(torch.isfinite(p).all() for p in head.parameters())
        with torch.inference_mode():
            for first in range(0,n,16):
                z=torch.tensor(np.array(raw[first:first+16],copy=True),device='cuda',dtype=torch.float32)
                predictions[first:first+len(z),li]=head((z-mean)/scale).sigmoid().squeeze(-1).cpu().numpy()
                prototypes[first:first+len(z),li]=(F.normalize(z-mean,dim=-1)@(fgproto-bgproto)).cpu().numpy()
        states[layer]={'weight':head.weight.detach().cpu(),'bias':head.bias.detach().cpu(),'mean':mean.cpu(),'scale':scale.cpu(),'foreground_prototype':fgproto.cpu(),'background_prototype':bgproto.cpu()}
        atomic(out/'progress.json',{'stage':'train_only_linear_probes','completed_layers':li+1,'layers':layers,'steps_per_layer':500,'final_loss':float(loss),'seconds':time.time()-started})
        del x,unit,centered,head,opt;torch.cuda.empty_cache()
    predictions.flush();prototypes.flush();torch.save(states,out/'linear_probes.pt')
    atomic(out/'probe_complete.json',{'status':'complete','model':model_id,'layers':layers,'steps_per_layer':500,'parameter_count_per_layer':1025,
        'training_region_ids':sorted({i['region_id'] for i in items if i['role']=='train'}),'evaluation_GT_read':False,
        'probe_sha256':sha(out/'linear_probes.pt'),'coordinate_batch_hash':batch_hash,'train_sources':sources,
        'sources_with_both_classes':available,'class_counts':class_info,'seconds':time.time()-started,
        'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'encoder_asset_unchanged':sha(path)==spec['sha256']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);run(parser.parse_args().model)
