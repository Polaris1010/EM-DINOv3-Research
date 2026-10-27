"""Two-scalar train-only convex fit in a separate process; no evaluation labels."""
import sys,time
import numpy as np
import torch
from common import R,P,MODELS,read,sha,atomic,parent_data,samples,native,setup
def solve_bias(z,target,w):
    prior=(w*target).sum();lo,hi=-64.,64.
    def grad(b):return float((w*torch.sigmoid(z+b)).sum()-prior)
    assert grad(lo)<0 and grad(hi)>0
    for iteration in range(80):
        b=(lo+hi)/2;value=grad(b)
        if value>0:hi=b
        else:lo=b
        if abs(value)<1e-11:break
    return b,value,iteration+1
def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes)):return
    name=str(args[0]);assert not any(x in name for x in ['evaluation_labels.npz','/baseline/','baseline_','/mb1_official_benchmark_20261015/']), 'Forbidden eval/baseline read: '+name
def run():
    setup();start=time.time();train=[i for i in samples() if i['role']=='train'];ids=[i['id'] for i in train]
    assert len(train)==312 and len({i['region_id'] for i in train})==156
    data=parent_data();sys.addaudithook(guard)
    gt=[];valid=[];sources=sorted({i['source'] for i in train})
    for i in train:
        assert i['domain']=='MV'
        gt.append(data.tile(i,'label')>0);valid.append(data.tile(i,'valid').astype(bool))
    gt=torch.tensor(np.stack(gt),device='cuda');valid=torch.tensor(np.stack(valid),device='cuda')
    source=torch.tensor([sources.index(i['source']) for i in train],device='cuda')[:,None,None].expand_as(valid)[valid]
    target=gt[valid].double();w=torch.zeros_like(target);info={}
    for k,s in enumerate(sources):
        mask=source==k;n=int(mask.sum());assert n>0;w[mask]=1/(len(sources)*n)
        info[s]={'valid_pixels':n,'foreground_pixels':int(target[mask].sum()),'weight':1/len(sources)}
    prior=(w*target).sum();answers={}
    with torch.inference_mode():
        for model in MODELS:
            p=native(model,ids)[valid].double();clipped=(p<1e-6)|(p>1-1e-6)
            z=torch.logit(p.clamp(1e-6,1-1e-6));b,value,iterations=solve_bias(z,target,w)
            def loss(shift):return float((w*(torch.nn.functional.softplus(z+shift)-target*(z+shift))).sum())
            answers[model]={'bias':b,'threshold':float(1/(1+np.exp(b))),'T':1.,'gradient_residual':value,'bisection_iterations':iterations,
              'source_equal_foreground_prior':float(prior),'weighted_BCE_raw':loss(0),'weighted_BCE_fit':loss(b),'fit_clip_fraction':float(clipped.double().mean()),'fit_epsilon':1e-6,
              'probe_sha256':sha(P/'models'/model/'linear_probes.pt'),'probability_sha256':sha(P/'models'/model/'linear_probability.npy')}
    atomic(R/'train_only_bias.json',{'status':'PINNED','models':answers,'fit_role':'MV-Generalist-v2_train_only','train_sample_ids':ids,
      'train_regions':sorted({i['region_id'] for i in train}),'train_sources':info,'samples_sha256':sha(P/'samples.json'),'code_commit':read(R/'protocol.json')['code_commit'],
      'source_probability_weighting':'source-equal; natural within-source pixel class proportions','evaluation_GT_read':False,'baseline_metrics_read':False,
      'encoder_forward_calls':0,'network_updates':0,'seconds':time.time()-start,'interpretation':'Separately fitted probes; no encoder common-mode drift attribution'})
if __name__=='__main__':run()
