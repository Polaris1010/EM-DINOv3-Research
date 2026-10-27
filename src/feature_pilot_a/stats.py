"""GPU numerical statistics. Rank is a descriptive covariance statistic, never weight SVD."""
import math
import numpy as np
import torch
import torch.nn.functional as F


def ranks(x):
    order=torch.argsort(x,stable=True);sorted_x=x[order]
    _,counts=torch.unique_consecutive(sorted_x,return_counts=True)
    ends=counts.cumsum(0);starts=ends-counts
    average=(starts+ends-1).float()/2+1
    values=torch.repeat_interleave(average,counts)
    result=torch.empty_like(values);result[order]=values
    return result


def auc(positive,negative):
    if len(positive)==0 or len(negative)==0:return None
    values=torch.cat([positive,negative]);rank=ranks(values)
    return float((rank[:len(positive)].sum(dtype=torch.float64)-len(positive)*(len(positive)+1)/2)/(len(positive)*len(negative)))


def cosine_within(x):
    if len(x)<2:return None
    u=F.normalize(x,dim=-1)
    return float((u.sum(0).square().sum()-len(x))/(len(x)*(len(x)-1)))


def separation(fg,bg,train_mean):
    if len(fg)<2 or len(bg)<2:return {}
    def between(a,b):return float(F.normalize(a,dim=-1).mean(0)@F.normalize(b,dim=-1).mean(0))
    variance=fg.var(0,unbiased=False).sum()+bg.var(0,unbiased=False).sum()
    fisher=float((fg.mean(0)-bg.mean(0)).square().sum()/variance.clamp_min(1e-12))
    return {'FG_within_cosine':cosine_within(fg),'BG_within_cosine':cosine_within(bg),
            'between_cosine':between(fg,bg),'train_centered_between_cosine':between(fg-train_mean,bg-train_mean),
            'train_centered_FG_within_cosine':cosine_within(fg-train_mean),
            'train_centered_BG_within_cosine':cosine_within(bg-train_mean),
            'centered_Fisher_ratio':fisher}


def spectrum(x):
    x=x.double();centered=x-x.mean(0)
    # Nonzero covariance eigenvalues via sample-space covariance; no model weight decomposition.
    eig=torch.linalg.eigvalsh(centered@centered.T/(len(x)-1)).clamp_min(0)
    total=eig.sum();p=eig/total.clamp_min(1e-30);entropy=-(p*p.clamp_min(1e-30).log()).sum()
    unit=F.normalize(x,dim=1);mean_unit=unit.mean(0)
    return {'effective_rank':float(entropy.exp()) if float(total)>1e-20 else 0.,'top1_explained_variance':float(p[-1]) if float(total)>1e-20 else None,
            'anisotropy_mean_cosine':float((mean_unit.square().sum()*len(x)-1)/(len(x)-1)),
            'feature_total_variance':float(eig.sum()),'normalized_eigenvalues':p.flip(0).cpu().numpy()}


def spearman(x,y):
    good=torch.isfinite(x)&torch.isfinite(y);x,y=x[good],y[good]
    if len(x)<3:return None
    a,b=ranks(x),ranks(y);a-=a.mean();b-=b.mean()
    den=a.square().sum().sqrt()*b.square().sum().sqrt()
    return float((a*b).sum()/den) if float(den)>0 else None


def partial_rank(x,y,covariates):
    a,b=ranks(x),ranks(y)
    basis=[torch.ones_like(x)/math.sqrt(len(x))]
    for col in covariates.T:
        if float(col.std())<=1e-10:continue
        residual=ranks(col)
        for _ in range(2):
            for q in basis:residual=residual-q*(q@residual)
        norm=residual.norm()
        if float(norm)>1e-4:basis.append(residual/norm)
    for q in basis:
        a=a-q*(q@a);b=b-q*(q@b)
    den=a.square().sum().sqrt()*b.square().sum().sqrt()
    return float((a*b).sum()/den) if float(den)>0 else None


def periodicity(windows):
    x=windows.float()/255;contrast=x.std((-2,-1),unbiased=False)
    gradient=((x[:,:,1:]-x[:,:,:-1]).square().mean((-2,-1))+(x[:,1:,:]-x[:,:-1,:]).square().mean((-2,-1))).sqrt()
    taper=torch.hann_window(64,periodic=False,device=x.device)
    y=(x-x.mean((-2,-1),keepdim=True))*taper[None,:,None]*taper[None,None,:]
    power=torch.fft.rfft2(y).abs().square()
    fy=torch.fft.fftfreq(64,device=x.device);fx=torch.fft.rfftfreq(64,device=x.device)
    radius=(fy[:,None].square()+fx[None,:].square()).sqrt();mask=(radius>=1/32)&(radius<=1/4)
    p=power[:,mask];p=p/p.sum(1,keepdim=True).clamp_min(1e-20)
    concentration=p.topk(4,dim=1).values.sum(1)
    entropy=-(p*p.clamp_min(1e-20).log()).sum(1)/math.log(p.shape[1])
    peak=p.argmax(1);cycles=radius[mask][peak]
    return torch.stack([concentration,entropy,contrast,gradient,cycles],1)


def bootstrap_auc_delta(score_a,score_b,positive,background,blocks,repeats,seed):
    unique=sorted(set(blocks));indices=[np.flatnonzero(np.array(blocks)==b) for b in unique]
    rng=np.random.default_rng(seed);values=[]
    for _ in range(repeats):
        picked=rng.integers(len(indices),size=len(indices));idx=torch.tensor(np.concatenate([indices[k] for k in picked]),device=score_a.device)
        p,b=positive[idx],background[idx]
        a=auc(score_a[idx][p],score_a[idx][b]);c=auc(score_b[idx][p],score_b[idx][b])
        if a is not None and c is not None:values.append(c-a)
    return {'delta_bootstrap_low':float(np.quantile(values,.025)) if values else None,
            'delta_bootstrap_high':float(np.quantile(values,.975)) if values else None,
            'Z_blocks':len(unique),'bootstrap_valid_repeats':len(values)}


def bootstrap_spearman(x,y,blocks,repeats,seed):
    unique=sorted(set(blocks));groups=[np.flatnonzero(np.array(blocks)==b) for b in unique]
    rng=np.random.default_rng(seed);values=[]
    for _ in range(repeats):
        selected=rng.integers(len(groups),size=len(groups));idx=torch.tensor(np.concatenate([groups[k] for k in selected]),device=x.device)
        value=spearman(x[idx],y[idx])
        if value is not None:values.append(value)
    return {'rho_block_low':float(np.quantile(values,.025)) if values else None,
            'rho_block_high':float(np.quantile(values,.975)) if values else None,'rho_Z_blocks':len(unique)}
