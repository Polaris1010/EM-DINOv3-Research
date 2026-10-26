"""Tiny controller-owned numerical checks; never launch alongside a foreign controller."""
import math
import torch
from common import R,atomic
from stats import auc,ranks,spectrum,periodicity,spearman

torch.set_num_threads(2)
x=torch.tensor([1.,1.,4.,2.],device='cuda')
assert torch.allclose(ranks(x),torch.tensor([1.5,1.5,4.,3.],device='cuda'))
assert abs(auc(torch.tensor([2.,2.],device='cuda'),torch.tensor([1.,2.],device='cuda'))-.75)<1e-12
assert spearman(x,torch.ones_like(x)) is None
assert spectrum(torch.zeros((16,32),device='cuda'))['effective_rank']==0
identity=spectrum(torch.eye(16,device='cuda'))
assert abs(identity['effective_rank']-15)<1e-6
stripe=(127.5+100*torch.cos(torch.arange(64,device='cuda')*2*math.pi/8))[None,:].repeat(64,1)
generator=torch.Generator(device='cuda').manual_seed(20261027)
noise=torch.rand((64,64),device='cuda',generator=generator)*255
values=periodicity(torch.stack([stripe,noise]))
assert values[0,0]>values[1,0]
atomic(R/'evidence/numerical_smoke.json',{'status':'PASS','tie_AUROC':.75,'centered_identity_effective_rank':identity['effective_rank'],
    'stripe_periodicity':float(values[0,0]),'noise_periodicity':float(values[1,0]),'GPU_accelerated':True,'no_training':True})
