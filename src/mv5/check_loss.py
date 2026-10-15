"""CPU checks of the only loss adaptation: ignored synthetic padding."""
import torch
from empanada.losses import PanopticLoss
from train import panoptic_loss

def main():
 torch.manual_seed(17);criterion=PanopticLoss();b,h,w=2,8,8
 output={'sem_logits':torch.randn(b,1,h,w,requires_grad=True),'ctr_hmp':torch.randn(b,1,h,w,requires_grad=True),'offsets':torch.randn(b,2,h,w,requires_grad=True),'sem_points':torch.randn(b,1,4,requires_grad=True),'point_coords':torch.tensor([[[.125,.125],[.375,.375],[.625,.625],[.875,.875]]]*b)}
 target={'sem':(torch.rand(b,h,w)>.5).float(),'ctr_hmp':torch.rand(b,1,h,w),'offsets':torch.randn(b,2,h,w),'valid':torch.ones(b,h,w,dtype=torch.bool)}
 a,_=criterion(output,target);v,_=panoptic_loss(criterion,output,target);assert torch.equal(a,v)
 target['valid'][:,:,4:]=False
 first,_=panoptic_loss(criterion,output,target)
 changed={k:v.clone() for k,v in output.items()}
 for k in ['sem_logits','ctr_hmp','offsets']:changed[k][:,:,:,4:]+=1000
 changed['sem_points'][:,:,2:]+=1000
 second,_=panoptic_loss(criterion,changed,target);assert torch.equal(first,second)
 first.backward()
 for k in ['sem_logits','ctr_hmp','offsets']:assert torch.count_nonzero(output[k].grad[:,:,:,4:])==0
 assert torch.count_nonzero(output['sem_points'].grad[:,:,2:])==0
 assert torch.isfinite(first)
 print('PASS official loss exact without padding; padded outputs and points have zero effect and gradients')

if __name__=='__main__':main()
