"""Accepted one-sided Barrier, one real student forward and one total backward."""
import torch
from barrier import separation_barrier


class ProductionBarrierAdapter:
    def __init__(self,net,anchor):
        self.net,self.anchor=net,anchor
        assert not any(p.requires_grad for p in anchor.parameters()) and not anchor.training
        self.original_student=net.get_student_output;self.original_losses=net.compute_losses
        self.latest={};self.anchor_tokens=None;self.student_calls=0
        net.get_student_output=self.capture_student;net.compute_losses=self.compute_losses

    def capture_student(self,**kwargs):
        self.latest={};self.student_calls=0
        images=kwargs['global_crops'].flatten(0,1)
        # Same unmasked eval anchor and RNG preservation as accepted C-1.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
            self.anchor_tokens=self.anchor(images,is_training=True)['x_norm_patchtokens']
        self.student_calls+=1
        return self.original_student(**kwargs)

    def _barrier_backward(self,gradient):
        scale=float(gradient.detach())
        assert scale==1.0,'Fixed lambda1 must participate in the same backward'
        self.latest['barrier_backward_seen']=True;self.latest['barrier_upstream_scale']=scale

    def _token_gradient(self,gradient):
        assert torch.isfinite(gradient).all()
        self.latest['student_patch_total_gradient_norm']=float(gradient.detach().float().norm())

    def compute_losses(self,**kwargs):
        base,metrics=self.original_losses(**kwargs)
        tokens=kwargs['student_global']['patch_pre_head'].flatten(0,1)
        assert tokens.shape==self.anchor_tokens.shape and tokens.shape[1:]==(256,1024)
        raw,stats=separation_barrier(tokens,self.anchor_tokens,grid_hw=(16,16),radius=2,quantile=.5,margin=.05)
        assert raw.requires_grad and torch.isfinite(raw)
        self.latest={**stats,'barrier_raw':float(raw.detach()),'barrier_weighted':float(raw.detach()),
          'vanilla_loss':float(base.detach()),'lambda':1.0,'barrier_backward_seen':False,
          'barrier_upstream_scale':None,'production_mode':True,'diagnostic_extra_student_forward':False,
          'diagnostic_VJP':False,'student_forward_calls':self.student_calls,
          'mask_fraction':float(kwargs['masks'].float().mean()),'affinity_finite':True}
        raw.register_hook(self._barrier_backward);tokens.register_hook(self._token_gradient)
        return base+raw,metrics|{'barrier_raw':raw.detach()}

    def finish_step(self):
        assert self.latest['barrier_backward_seen'] and self.latest['student_forward_calls']==1
        assert self.latest['barrier_upstream_scale']==1.0
        assert not any(p.grad is not None for p in self.anchor.parameters())
        self.anchor_tokens=None
        return dict(self.latest,anchor_gradients_present=False)
