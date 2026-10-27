"""Add one loss before the official single backward; do not edit SSLMetaArch."""
import time,types
import torch
from barrier import separation_barrier


class BarrierAdapter:
    def __init__(self, net, anchor, weight=1.0, margin=0.05):
        self.net,self.anchor,self.weight,self.margin=net,anchor,weight,margin
        self.backbone_parameters=list(net.student.backbone.parameters())
        self.original_student=net.get_student_output;self.original_losses=net.compute_losses
        self.latest={};self.barrier_gradients=None;self.global_crops=None
        net.get_student_output=self.capture_student
        net.compute_losses=self.compute_losses

    def capture_student(self,**kwargs):
        self.global_crops=kwargs['global_crops'].flatten(0,1)
        out=self.original_student(**kwargs)
        assert out[0]['patch_pre_head'].shape==(2,16,256,1024)
        return out

    def compute_losses(self,**kwargs):
        vanilla,metrics=self.original_losses(**kwargs)
        torch.cuda.synchronize();start=time.perf_counter()
        # Preserve the exact main-process RNG trajectory. The frozen anchor uses
        # the original same global images, unmasked, as in the reference-teacher
        # interface. Student tokens retain the actual SSL iBOT mask/DropPath.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
            anchor_tokens=self.anchor(self.global_crops,is_training=True)['x_norm_patchtokens']
        torch.cuda.synchronize();anchor_seconds=time.perf_counter()-start
        student_tokens=kwargs['student_global']['patch_pre_head'].flatten(0,1)
        assert student_tokens.shape==anchor_tokens.shape==(32,256,1024)
        raw,stats=separation_barrier(student_tokens,anchor_tokens,margin=self.margin)
        torch.cuda.synchronize();diagnostic_start=time.perf_counter()
        # Measurement-only VJP. It adds no gradients to .grad and no updates.
        # Record its time separately instead of charging it as production loss.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            grads=torch.autograd.grad(raw,self.backbone_parameters,retain_graph=True,allow_unused=True)
        torch.cuda.synchronize();diagnostic_seconds=time.perf_counter()-diagnostic_start
        self.barrier_gradients=grads
        self.latest={**stats,'barrier_raw':float(raw.detach()),'barrier_weight':self.weight,
          'barrier_weighted':float(raw.detach())*self.weight,'vanilla_loss':float(vanilla.detach()),
          'anchor_forward_seconds':anchor_seconds,'diagnostic_VJP_seconds':diagnostic_seconds,
          'student_token_dtype':str(student_tokens.dtype),'anchor_token_dtype':str(anchor_tokens.dtype),
          'anchor_gradients_present':any(p.grad is not None for p in self.anchor.parameters()),
          'reference_input':'same_global_images_unmasked','student_input':'original_SSL_iBOT_mask_and_DropPath',
          'mask_fraction':float(kwargs['masks'].float().mean())}
        if self.latest['anchor_gradients_present']:raise RuntimeError('Frozen anchor acquired gradients')
        return vanilla+self.weight*raw,metrics|{'barrier_raw':raw.detach()}

    def gradient_diagnostics(self):
        """Local pre-DDP backbone norms: total gradient minus exact barrier VJP."""
        torch.cuda.synchronize();start=time.perf_counter()
        squares_bar=[];squares_base=[]
        for p,g in zip(self.backbone_parameters,self.barrier_gradients):
            if g is None:
                if p.grad is not None:squares_base.append(p.grad.float().square().sum())
            else:
                squares_bar.append(g.float().square().sum())
                squares_base.append((p.grad.float()-self.weight*g.float()).square().sum())
        bar=float(torch.stack(squares_bar).sum().sqrt()) if squares_bar else 0.0
        base=float(torch.stack(squares_base).sum().sqrt())
        self.barrier_gradients=None
        torch.cuda.synchronize()
        self.latest.update(barrier_backbone_grad_norm_local=bar,vanilla_backbone_grad_norm_local=base,
          weighted_barrier_to_vanilla_grad_ratio=self.weight*bar/base if base else None,
          diagnostic_norm_seconds=time.perf_counter()-start)
        return self.latest
