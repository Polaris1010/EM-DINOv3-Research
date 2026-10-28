"""Production integration: no extra student forward/VJP, same total backward."""
import os,unittest,torch
from production_adapter import ProductionBarrierAdapter
class Backbone(torch.nn.Module):
 def __init__(self,bias):
  super().__init__();self.conv=torch.nn.Conv2d(3,1024,16,stride=16);self.calls=0
  torch.nn.init.normal_(self.conv.weight,std=.01);torch.nn.init.constant_(self.conv.bias,bias)
 def forward(self,x,is_training=True):
  self.calls+=1;return {'x_norm_patchtokens':self.conv(x).flatten(2).transpose(1,2)}
class Toy(torch.nn.Module):
 def __init__(self):super().__init__();self.backbone=Backbone(1)
 def get_student_output(self,**kwargs):
  images=kwargs['global_crops'];b=images.shape[1];f=self.backbone(images.flatten(0,1))['x_norm_patchtokens'];return {'patch_pre_head':f.unflatten(0,(2,b))},{}
 def compute_losses(self,**kwargs):return kwargs['student_global']['patch_pre_head'].square().mean(),{}
class ProductionTests(unittest.TestCase):
 def test_single_forward_backward_and_optimizer(self):
  self.assertTrue(torch.cuda.is_available());torch.manual_seed(28);net=Toy().cuda();anchor=Backbone(0).cuda().requires_grad_(False).eval();adapter=ProductionBarrierAdapter(net,anchor)
  images=torch.randn(2,1,3,256,256,device='cuda');masks=torch.zeros(2,256,dtype=torch.bool,device='cuda')
  before=net.backbone.conv.weight.detach().clone();opt=torch.optim.AdamW(net.parameters(),lr=1e-4)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   g,l=net.get_student_output(global_crops=images);loss,_=net.compute_losses(student_global=g,student_local=l,masks=masks)
  loss.backward();health=adapter.finish_step();self.assertTrue(health['barrier_backward_seen']);self.assertEqual(health['barrier_upstream_scale'],1);self.assertGreater(health['barrier_raw'],0)
  self.assertEqual(net.backbone.calls,1);self.assertEqual(anchor.calls,1);self.assertFalse(health['diagnostic_VJP']);self.assertTrue(torch.isfinite(net.backbone.conv.weight.grad).all());opt.step();self.assertFalse(torch.equal(before,net.backbone.conv.weight))
  self.assertTrue(all(p.grad is None for p in anchor.parameters()))
if __name__=='__main__':unittest.main(verbosity=2)
