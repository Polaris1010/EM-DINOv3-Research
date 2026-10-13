import os,json,signal,gc,copy
from pathlib import Path
import torch,numpy as np
import torch.nn.functional as F
from torch import nn
from scipy import ndimage
from skimage.segmentation import watershed
import runtime as rt
import compare as co
BASE=Path(__file__).resolve().parent
class Decoder(nn.Module):
 def __init__(self):
  super().__init__();self.proj=nn.ModuleList([nn.Conv2d(1024,32,1) for _ in range(4)]);self.stem=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.GELU());self.head=nn.Sequential(nn.Conv2d(144,64,3,padding=1),nn.GELU(),nn.Conv2d(64,3,1))
 def forward(self,x,feats):
  f=[F.interpolate(p(v.float()),size=x.shape[-2:],mode='bilinear',align_corners=False) for p,v in zip(self.proj,feats)];return self.head(torch.cat([*f,self.stem(x)],1))
def main():
 signal.signal(signal.SIGINT,signal.default_int_handler);rt.seed();rt.OUT=BASE;out=Path(os.environ['RUN_OUTPUT']);co.OUT=out
 from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.vision_transformer import vit_large
 with torch.device('meta'):enc=vit_large(patch_size=16,drop_path_rate=.3,layerscale_init=1e-5,n_storage_tokens=0,qkv_bias=True,mask_k_bias=False)
 ck=torch.load(rt.EM,map_location='cpu',mmap=True,weights_only=True)['teacher'];enc.load_state_dict({k.removeprefix('backbone.'):v for k,v in ck.items() if k.startswith('backbone.')},strict=True,assign=True);enc=enc.eval().cuda().requires_grad_(False);del ck;gc.collect()
 def features(x):
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):return enc.get_intermediate_layers(x.repeat(1,3,1,1),n=[5,11,17,23],reshape=True)
 (out/'config.json').write_text(json.dumps(dict(encoder=str(rt.EM),encoder_frozen=True,steps=200,seed=rt.SEED,lr=.001,weight_decay=.04,batch=1,num_workers=2,crop=192,loss='foreground CE weight4 + soft Dice; treatment adds boundary BCE weight.2',train='jiang25_vol1',val='same 20 fixed patches; 3 constant raw patches flagged',comparison='semantic/boundary: auxiliary loss only; Gram/control: SSL encoder only; same decoder seed/data/optimizer'),indent=2))
 experiments=[('light_semantic',rt.EM,0.),('light_boundary',rt.EM,.2),('light_gram',BASE/'gram_smoke_workers0/checkpoint.pth',0.),('light_control',BASE/'ssl_control_workers0/checkpoint.pth',0.)]
 for name,encoder_path,boundary_weight in experiments:
  if encoder_path!=rt.EM:
   sd=torch.load(encoder_path,map_location='cpu',mmap=True,weights_only=True)['teacher'];enc.load_state_dict({k.removeprefix('backbone.'):v for k,v in sd.items() if k.startswith('backbone.')},strict=True);del sd;gc.collect()
  rt.seed();decoder=Decoder().cuda();opt=torch.optim.AdamW(decoder.parameters(),lr=.001,weight_decay=.04);completed=0
  try:
   with (out/(name+'_steps.jsonl')).open('w') as f:
    for step,b in enumerate(rt.loader('train',200,192,True)):
     x=b['image'].cuda();target=b['label'][:,0].cuda();feats=features(x);decoder.train();opt.zero_grad(set_to_none=True);r=decoder(x,feats);prob=r[:,:2].softmax(1)[:,1];truth=target.float()
     dice=1-(2*(prob*truth).sum()+1)/(prob.sum()+truth.sum()+1);ce=F.cross_entropy(r[:,:2],target,weight=torch.tensor([1.,4.],device='cuda'))
     # Semantic foreground contour only; touching-instance boundaries require the proposed instance-aware target.
     edge=F.max_pool2d(truth[:,None],3,1,1)-(-F.max_pool2d(-truth[:,None],3,1,1));aux=F.binary_cross_entropy_with_logits(r[:,2:3],edge)
     loss=ce+dice+boundary_weight*aux;loss.backward();grad=nn.utils.clip_grad_norm_(decoder.parameters(),12,error_if_nonfinite=True);opt.step();completed=step+1
     f.write(json.dumps(dict(step=completed,loss=float(loss.detach()),ce=float(ce.detach()),dice_loss=float(dice.detach()),boundary_loss=float(aux.detach()),grad=float(grad)))+'\n');f.flush()
     if completed%25==0:print(name,completed,float(loss.detach()),flush=True)
  finally:
   torch.save(dict(head=decoder.state_dict(),completed_steps=completed,encoder_reference=str(encoder_path),boundary_weight=boundary_weight),out/(name+'.pth'))
  decoder.eval();rows=[];wrows=[]
  coords=json.loads((BASE/'comparison_01/mitonet.json').read_text())['rows']
  for i in range(20):
   a=np.load(BASE/'comparison_01'/f'mitonet_{i:02}.npz');raw=a['raw'];lab=a['truth'];x=torch.from_numpy((raw.astype('float32')/255.-.485)/.229)[None,None].cuda()
   with torch.no_grad():r=decoder(x,features(x));prob=r[:,:2].softmax(1)[0,1].cpu().numpy();edge=r[0,2].sigmoid().cpu().numpy()
   mask=prob>=.5;pred=ndimage.label(mask)[0]
   rows.append(dict(index=i,volume=coords[i]['volume'],zyx=coords[i]['zyx'],**rt.metrics(rt.confusion(mask,lab>0)),**co.instance_score(lab,pred)));co.save_rows(name,rows)
   np.savez_compressed(out/f'{name}_{i:02}.npz',probability=prob,boundary=edge,instance=pred)
  del decoder,opt;gc.collect();torch.cuda.empty_cache()
 (out/'complete.json').write_text(json.dumps(dict(status='completed',experiments=4)))
if __name__=='__main__':main()
