import os,sys,json,hashlib,random,gc
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=Path('/opt/data/3_117/tanghr')
NN=ROOT/'Med_DINOv3/MedDINOv3/nnUNet'
sys.path.insert(0,str(ROOT/'codex/mitotopo_20260908_round2/deps'))
sys.path.insert(0,str(NN));sys.path.insert(0,str(NN/'nnunetv2/training/nnUNetTrainer/dinov3'))
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np,torch,zarr
from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.vision_transformer import vit_large
from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import Primus_Multiscale
PATHS={'P0':ROOT/'DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth','P1':ROOT/'DINOv3/med_checkpoints_continual_v2/eval/training_124999/teacher_checkpoint.pth','P2':ROOT/'codex/mitotopo_20260909_round3/jobs/ssl_b2_w0_lr1_s2000/teacher_latest.pth'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for a in iter(lambda:f.read(8*1024**2),b''):h.update(a)
 return h.hexdigest()
def save(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,default=str));tmp.replace(p)
def seed(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.set_num_threads(2)
 torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False

def encoder_sd(p):
 ck=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
 return {k.removeprefix('backbone.'):v for k,v in ck['teacher'].items() if k.startswith('backbone.')} if 'teacher' in ck else ck

def encoder(sd=None,natural=False):
 with torch.device('meta'):
  e=vit_large(patch_size=16,drop_path_rate=.3,layerscale_init=1e-5,n_storage_tokens=4 if natural else 0,qkv_bias=True,mask_k_bias=natural)
 if sd is not None:e.load_state_dict(sd,strict=True,assign=True)
 return e

def model(pid,s,device='cuda'):
 sd=encoder_sd(BASE/'natural_converted.pth' if pid=='P0' else PATHS[pid]);e=encoder(sd);e.requires_grad_(False)
 seed(s);net=Primus_Multiscale(embed_dim=1024,patch_embed_size=16,num_classes=2,dino_encoder=e,interaction_indices=[5,11,17,23])
 load_head(net,torch.load(BASE/f'shared_head_init_seed{s}.pth',map_location='cpu',weights_only=True))
 net.to(device);net.dino_encoder.eval();return net

def head(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items() if not k.startswith('dino_encoder.')}
def load_head(net,state):
 expected={k for k in net.state_dict() if not k.startswith('dino_encoder.')};assert set(state)==expected
 missing,unexpected=net.load_state_dict(state,strict=False)
 assert not unexpected and set(missing)=={k for k in net.state_dict() if k.startswith('dino_encoder.')}

def forward(net,x):
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  fs=net.dino_encoder.get_intermediate_layers(x,n=net.interaction_indices,reshape=True)
 return net.up_projection(torch.cat([g(f.float()) for g,f in zip(net.last_gates,fs)],dim=1))

def normalize(x,norm):
 a=x.astype(np.float32)/255.;a=np.repeat(a[None],3,axis=0)
 mean=np.array([.485]*3 if norm=='N1' else [.485,.456,.406],dtype=np.float32)[:,None,None]
 std=np.array([.229]*3 if norm=='N1' else [.229,.224,.225],dtype=np.float32)[:,None,None]
 return (a-mean)/std

def confusion(p,t):return [int((p&t).sum()),int((p&~t).sum()),int((~p&t).sum()),int((~p&~t).sum())]
def metrics(c):
 tp,fp,fn,tn=map(int,c)
 return dict(tp=tp,fp=fp,fn=fn,tn=tn,dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,precision=tp/(tp+fp) if tp+fp else (0. if fn else 1.),recall=tp/(tp+fn) if tp+fn else 1.)
