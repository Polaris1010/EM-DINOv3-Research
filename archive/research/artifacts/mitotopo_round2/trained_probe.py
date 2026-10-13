import os,json,gc,signal
from pathlib import Path
import torch,numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import runtime as rt
import compare as co
signal.signal(signal.SIGINT,signal.default_int_handler)
BASE=Path(__file__).resolve().parent;OUT=Path(os.environ['RUN_OUTPUT']);co.OUT=OUT
rt.seed()
from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.vision_transformer import vit_large
from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import Primus_Multiscale
with torch.device('meta'):
 enc=vit_large(patch_size=16,drop_path_rate=.3,layerscale_init=1e-5,n_storage_tokens=0,qkv_bias=True,mask_k_bias=False)
 net=Primus_Multiscale(embed_dim=1024,patch_embed_size=16,num_classes=2,dino_encoder=enc,interaction_indices=[5,11,17,23])
path=rt.ROOT/'Med_DINOv3/nnUNet_results/Dataset131_CMCT/meddinov3_base_primus_multiscale_Trainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth'
ck=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
sd={k.removeprefix('_orig_mod.').removeprefix('module.'):v for k,v in ck['network_weights'].items()}
net.load_state_dict(sd,strict=True,assign=True)
(OUT/'provenance.json').write_text(json.dumps(dict(path=str(path),epoch=ck['current_epoch'],init_args=ck.get('init_args'),strict_match=True),default=str,indent=2))
net=net.eval().cuda();del ck,sd;gc.collect()
coords=json.loads((BASE/'comparison_01/mitonet.json').read_text())['rows']
for variant in ['current_epoch266','tta4','gates_off']:
 rows=[];wrows=[]
 if variant=='gates_off':
  for gate in net.last_gates:gate.gamma.data.zero_()
 for i in range(20):
  sample=np.load(BASE/'comparison_01'/f'mitonet_{i:02}.npz');raw=sample['raw'];lab=sample['truth'];x=torch.from_numpy((raw.astype('float32')/255.-.485)/.229)[None,None].cuda()
  axes=[()] if variant!='tta4' else [(),(2,),(3,),(2,3)]
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   probs=[]
   for a in axes:
    xx=x.flip(a) if a else x;p=net(xx).float().softmax(1)[:,1:2];probs.append(p.flip(a) if a else p)
   prob=torch.stack(probs).mean(0).squeeze().cpu().numpy()
  pred=ndimage.label(prob>=.5)[0]
  row=dict(index=i,volume=coords[i]['volume'],zyx=coords[i]['zyx'],**rt.metrics(rt.confusion(pred>0,lab>0)),**co.instance_score(lab,pred));rows.append(row)
  distance=ndimage.distance_transform_edt(pred>0);peaks=peak_local_max(distance,min_distance=3,labels=pred>0,exclude_border=False);seeds=np.zeros(pred.shape,np.int32)
  if len(peaks):seeds[tuple(peaks.T)]=np.arange(1,len(peaks)+1)
  # Guarantee one seed per connected component, including thin components.
  for k in range(1,int(pred.max())+1):
   region=pred==k
   if not np.any(seeds[region]):
    pos=np.unravel_index(np.argmax(distance*region),pred.shape);seeds[pos]=int(seeds.max())+1
  wp=watershed(-distance,seeds,mask=pred>0);wrows.append(dict(row,**co.instance_score(lab,wp)))
  np.savez_compressed(OUT/f'{variant}_{i:02}.npz',probability=prob,instance=pred,watershed=wp)
  co.save_rows(variant,rows);co.save_rows(variant+'_watershed',wrows)
(OUT/'complete.json').write_text(json.dumps(dict(status='completed',samples=20,variants=['baseline','tta4','gates_off'],postprocess='CC and distance watershed min_distance3; fixed exploratory defaults; no validation parameter search')))
