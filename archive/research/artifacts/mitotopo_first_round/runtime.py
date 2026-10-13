import os,sys,json,time,hashlib,random
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np,torch,zarr
from torch.utils.data import Dataset,DataLoader
ROOT=Path('/opt/data/3_117/tanghr'); OUT=Path(__file__).resolve().parent
NNROOT=ROOT/'Med_DINOv3/MedDINOv3/nnUNet'
sys.path.insert(0,str(NNROOT))
sys.path.insert(0,str(NNROOT/'nnunetv2/training/nnUNetTrainer/dinov3'))
EM=ROOT/'DINOv3/med_checkpoints_continual_v2/eval/training_124999/teacher_checkpoint.pth'
DOWN=ROOT/'Med_DINOv3/nnUNet_results/Dataset_13799_CMCT_upgrade2_v4/meddinov3_base_primus_multiscale_Trainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth'
SEED=20260907

def seed():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2)

def save(name,data): (OUT/name).write_text(json.dumps(data,indent=2,default=str))

def build_model():
    from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.vision_transformer import vit_large
    from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import Primus_Multiscale
    encoder=vit_large(patch_size=16,drop_path_rate=0.3,layerscale_init=1e-5,n_storage_tokens=0,qkv_bias=True,mask_k_bias=False)
    ck=torch.load(EM,map_location='cpu',mmap=True,weights_only=True)
    sd={k.removeprefix('backbone.'):v for k,v in ck['teacher'].items() if k.startswith('backbone.')}
    expected=encoder.state_dict()
    report={'checkpoint':str(EM),'selection':'teacher.backbone only','missing':sorted(set(expected)-set(sd)),'unexpected':sorted(set(sd)-set(expected)),'shape_mismatch':{k:[list(expected[k].shape),list(sd[k].shape)] for k in set(expected)&set(sd) if expected[k].shape!=sd[k].shape},'excluded_pretraining_heads':[k for k in ck['teacher'] if not k.startswith('backbone.')]}
    save('em_key_match.json',report)
    assert not report['missing'] and not report['unexpected'] and not report['shape_mismatch'],report
    encoder.load_state_dict(sd,strict=True)
    model=Primus_Multiscale(embed_dim=1024,patch_embed_size=16,num_classes=2,dino_encoder=encoder,interaction_indices=[5,11,17,23])
    return model

class PatchDataset(Dataset):
    def __init__(self,split='train',count=200,crop=192,augment=False):
        self.cat=json.loads((ROOT/'MitoVerse_data/catalog.json').read_text());self.ids=json.loads((OUT/'exploratory_split.json').read_text())[split]
        assert self.ids;self.count=count;self.crop=crop;self.augment=augment;self.groups={}
    def __len__(self):return self.count
    def __getitem__(self,i):
        rng=np.random.default_rng(SEED+i);vid=self.ids[i%len(self.ids)]
        if vid not in self.groups:
            # Bound open handles; no pixel cache.
            if len(self.groups)>=2:self.groups.clear()
            self.groups[vid]=zarr.open_group(str(ROOT/'MitoVerse_data'/self.cat[vid]['zarr']),mode='r')
        g=self.groups[vid];im,lab=g['img'],g['mito'];h=min(self.crop,im.shape[1]);w=min(self.crop,im.shape[2])
        z=int(rng.integers(im.shape[0]));y=int(rng.integers(im.shape[1]-h+1));x=int(rng.integers(im.shape[2]-w+1))
        raw=np.asarray(im[z,y:y+h,x:x+w]);inst=np.asarray(lab[z,y:y+h,x:x+w])
        assert raw.dtype==np.uint8 and np.issubdtype(inst.dtype,np.integer)
        a=(raw.astype('float32')/255.-.485)/.229;b=(inst>0).astype('int64')
        if self.augment:
            if rng.random()<.5:a=np.flip(a,0);b=np.flip(b,0)
            if rng.random()<.5:a=np.flip(a,1);b=np.flip(b,1)
        assert a.shape==(self.crop,self.crop), 'Select a crop fitting every selected volume; do not mix padding in metrics'
        return {'image':torch.from_numpy(a.copy())[None],'label':torch.from_numpy(b.copy())[None],'volume_id':vid,'index':i,'zyx':torch.tensor([z,y,x])}

def loader(split,count,crop=192,augment=False):
    return DataLoader(PatchDataset(split,count,crop,augment),batch_size=1,num_workers=2,prefetch_factor=1,persistent_workers=False,pin_memory=False,multiprocessing_context='spawn',generator=torch.Generator().manual_seed(SEED))

def loss_fn():
    from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    return DC_and_CE_loss({'batch_dice':False,'smooth':1e-5,'do_bg':False,'ddp':False},{},weight_ce=1,weight_dice=1,ignore_label=None,dice_class=MemoryEfficientSoftDiceLoss)

def confusion(pred,true):
    p=np.asarray(pred,dtype=bool);t=np.asarray(true,dtype=bool)
    return np.array([(p&t).sum(),(p&~t).sum(),(~p&t).sum(),(~p&~t).sum()],dtype=np.int64)

def metrics(c):
    tp,fp,fn,tn=map(int,c)
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,precision=tp/(tp+fp) if tp+fp else (1. if not fn else 0.),recall=tp/(tp+fn) if tp+fn else 1.)

def metric_tests():
    t=np.zeros((4,4),int);t[1:3,1:3]=1
    assert metrics(confusion(t,t))['dice']==1
    assert metrics(confusion(t*0,t))['dice']==0
    assert abs(metrics(confusion(np.ones_like(t),t))['dice']-.4)<1e-8
    split=t.copy();split[2,1:3]=2
    merged=t.copy();two=t.copy();two[2,1:3]=2
    assert metrics(confusion(split>0,t>0))['dice']==1 and len(np.unique(split))-1==2
    assert metrics(confusion(merged>0,two>0))['dice']==1 and len(np.unique(merged))-1==1
    return {'perfect':'pass','all_background':'pass','all_foreground':'pass','split_merge_semantic_blindness':'pass','instance_metrics':'not implemented in first-round semantic smoke; do not claim instance performance'}

def cpu():
    seed();report={'metric_tests':metric_tests(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
    ds=PatchDataset('train',2,192,True);a=ds[0];b=ds[0];assert torch.equal(a['image'],b['image']) and torch.equal(a['label'],b['label'])
    batch=next(iter(loader('train',2,192,True)));assert torch.equal(batch['image'][0],a['image'])
    report['sample']={k:({'shape':list(v.shape),'dtype':str(v.dtype),'min':v.min().item(),'max':v.max().item()} if torch.is_tensor(v) else v) for k,v in a.items()}
    model=build_model();report['parameters']=sum(p.numel() for p in model.parameters());report['trainable']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.eval()
    with torch.no_grad():
        logits=model(a['image'][None,:,:64,:64]);assert logits.shape==(1,2,64,64);assert torch.isfinite(logits).all()
        report['cpu_forward']={'shape':list(logits.shape),'finite':True}
    criterion=loss_fn();toy=torch.randn(1,2,16,16,requires_grad=True);loss=criterion(toy,torch.randint(0,2,(1,1,16,16)));loss.backward();assert torch.isfinite(toy.grad).all();report['toy_loss']=float(loss.detach())
    ck=torch.load(DOWN,map_location='cpu',mmap=True,weights_only=False);sd=ck['network_weights'];sd={k.removeprefix('_orig_mod.').removeprefix('module.'):v for k,v in sd.items()};ms=model.state_dict()
    report['downstream_match']={'path':str(DOWN),'missing':sorted(set(ms)-set(sd)),'unexpected':sorted(set(sd)-set(ms)),'shape_mismatch':{k:[list(ms[k].shape),list(sd[k].shape)] for k in set(ms)&set(sd) if ms[k].shape!=sd[k].shape},'init_args':ck.get('init_args'),'epoch':ck.get('current_epoch'),'best_ema':ck.get('_best_ema')}
    save('cpu_check.json',report);print(json.dumps(report,default=str)[:5000],flush=True)

if __name__=='__main__':cpu()
