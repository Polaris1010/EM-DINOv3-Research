import os,sys,json,random,time,signal,argparse,gc
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=Path('/opt/data/3_117/tanghr');SRC=ROOT/'DINOv3/dinov3'
sys.path.insert(0,str(BASE.parent/'mitotopo_20260908_round2/deps'));sys.path.insert(0,str(SRC))
import numpy as np,torch
from omegaconf import OmegaConf
from dinov3.configs import get_default_config
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.data import DataAugmentationDINO
from dinov3.data.collate import collate_data_and_cast
from dinov3.data.masking import MaskingGenerator
from torch.utils.data import Dataset,DataLoader
from functools import partial
from PIL import Image
EM=ROOT/'DINOv3/med_checkpoints_continual_v2/eval/training_124999/teacher_checkpoint.pth'
SEED=20260907;OFFSET=1000000
signal.signal(signal.SIGINT,signal.default_int_handler)

def config(control=False):
    orig=OmegaConf.load(BASE.parent/'mitotopo_20260908_round2/official/gram_anchor_official.yaml')
    cfg=OmegaConf.merge(get_default_config(),orig)
    old=OmegaConf.load(ROOT/'DINOv3/med_checkpoints_continual_v2/config.yaml')
    cfg.student=old.student;cfg.student.pretrained_weights='';cfg.student.resume_from_teacher_chkpt=str(EM)
    for head in ['dino','ibot']:
        for k,v in old[head].items():
            if k.startswith('head_'):cfg[head][k]=v
    cfg.gram.ckpt=str(EM);cfg.gram.use_loss=not control
    cfg.train.batch_size_per_gpu=2;cfg.train.num_workers=2;cfg.train.cache_dataset=False
    cfg.train.compile=False;cfg.train.sharded_eval_checkpoint=False
    return cfg

class Images(Dataset):
    def __init__(self,cfg,count,start=0):
        self.count=count;self.start=start;self.cat=json.loads((ROOT/'MitoVerse_data/catalog.json').read_text());self.aug=DataAugmentationDINO(global_crops_scale=cfg.crops.global_crops_scale,local_crops_scale=cfg.crops.local_crops_scale,local_crops_number=cfg.crops.local_crops_number,global_crops_size=cfg.crops.global_crops_size,local_crops_size=cfg.crops.local_crops_size,gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,gram_teacher_no_distortions=cfg.crops.gram_teacher_no_distortions,horizontal_flips=cfg.crops.horizontal_flips,mean=cfg.crops.rgb_mean,std=cfg.crops.rgb_std);self.group=None
    def __len__(self):return self.count
    def __getitem__(self,i):
        import zarr
        i+=self.start;random.seed(SEED+i);np.random.seed(SEED+i);torch.manual_seed(SEED+i)
        if self.group is None:self.group=zarr.open_group(str(ROOT/'MitoVerse_data'/self.cat['jiang25_vol1']['zarr']),mode='r')
        a=self.group['img'];rng=np.random.default_rng(SEED+i);z=int(rng.integers(a.shape[0]));y=int(rng.integers(a.shape[1]-512+1));x=int(rng.integers(a.shape[2]-512+1))
        raw=Image.fromarray(np.asarray(a[z,y:y+512,x:x+512])).convert('RGB')
        return self.aug(raw),i

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--control',action='store_true');ap.add_argument('--gram-weight',type=float,default=2.);ap.add_argument('--batch',type=int,default=2);ap.add_argument('--lr-factor',type=float,default=4.);ap.add_argument('--steps',type=int,default=50);ap.add_argument('--cpu',action='store_true');ap.add_argument('--workers',type=int,default=0);ap.add_argument('--resume',type=Path);a=ap.parse_args()
    out=Path(os.environ.get('RUN_OUTPUT',BASE/'ssl_cpu'));out.mkdir(exist_ok=True,parents=True)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2)
    cfg=config(False);cfg.gram.loss_weight_schedule.end=a.gram_weight;cfg.train.batch_size_per_gpu=a.batch;cfg.train.num_workers=a.workers;OmegaConf.save(cfg,out/'resolved_config.yaml')
    (out/'adaptations.json').write_text(json.dumps(dict(reference='official ViT-7B Gram YAML',architecture='existing ViT-L and matching DINO/iBOT heads retained',checkpoint=str(EM),initialization='teacher export to student and EMA; fresh optimizer; not optimizer resume',schedule_offset=OFFSET,actual_prior_em_steps=125000,batch_size=a.batch,gram_weight=a.gram_weight,lr_scale=a.lr_factor*(a.batch/1024)**.5,batch_reason='minimum nondegenerate per-crop KoLeo cohort',loader=dict(workers=a.workers,prefetch=1 if a.workers else None,persistent=False,pin=False,cache=False),single_gpu='unsharded fp32 master parameters, bf16 autocast, full block activation checkpointing, compile off',train_source='jiang25_vol1 only, lazy Zarr random 512 crops; no validation images; same coordinate RNG as smoke PNGs'),indent=2))
    resume=torch.load(a.resume,map_location='cpu',mmap=True,weights_only=False) if a.resume else None
    start=int(resume['completed_steps']) if resume else 0
    assert start<a.steps<=200000
    if resume:assert resume['config']['gram']['use_loss']==cfg.gram.use_loss
    ds=Images(cfg,(a.steps-start)*a.batch,start=start*a.batch);mask=MaskingGenerator(input_size=(16,16),max_num_patches=128)
    collate=partial(collate_data_and_cast,mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,mask_probability=cfg.ibot.mask_sample_probability,dtype=torch.float32,n_tokens=256,mask_generator=mask)
    assert a.resume is None, 'Fresh matched arms only; explicit resume implementation required'
    assert a.batch>=2 and a.gram_weight>=0 and a.lr_factor>0
    if a.cpu:
        from dinov3.train.cosine_lr_scheduler import linear_warmup_cosine_decay
        sched=linear_warmup_cosine_decay(0,0,a.gram_weight,1000000,1200000,1000)
        batch=collate([ds[i] for i in range(a.batch)])
        from dinov3.loss.gram_loss import GramLoss
        loss=GramLoss(apply_norm=True,remove_neg=False,remove_only_teacher_neg=False)
        x=torch.randn(2,16,32,requires_grad=True);y=torch.randn(2,16,32)
        assert loss(x,x.detach()).item()==0
        v=loss(x,y);v.backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
        result=dict(schedule={str(i):float(sched[OFFSET+i]) for i in [0,1,49,500,999,1000]},batch={k:list(v.shape) for k,v in batch.items() if torch.is_tensor(v)},gram_gradient_finite=True)
        (out/'cpu_tests.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return
    # GPU initialization is only reached through the external resource guard.
    print('CUDA',torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),flush=True)
    torch.distributed.init_process_group('nccl',init_method='file://'+str(out/'dist_init'),rank=0,world_size=1)
    net=SSLMetaArch(cfg);net.to_empty(device='cuda')
    ck=resume['teacher'] if resume else torch.load(EM,map_location='cpu',mmap=True,weights_only=True)['teacher']
    net.student.load_state_dict(resume['student'] if resume else ck,strict=True);net.model_ema.load_state_dict(ck,strict=True)
    if net.has_gram_teacher:
        net.gram_teacher.load_state_dict(resume['gram_teacher'] if resume else {k:v for k,v in ck.items() if k.startswith('backbone.')},strict=True);net.gram_teacher_initialized=True
    net.dino_loss.init_weights();net.ibot_patch_loss.init_weights();del ck;gc.collect()
    from dinov3.fsdp.ac_compile_parallelize import activation_checkpoint_transformer
    activation_checkpoint_transformer(cfg,net.student.backbone)
    params=net.get_params_groups();lr=float(cfg.schedules.lr.end)*a.lr_factor*(a.batch/1024)**.5
    for g in params:g['lr']=lr*g['lr_multiplier'];g['weight_decay']=.04*g['wd_multiplier']
    opt=torch.optim.AdamW(params,lr=lr,betas=(.9,.99),foreach=True)
    if resume:opt.load_state_dict(resume['optimizer'])
    del resume;gc.collect()
    dl=DataLoader(ds,batch_size=a.batch,num_workers=a.workers,prefetch_factor=1 if a.workers else None,persistent_workers=False,pin_memory=False,multiprocessing_context='spawn' if a.workers else None,collate_fn=collate,generator=torch.Generator().manual_seed(SEED))
    net.train();completed=start;status='interrupted';anchor_updates=0
    def checkpoint(name):
        # Unwrap checkpoint wrappers to export architecture-compatible keys.
        def clean(sd):return {k.replace('_checkpoint_wrapped_module.',''):v.detach().cpu() for k,v in sd.items()}
        payload=dict(teacher=clean(net.model_ema.state_dict()),student=clean(net.student.state_dict()),optimizer=opt.state_dict(),completed_steps=completed,schedule_iteration=OFFSET+completed,gram_teacher=clean(net.gram_teacher.state_dict()) if net.has_gram_teacher else None,config=OmegaConf.to_container(cfg),status=status)
        tmp=out/(name+'.tmp');torch.save(payload,tmp);tmp.replace(out/name)
    try:
        with (out/'steps.jsonl').open('w') as log:
            for step,b in enumerate(dl,start=start):
                b['global_batch_size']=a.batch;opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):total,values=net.forward_backward(b,teacher_temp=.07,iteration=OFFSET+step)
                grad=torch.nn.utils.clip_grad_norm_(net.student.parameters(),30,error_if_nonfinite=True)
                assert torch.isfinite(total), 'nonfinite SSL loss'
                opt.step();net.update_ema(.999);completed=step+1
                if cfg.gram.rep_update and net.has_gram_teacher and OFFSET+completed>=cfg.gram.it_first_update and (OFFSET+completed)%cfg.gram.update_frequency==0 and (cfg.gram.max_updates is None or anchor_updates<cfg.gram.max_updates):
                    net.update_gram();anchor_updates+=1;print('ANCHOR_UPDATE',completed,anchor_updates,flush=True)
                row=dict(anchor_updates=anchor_updates,batch_size=a.batch,step=completed,schedule_iteration=OFFSET+step,total=float(total.detach()),grad_norm=float(grad),base_lr=lr,**{k:float(v.detach()) if torch.is_tensor(v) else v for k,v in values.items()})
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                if completed%cfg.checkpointing.period==0:
                    checkpoint('checkpoint_latest.pth')
                    torch.save({'teacher':{k:v.detach().cpu() for k,v in net.model_ema.state_dict().items()},'completed_steps':completed},out/'teacher_latest.pth')
                    (out/'progress.json').write_text(json.dumps({'completed_steps':completed,'target_steps':a.steps,'gram_weight':row.get('gram_loss_weight'),'status':'running'}))
                if completed>=a.steps:break
        status='complete'
    finally:
        if completed>start:
            checkpoint('checkpoint_latest.pth')
            torch.save({'teacher':{k:v.detach().cpu() for k,v in net.model_ema.state_dict().items()},'completed_steps':completed},out/'teacher_latest.pth')
        (out/'progress.json').write_text(json.dumps({'completed_steps':completed,'target_steps':a.steps,'status':status}))
        torch.distributed.destroy_process_group()
        (out/'result.json').write_text(json.dumps(dict(status=status,completed_steps=completed,gram_enabled=a.gram_weight>0,gram_weight=a.gram_weight,batch_size=a.batch,lr_factor=a.lr_factor,base_lr=lr,anchor_updates=anchor_updates,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()),indent=2))
if __name__=='__main__':main()
