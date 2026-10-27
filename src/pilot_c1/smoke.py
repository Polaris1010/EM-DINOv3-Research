"""Bounded paired C-1 preflight: exactly20 updates/branch, never formal training."""
from __future__ import annotations
import os,sys,json,time,hashlib,gc,random,argparse
from pathlib import Path
ROOT=Path(os.environ['PILOT_C1_ROOT']);ASSET=ROOT.parent.parent
LATE=ROOT.parent/'ssl_diversity_d12_100k_20261018'
sys.path[:0]=[str(ASSET/'DINOv3/dinov3'),str(LATE/'code')]
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from dinov3.configs import get_default_config
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.fsdp.ac_compile_parallelize import activation_checkpoint_transformer
from dinov3.models import build_model_from_cfg
from ssl_full_data import Images,Collate
from runtime_guard import training_gpu_guard,website
import importlib.util
spec=importlib.util.spec_from_file_location('historical_ssl_trainer',LATE/'code/train_ssl.py')
historical=importlib.util.module_from_spec(spec);spec.loader.exec_module(historical)
from adapter import BarrierAdapter


def atomic(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)


def tensor_hash(t):
    t=t.detach().cpu().contiguous()
    return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def rng_hash():
    h=hashlib.sha256();h.update(repr(random.getstate()).encode());h.update(repr(np.random.get_state()).encode())
    h.update(torch.get_rng_state().numpy().tobytes());h.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    return h.hexdigest()


def all_steps(optimizer):
    return sorted({int(x['step']) for x in optimizer.state.values()})


def main(branch):
    assert branch in ['C0','C1'];rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world==3;training_gpu_guard(world);torch.cuda.set_device(local);torch.set_num_threads(4)
    dist.init_process_group('nccl',device_id=torch.device('cuda',local));website()
    protocol=json.loads((ROOT/'protocol.json').read_text());assert protocol['max_updates_per_branch']==20
    out=ROOT/'smoke'/branch;out.mkdir(parents=True,exist_ok=True)
    cfg=OmegaConf.merge(get_default_config(),OmegaConf.load(LATE/'D12_base.yaml'))
    assert cfg.train.batch_size_per_gpu==16 and cfg.crops.global_crops_size==256 and cfg.crops.local_crops_size==112 and not cfg.gram.use_loss
    seed=1000;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark=False
    with torch.device('meta'):net=SSLMetaArch(cfg)
    net.to_empty(device='cuda')
    official=ASSET/'DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
    natural=torch.load(official,map_location='cpu',mmap=True,weights_only=True)
    net.student.backbone.load_state_dict(natural,strict=True);del natural
    net.student.dino_head.init_weights();net.student.ibot_head.init_weights();net.dino_loss.init_weights();net.ibot_patch_loss.init_weights()
    net.model_ema.load_state_dict(net.student.state_dict(),strict=True)
    activation_checkpoint_transformer(cfg,net.student.backbone)
    params=list(net.student.parameters());opt=torch.optim.AdamW(net.get_params_groups(),lr=float(cfg.optim.lr),betas=(float(cfg.optim.adamw_beta1),float(cfg.optim.adamw_beta2)),foreach=True)
    full=ROOT.parent/'ssl_diversity_d12_20261017/runs/D2/formal/checkpoints/step000030000/full.pth'
    saved=torch.load(full,map_location='cpu',mmap=True,weights_only=False)
    assert saved['step']==30000 and saved['identity']['world_size']==3
    net.student.load_state_dict(saved['student'],strict=True);net.model_ema.load_state_dict(saved['teacher'],strict=True)
    net.dino_loss.load_state_dict(saved['dino_loss'],strict=True);net.ibot_patch_loss.load_state_dict(saved['ibot_loss'],strict=True);opt.load_state_dict(saved['optimizer'])
    initial_exact=all(torch.equal(t,saved['student'][k]) for k,t in historical.full_state(net.student).items())
    teacher_exact=all(torch.equal(t,saved['teacher'][k]) for k,t in historical.full_state(net.model_ema).items())
    assert initial_exact and teacher_exact and all_steps(opt)==[30000]
    prior_rng=saved['rng_by_rank'][rank];chain=saved['sample_chain'];del saved;gc.collect()
    adapter=None;anchor=None
    if branch=='C1':
        with torch.device('meta'):anchor,dim=build_model_from_cfg(cfg,only_teacher=True)
        anchor.to_empty(device='cuda');v=torch.load(full.with_name('teacher_backbone.pth'),map_location='cpu',mmap=True,weights_only=True)
        anchor.load_state_dict(v['backbone'],strict=True);del v
        anchor.requires_grad_(False).eval();assert dim==1024 and anchor.n_storage_tokens==4 and anchor.patch_size==16
        adapter=BarrierAdapter(net,anchor,weight=protocol['barrier']['smoke_lambda'],margin=protocol['barrier']['margin'])
    net.train()
    dataset=Images(LATE/'D2_manifest.json',cfg,0,30000*16,20*16,rank=rank,world_size=3,batch_per_rank=16)
    loader=DataLoader(dataset,batch_size=16,num_workers=2,prefetch_factor=1,persistent_workers=False,pin_memory=False,multiprocessing_context='spawn',collate_fn=Collate(cfg),generator=torch.Generator().manual_seed(seed+rank))
    iterator=iter(loader);first_batch=next(iterator)
    if anchor is not None:
        with torch.random.fork_rng(devices=[local]),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            image=first_batch['collated_global_crops'][:1].cuda()
            direct=anchor(image,is_training=True)
            independent=anchor.get_intermediate_layers(image,n=[23],norm=True)[0]
            tokens=direct['x_norm_patchtokens']
            relative=float((tokens-independent).float().norm()/independent.float().norm())
            assert relative<1e-3 and tokens.shape==(1,256,1024)
            assert direct['x_storage_tokens'].shape[1]==4 and direct['x_prenorm'].shape[1]==261
            atomic(out/f'coordinate_check_rank{rank}.json',{'status':'PASS','independent_official_layer23_relative_L2':relative,'max_abs_difference':float((tokens-independent).abs().max()),'raster':'index=r*16+c; same global-crop tensor; no interpolation','CLS_tokens_excluded':1,'register_tokens_excluded':4,'patches':256,'real_training_image_used':True,'GT_used':False})
        del image,direct,independent,tokens
    historical.restore_rng(prior_rng)
    initial_rng=rng_hash();atomic(out/f'initial_rank{rank}.json',{'student_strict_exact':initial_exact,'teacher_strict_exact':teacher_exact,'optimizer_steps':all_steps(opt),'optimizer_groups':len(opt.param_groups),'RNG_restored_hash':initial_rng,'sample_chain':chain,'global_batch':48,'rank':rank,'teacher_is_model_ema':net.teacher is net.model_ema,'scheduler':'functional inherited100k schedule at step30000','GradScaler':'not used in historicalBF16 trainer'})
    rows=[];start=time.perf_counter()
    for step in range(30000,30020):
        website();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();beg=time.perf_counter()
        batch=first_batch if step==30000 else next(iterator);data_seconds=time.perf_counter()-beg
        locations=batch.pop('_sample_locations');first=batch.pop('_first_index');digest=batch.pop('_sample_digest')
        assert first==step*48+rank*16
        hashes={k:tensor_hash(v) for k,v in batch.items() if torch.is_tensor(v)}
        digests=[None]*3;dist.all_gather_object(digests,digest)
        for item in digests:chain=hashlib.sha256(bytes.fromhex(chain)+bytes.fromhex(item)).hexdigest()
        batch['global_batch_size']=48;lr=historical.learning_rate(step,float(cfg.optim.lr),float(cfg.optim.min_lr))
        for group in opt.param_groups:
            group['lr']=lr*group['lr_multiplier'];group['weight_decay']=float(cfg.optim.weight_decay)*group['wd_multiplier']
        before_rng=rng_hash();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=net.forward_backward(batch,teacher_temp=.07,iteration=step)
        assert torch.isfinite(loss)
        extra=adapter.gradient_diagnostics() if adapter else {}
        historical.reduce_gradients(params,3)
        grad=torch.nn.utils.clip_grad_norm_(params,float(cfg.optim.clip_grad),error_if_nonfinite=True)
        opt.step();net.update_ema(.999);torch.cuda.synchronize()
        seconds=time.perf_counter()-beg
        row={'branch':branch,'step':step+1,'rank':rank,'global_batch':48,'batch_first_index':first,'sample_digest':digest,'sample_chain':chain,'tensor_hashes':hashes,'before_model_RNG':before_rng,'after_model_RNG':rng_hash(),'lr':lr,'weight_decay':float(cfg.optim.weight_decay),'teacher_temp':.07,'teacher_momentum':.999,'loss':float(loss.detach()),'total_grad_norm':float(grad),'data_seconds':data_seconds,'iteration_seconds':seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'loss_components':{k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in metrics.items() if 'loss' in k},**extra}
        if adapter:
            row['production_estimate_seconds']=seconds-extra['diagnostic_VJP_seconds']-extra['diagnostic_norm_seconds']
        rows.append(row);atomic(out/f'progress_rank{rank}.json',{'step':step+1,'branch':branch,'seconds':time.perf_counter()-start})
        with (out/f'steps_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    assert all_steps(opt)==[30020]
    if anchor is not None:
        reference=torch.load(full.with_name('teacher_backbone.pth'),map_location='cpu',mmap=True,weights_only=True)
        assert all(torch.equal(t.cpu(),reference['backbone'][k]) for k,t in anchor.state_dict().items())
        del reference
    tracked=next(net.student.backbone.parameters()).detach();copies=[torch.empty_like(tracked) for _ in range(3)];dist.all_gather(copies,tracked)
    difference=max(float((copies[0]-x).abs().max()) for x in copies);assert difference<=1e-6
    atomic(out/f'complete_rank{rank}.json',{'status':'complete','updates':20,'initial_step':30000,'last_step':30020,'optimizer_steps':all_steps(opt),'rank_parameter_max_difference':difference,'sample_chain':chain,'anchor_no_grad':not any(p.grad is not None for p in anchor.parameters()) if anchor else True,'seconds':time.perf_counter()-start,'checkpoint_written':False,'formal_SSL_started':False,'Primus_started':False})
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--branch',choices=['C0','C1'],required=True);main(p.parse_args().branch)
