import os,time,json,gc,hashlib,traceback,signal
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3' and os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID'
import torch,numpy as np
from PIL import Image,ImageDraw
import runtime as r
OUT=r.OUT
signal.signal(signal.SIGINT,signal.default_int_handler)
ACTIVE_MODEL=None
COMPLETED_STEPS=0
STEPS=int(os.environ.get('MITOTOPO_STEPS','200'))
assert 50<=STEPS<=200, 'First-round runner accepts 50-200 steps only'

def preview(batch,prob,prefix):
    a=((batch['image'][0,0].cpu().numpy()*.229+.485)*255).clip(0,255).astype('uint8');b=batch['label'][0,0].cpu().numpy();p=prob[0].cpu().float().numpy();h,w=a.shape
    sheet=Image.new('RGB',(4*w,h+24),'white')
    for i,(title,x) in enumerate([('input',a),('label',b*255),('prediction',(p>.5)*255),('probability',p*255)]):
        sheet.paste(Image.fromarray(x.astype('uint8')).convert('RGB'),(i*w,24));ImageDraw.Draw(sheet).text((i*w+3,3),title,fill='black')
    (OUT/'predictions_preview').mkdir(exist_ok=True);sheet.save(OUT/'predictions_preview'/f'{prefix}.png')

def evaluate(model,split,prefix,count=5,save_images=True):
    model.eval();records=[];counts=np.zeros(4,dtype=np.int64)
    with torch.inference_mode():
        for i,b in enumerate(r.loader(split,count)):
            x=b['image'].to('cuda');torch.cuda.synchronize();start=time.monotonic()
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x)
            torch.cuda.synchronize();elapsed=time.monotonic()-start
            assert logits.shape[2:]==b['label'].shape[2:] and torch.isfinite(logits).all()
            prob=logits.float().softmax(1)[:,1];assert torch.isfinite(prob).all()
            c=r.confusion(prob.cpu().numpy()>.5,b['label'][:,0].numpy());counts+=c
            records.append(dict(volume_id=b['volume_id'][0],index=i,zyx=b['zyx'][0].tolist(),seconds=elapsed,predicted_foreground=float((prob>.5).float().mean()),**r.metrics(c)))
            if save_images:preview(b,prob,f'{prefix}_{i}')
    result=dict(scope='five deterministic 192x192 patches, not full-volume evaluation',split=split,patches=records,pooled_sampled_voxels=r.metrics(counts),mean_patch_dice=float(np.mean([a['dice'] for a in records])))
    r.save(prefix+'_metrics.json',result);return result

def main():
    global ACTIVE_MODEL,COMPLETED_STEPS
    started=time.monotonic();r.seed();torch.backends.cudnn.benchmark=False
    torch.cuda.set_device(0)
    config={'experiment_id':'E0b_smoke','seed':r.SEED,'physical_gpu':3,'cuda_visible_devices':'3','cuda_device_order':'PCI_BUS_ID','em_checkpoint':str(r.EM),'gram_anchoring':False,'model':'original Primus_Multiscale + vit_large','layers_zero_based':[5,11,17,23],'n_storage_tokens':0,'patch_size':16,'crop':192,'classes':2,'max_steps':STEPS,'batch_size_per_gpu':1,'num_workers':2,'prefetch_factor':1,'persistent_workers':False,'pin_memory':False,'cache_dataset_in_ram':False,'amp':'bf16','gradient_accumulation_steps':1,'decoder_lr':3e-5,'backbone_lr':3e-6,'warmup_epochs':5,'steps_per_epoch':100,'epochs_reference':300,'weight_decay':.05,'adam_betas':[.9,.98],'normalization':'(uint8/255-0.485)/0.229; original ImageNetNormalization','augmentation':'synchronized flips only; exploratory adapter, not exact nnUNet augmentation','split':'exploratory_split.json','official_benchmark':False,'save_optimizer':False,'notes':'Encoder pretrained; decoder/gates initialized anew. No downstream checkpoint loaded.'}
    import yaml
    (OUT/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    model=r.build_model().to('cuda');ACTIVE_MODEL=model;criterion=r.loss_fn();params=sum(p.numel() for p in model.parameters())
    initial=evaluate(model,'val','before',save_images=True)
    initial_infer_peak=torch.cuda.max_memory_allocated()/1024**2
    trainloader=r.loader('train',STEPS,augment=True)
    probe=r.PatchDataset('train',1,192,True)[0];px=probe['image'][None].to('cuda');py=probe['label'][None].to('cuda')
    model.train()
    def probe_loss():
        torch.manual_seed(r.SEED)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):return float(criterion(model(px),py))
    initial_loss=probe_loss();repeat_loss=probe_loss();assert abs(initial_loss-repeat_loss)<1e-6,(initial_loss,repeat_loss)
    others=[p for n,p in model.named_parameters() if 'dino_encoder' not in n];backbone=[p for n,p in model.named_parameters() if 'dino_encoder' in n]
    optimizer=torch.optim.AdamW([{'params':others,'lr':3e-5},{'params':backbone,'lr':3e-6}],betas=(.9,.98),weight_decay=.05)
    trainstart=time.monotonic();gate_min=None;losses=[];times=[];torch.cuda.reset_peak_memory_stats();snapshots=[]
    for step,b in enumerate(trainloader):
        tick=time.monotonic();model.train();epoch=step//100
        factor=(1e-6+(3e-5-1e-6)*(epoch/5))/3e-5 if epoch<5 else 1-(epoch-5)/(300-5)
        optimizer.param_groups[0]['lr']=3e-5*factor;optimizer.param_groups[1]['lr']=3e-6*factor
        x=b['image'].to('cuda');y=b['label'].to('cuda');optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x);loss=criterion(logits,y)
        assert torch.isfinite(loss),'Non-finite loss';loss.backward()
        gates={n:float(p.grad.detach().norm()) for n,p in model.named_parameters() if 'last_gates' in n and p.grad is not None}
        assert gates and all(np.isfinite(v) for v in gates.values()),'Non-finite/missing gate gradients'
        if step==0:r.save('first_gate_gradients.json',gates)
        gradnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),12.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize();COMPLETED_STEPS=step+1
        elapsed=time.monotonic()-tick;times.append(elapsed);losses.append(float(loss.detach()))
        if step%10==0:print(json.dumps({'step':step+1,'loss':losses[-1],'lrs':[g['lr'] for g in optimizer.param_groups],'gradnorm':float(gradnorm),'seconds':elapsed}),flush=True)
        if (step+1)%50==0:
            snap=evaluate(model,'val',f'step_{step+1}',save_images=False);snapshots.append({'step':step+1,'mean_patch_dice':snap['mean_patch_dice']});print('VALIDATION',snapshots[-1],flush=True)
    train_elapsed=time.monotonic()-trainstart
    model.train();final_loss=probe_loss();final=evaluate(model,'val','after',save_images=True)
    allocated=torch.cuda.max_memory_allocated()/1024**2;reserved=torch.cuda.max_memory_reserved()/1024**2
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):reference=model(px).float().cpu()
    target=OUT/'smoke_checkpoint.pth';assert not target.exists()
    torch.save({'network_weights':model.state_dict(),'resolved_config':config,'step':COMPLETED_STEPS},target)
    ck=torch.load(target,map_location='cpu',mmap=True,weights_only=True);model.load_state_dict(ck['network_weights'],strict=True);del ck;gc.collect()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):restored=model(px).float().cpu()
    assert torch.equal(reference,restored),'Checkpoint reloaded predictions differ'
    import psutil
    down=torch.load(r.DOWN,map_location='cpu',mmap=True,weights_only=False)
    down_sd={k.removeprefix('_orig_mod.').removeprefix('module.'):v for k,v in down['network_weights'].items()}
    model.load_state_dict(down_sd,strict=True);del down,down_sd;gc.collect()
    downstream_result=evaluate(model,'val','existing_downstream',save_images=True)
    result={'status':'completed','initial_inference_peak_allocated_mib':initial_infer_peak,'existing_downstream':downstream_result,'steps':len(losses),'parameters':params,'initial_fixed_probe_loss':initial_loss,'repeat_initial_loss':repeat_loss,'final_fixed_probe_loss':final_loss,'fixed_probe_loss_decreased':final_loss<initial_loss,'first10_mean_loss':float(np.mean(losses[:10])),'last10_mean_loss':float(np.mean(losses[-10:])),'finite_gate_gradients':True,'checkpoint_reload_exact':True,'training_peak_allocated_mib':allocated,'training_peak_reserved_mib':reserved,'train_wall_seconds':train_elapsed,'mean_compute_step_seconds':float(np.mean(times)),'total_wall_seconds':time.monotonic()-started,'initial_val':initial,'final_val':final,'validation_history':snapshots,'losses':losses,'checkpoint_bytes':target.stat().st_size,'limitations':['Exploratory domain-disjoint split, not official benchmark','Only sampled patch metrics; no full-volume Dice or instance metrics','Original decoder starts random','Sparse data audit does not certify volume-global instance IDs','Not an exact nnUNet training reproduction']}
    r.save('metrics.json',result);print('COMPLETED',json.dumps({k:v for k,v in result.items() if k not in ['initial_val','final_val','losses','limitations']}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        if isinstance(e,KeyboardInterrupt) and ACTIVE_MODEL is not None and COMPLETED_STEPS:
            try:
                torch.save({'network_weights':ACTIVE_MODEL.state_dict(),'completed_steps':COMPLETED_STEPS,'status':'interrupted_not_validated'},OUT/'interrupted_checkpoint.pth')
            except Exception as save_error:
                r.save('interrupted_save_error.json',{'error':str(save_error)})
        r.save('smoke_failure.json',{'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()});raise



