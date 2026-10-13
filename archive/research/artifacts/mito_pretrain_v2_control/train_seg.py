"""Paired frozen-probe or last-four-block 2D/2.5D segmentation training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

ROOT = Path('/opt/data/3_117/tanghr')
RUN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'codex/mitotopo_20260908_round2/deps'))
sys.path.insert(0, str(ROOT/'DINOv3/dinov3'))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from seg_data import SegCrops, collate
from seg_eval import evaluate
from seg_model import SegModel, load_encoder

STOP = False


def stopped(*_):
    global STOP
    STOP = True


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    temp.replace(path)


def supervised_loss(logits, target):
    valid = target >= 0
    if not valid.any():
        raise RuntimeError('Empty valid supervision batch')
    ce = F.cross_entropy(logits.float(), target, ignore_index=-1, reduction='none')
    ce = (ce*valid).sum()/valid.sum()
    foreground = (target == 1).float()
    probability = logits.float().softmax(1)[:,1]
    intersection = (probability*foreground*valid).sum()
    soft_dice = 1-(2*intersection+1)/( (probability*valid).sum() + (foreground*valid).sum() + 1 )
    return ce+soft_dice, {'ce':float(ce.detach()), 'soft_dice_loss':float(soft_dice.detach())}


def last4_state(model):
    return {key:value.detach().cpu() for key,value in model.encoder.state_dict().items()
            if any(key.startswith(f'blocks.{i}.') for i in range(20,24))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=[0,1,2], required=True)
    parser.add_argument('--steps', type=int, choices=[10,10000,20000], required=True)
    parser.add_argument('--mode', choices=['2d','2p5d'], default='2d')
    parser.add_argument('--fine-tune-last4', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.steps == 20000 and not args.fine_tune_last4:
        raise SystemExit('20k budget requires last-four-block fine-tune')
    if args.steps == 10000 and args.fine_tune_last4:
        raise SystemExit('Frozen probe budget must be 10k')
    out = Path(os.environ['RUN_OUTPUT'])
    manifest = RUN/'data/seg_train.json'
    validation = RUN/'data/validation.json'
    identity = {'encoder':str(args.encoder.resolve(strict=True)), 'encoder_sha256':sha(args.encoder),
                'seed':args.seed,'steps':args.steps,'mode':args.mode,
                'fine_tune_last4':args.fine_tune_last4,'seg_manifest_sha256':sha(manifest),
                'validation_manifest_sha256':sha(validation),
                'decoder':'Primus PatchDecode four-scale, frequency gate identity',
                'normalization':'N2','batch':8,'ce_weight':1.0,'foreground_softdice_weight':1.0,
                'head_lr':1e-4,'encoder_lr':3e-6 if args.fine_tune_last4 else 0.0,
                'warmup_steps':1000,'weight_decay':0.05}
    atomic_json(out/'identity.json', identity)
    seed = 30_000 + args.seed
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    signal.signal(signal.SIGUSR1,stopped)
    signal.signal(signal.SIGTERM,stopped)
    encoder = load_encoder(ROOT/'DINOv3/dinov3',args.encoder)
    model = SegModel(encoder,args.seed,args.fine_tune_last4).cuda()
    head_init_sha = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()
                        for v in model.decoder.state_dict().values())).hexdigest()
    identity['head_init_sha256'] = head_init_sha
    atomic_json(out/'identity.json', identity)
    params = [{'params':list(model.decoder.parameters()),'base_lr':1e-4,'lr':1e-4}]
    if args.fine_tune_last4:
        last4 = [p for block in model.encoder.blocks[20:24] for p in block.parameters()]
        params.append({'params':last4,'base_lr':3e-6,'lr':3e-6})
    optimizer = torch.optim.AdamW(params, betas=(.9,.999),weight_decay=.05,foreach=True)
    resume = torch.load(args.resume,map_location='cpu',mmap=True,weights_only=False) if args.resume else None
    start = resume['completed_steps'] if resume else 0
    if resume:
        if resume['identity'] != identity:
            raise SystemExit('Segmentation resume identity mismatch')
        model.decoder.load_state_dict(resume['decoder'],strict=True)
        if args.fine_tune_last4:
            missing,unexpected=model.encoder.load_state_dict(resume['encoder_last4'],strict=False)
            if unexpected or any(any(k.startswith(f'blocks.{i}.') for i in range(20,24)) for k in missing):
                raise RuntimeError('Last-four-block resume mismatch')
        optimizer.load_state_dict(resume['optimizer'])
    chain = resume['consumption_chain'] if resume else '00'*32
    source_exposure = dict(resume['source_exposure']) if resume else {}
    best = resume['best_validation_dice'] if resume else -1.0
    best_step = resume['best_step'] if resume else None
    saved_rng = ({'torch':resume['torch_rng'],'cuda':resume['cuda_rng'],
                  'numpy':resume['numpy_rng'],'python':resume['python_rng']}
                 if resume else None)
    del resume
    dataset = SegCrops(manifest,args.seed,start*8,(args.steps-start)*8,args.mode)
    loader = DataLoader(dataset,batch_size=8,num_workers=2,prefetch_factor=1,
                        persistent_workers=False,pin_memory=False,
                        multiprocessing_context='spawn',collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed))
    if saved_rng:
        torch.set_rng_state(saved_rng['torch'])
        torch.cuda.set_rng_state_all(saved_rng['cuda'])
        np.random.set_state(saved_rng['numpy'])
        random.setstate(saved_rng['python'])
    model.train()
    completed=start
    started=time.monotonic()
    interval = 2000 if args.fine_tune_last4 else 1000
    status='running'
    last={}

    def checkpoint(path):
        payload = {'format_version':1,'identity':identity,'completed_steps':completed,
                   'decoder':{k:v.detach().cpu() for k,v in model.decoder.state_dict().items()},
                   'encoder_last4':last4_state(model) if args.fine_tune_last4 else {},
                   'optimizer':optimizer.state_dict(),
                   'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                   'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
                   'consumption_chain':chain,'source_exposure':source_exposure,
                   'best_validation_dice':best,'best_step':best_step,'status':status}
        temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)
        return payload

    with (out/'steps.jsonl').open('a') as trace:
        for step,batch in enumerate(loader,start):
            if STOP:
                status='interrupted';break
            locations=batch['locations']
            if [r['index'] for r in locations] != list(range(step*8,(step+1)*8)):
                raise RuntimeError('Paired sample index mismatch')
            chain=hashlib.sha256(bytes.fromhex(chain)+bytes.fromhex(batch['sample_digest'])).hexdigest()
            for r in locations:
                source_exposure[r['source']]=source_exposure.get(r['source'],0)+1
            image,target=batch['image'].cuda(),batch['target'].cuda()
            scale=min(1.0,(step+1)/1000)
            for group in optimizer.param_groups:
                group['lr']=group['base_lr']*scale
            optimizer.zero_grad(set_to_none=True)
            logits=model(image)
            loss,components=supervised_loss(logits,target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite supervised loss')
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']],12,
                                                error_if_nonfinite=True)
            optimizer.step()
            completed=step+1
            last={'step':completed,'loss':float(loss.detach()),'components':components,
                  'grad_norm':float(grad),'sample_chain':chain,'seconds':time.monotonic()-started,
                  'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
            trace.write(json.dumps(last)+'\n');trace.flush()
            if completed%25==0 or completed==args.steps:
                atomic_json(out/'progress.json',{'status':'running','target_steps':args.steps,**last})
                print(json.dumps(last),flush=True)
            if completed%interval==0 or completed==args.steps:
                if completed==args.steps:
                    status='complete'
                checkpoint(out/f'checkpoint_step{completed}.pth')
                if args.steps>10:
                    report=evaluate(model,validation,args.mode)
                    atomic_json(out/f'validation_step{completed}.json',report)
                    score=report['source_macro_dice']
                    if score>best:
                        best,best_step=score,completed
                        checkpoint(out/'head_best.pth')
                    checkpoint(out/f'checkpoint_step{completed}.pth')
                    model.train()
        else:
            status='complete'
    if completed and not (out/f'checkpoint_step{completed}.pth').exists():
        checkpoint(out/f'checkpoint_step{completed}.pth')
    atomic_json(out/'result.json',{'status':status,'completed_steps':completed,
                'target_steps':args.steps,'best_validation_dice':best,'best_step':best_step,
                'identity':identity,'source_exposure':source_exposure,**last})
    return 0 if status=='complete' else 75


if __name__=='__main__':
    raise SystemExit(main())
