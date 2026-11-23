import json
import math
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from heads_model import CONFIG, OUT, PREVIOUS, build
from heads_data import CEMDraws
from heads_targets import init_diffusion, smoke_preview
from model import atomic, gpu_guard, read, semantic, sha


def semantic_bootstrap(logit, truth, valid):
    elements = F.binary_cross_entropy_with_logits(logit,truth,reduction="none")
    selected = []
    for sample, known in zip(elements,valid):
        values = sample[known]
        if values.numel():
            count = max(1,int(CONFIG["panoptic"]["top_fraction"]*len(values)))
            selected.append(values.topk(count,sorted=False).values.mean())
    p = logit.float().sigmoid()
    dice_loss = 1-(2*(p*truth*valid).sum()+1e-5)/((p*valid).sum()+(truth*valid).sum()+1e-5)
    return torch.stack(selected).mean() + dice_loss


def losses(arm, output, ids, target):
    valid = ids >= 0
    foreground = ids > 0
    truth = foreground.float()
    z = output[:,0]
    terms = {"fg": semantic_bootstrap(z,truth,valid) if arm == "panoptic" else semantic(z,truth,valid)}
    if arm == "panoptic":
        cfg = CONFIG[arm]
        terms["heatmap"] = ((output[:,1]-target[:,0]).square()*valid).sum()/valid.sum()
        terms["offset"] = ((output[:,2:]-target[:,1:]).abs()*foreground[:,None]).sum()/(2*foreground.sum().clamp_min(1))
        return terms["fg"]+cfg["heat_weight"]*terms["heatmap"]+cfg["offset_weight"]*terms["offset"],terms
    if arm == "cellpose":
        terms["flow"] = ((output[:,1:]-target).square()*foreground[:,None]).sum()/(2*foreground.sum().clamp_min(1))
        return terms["fg"]+CONFIG[arm]["flow_weight"]*terms["flow"],terms
    terms["boundary"] = semantic(output[:,1],target[:,0],valid)
    dist = F.smooth_l1_loss(output[:,2].sigmoid(),target[:,1],reduction="none")
    bg = valid & ~foreground
    terms["distance"] = (dist*foreground).sum()/foreground.sum().clamp_min(1)+.05*(dist*bg).sum()/bg.sum().clamp_min(1)
    return sum(terms.values()),terms


def safe_save(payload, path):
    tmp = path.with_suffix(".tmp")
    torch.save(payload,tmp)
    tmp.replace(path)


def train(arm, smoke):
    gpu = gpu_guard()
    torch.set_num_threads(2)
    model, helper = build(arm)
    hash_initial = helper.state_hash(model.enc.state_dict())
    encoder = OUT/CONFIG["encoder"]
    assert sha(encoder) == CONFIG["encoder_sha256"]
    d = OUT/arm
    d.mkdir(exist_ok=True)
    if arm == "cellpose":
        init_diffusion()
    draws = CEMDraws()
    if arm == "batch32":
        draws.verify_extended_sampler()
        old = read(PREVIOUS/"arm_b_multiscale_bcd/train_config.json")
        shared = {k:v for k,v in model.head.state_dict().items() if not k.startswith("aux.")}
        assert helper.state_hash(shared)==old["initial_shared_head_sha"]
    setup = CONFIG["heads"][arm]
    batch = setup["batch"]
    planned = setup["steps"]
    throughput = None
    if not smoke:
        check = read(d/"smoke_pass.json")
        assert check["head_init_sha256"] == helper.state_hash(model.head.state_dict())
        throughput = check["projected_40k_seconds"]
        if arm == "batch32" and throughput > setup["cap_seconds"]:
            planned = setup["fallback_steps"]
    if smoke:
        samples = [(i,*draws.sample(arm,i)) for i in (6,54,108,111,112)]
        smoke_preview(arm,samples,d/"smoke_targets")
    opt_cfg = CONFIG["optimizer"]
    opt = torch.optim.AdamW(model.head.parameters(),lr=setup["lr"],betas=tuple(opt_cfg["betas"]),
                            weight_decay=opt_cfg["weight_decay"])
    init_head = helper.state_hash(model.head.state_dict())
    record = dict(arm=arm,gpu=gpu,batch=batch,seed=CONFIG["seed"],exposures=0,steps_planned=100 if smoke else planned,
                  prefix_draw_sha256=draws.prefix_sha,head_init_sha256=init_head,training_image_count=draws.image_count,
                  lr=setup["lr"],projected_40k_seconds=throughput)
    start = 0
    prior_elapsed = 0
    ck = d/"state.pt"
    if not smoke and ck.is_file():
        state = torch.load(ck,weights_only=True,map_location="cpu")
        assert state["config"] == record
        model.head.load_state_dict(state["weights"])
        opt.load_state_dict(state["optimizer"])
        start = state["step"]
        prior_elapsed = state["elapsed_seconds"]
        torch.set_rng_state(state["rng_cpu"])
        torch.cuda.set_rng_state(state["rng_cuda"])
    torch.cuda.reset_peak_memory_stats()
    end = 100 if smoke else planned
    assert start < end
    values = []
    t0 = time.monotonic()
    t20 = t0
    for step, (raw, ids, target) in zip(range(start,end), draws.batches(
            arm,start,end,batch,160000 if smoke and arm=="batch32" else 0)):
        image = np.where(ids>=0,raw.astype(np.float32),127.5)
        image = torch.as_tensor((image/255-.5)/.25,device="cuda")[:,None]
        labels = torch.as_tensor(ids,device="cuda")
        truth = torch.as_tensor(target,device="cuda")
        output = model(image)
        assert output.shape == (batch,4 if arm=="panoptic" else 3,256,256)
        assert not model.enc.training
        loss, terms = losses(arm,output,labels,truth)
        warmup = min(1,(step+1)/opt_cfg["warmup_updates"])
        minimum = opt_cfg["min_lr_multiplier"]
        opt.param_groups[0]["lr"] = setup["lr"]*warmup*(minimum+(1-minimum)*.5*(1+math.cos(math.pi*step/planned)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        assert all(param.grad is None for param in model.enc.parameters())
        torch.nn.utils.clip_grad_norm_(model.head.parameters(),12,error_if_nonfinite=True)
        opt.step()
        value = float(loss.detach())
        assert math.isfinite(value)
        values.append(value)
        n = step+1
        now = time.monotonic()
        elapsed = prior_elapsed+now-t0
        if n == 20:
            t20 = now
        if n == 1 or n % (20 if smoke else 100) == 0 or n==end:
            progress = record | dict(step=n,exposures=n*batch,seconds=elapsed,loss=float(np.median(values[-100:])),
                                     peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                                     loss_components={k:float(v.detach()) for k,v in terms.items()})
            atomic(d/("smoke_progress.json" if smoke else "progress.json"),progress)
            print(json.dumps(progress),flush=True)
        if not smoke and n%2500==0 and n<end:
            safe_save(dict(step=n,elapsed_seconds=elapsed,weights=model.head.state_dict(),
                           optimizer=opt.state_dict(),config=record,rng_cpu=torch.get_rng_state(),
                           rng_cuda=torch.cuda.get_rng_state()),ck)
    assert hash_initial == helper.state_hash(model.enc.state_dict()) and sha(encoder)==CONFIG["encoder_sha256"]
    free, _ = torch.cuda.mem_get_info()
    final = progress | dict(frozen_encoder_verified=True,
                            headroom_bytes=free+torch.cuda.memory_reserved()-torch.cuda.max_memory_reserved())
    if smoke:
        first, last = float(np.median(values[:20])), float(np.median(values[-20:]))
        assert last < first and final["headroom_bytes"] >= 1.5*1024**3,(first,last,final["headroom_bytes"])
        assert (d/"smoke_targets/112.png").is_file()
        atomic(d/"smoke_pass.json",final | dict(first20_median=first,last20_median=last,
                                               projected_40k_seconds=(now-t20)*40000/(end-20)))
    else:
        path = d/"best.pt"
        safe_save(model.head.state_dict(),path)
        atomic(d/"trained.json",final | dict(weights_sha256=sha(path),
                                            exposure_equivalent_epochs=n*batch/draws.image_count))


if __name__ == "__main__":
    mode, arm = sys.argv[1:]
    assert mode in ("smoke","train") and arm in CONFIG["heads"]
    train(arm,mode=="smoke")
