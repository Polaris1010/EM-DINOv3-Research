import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from heads_model import PREVIOUS, OUT
from evaluate_heads import group_summary
from evaluate_new import cc, confusion, datasets, gt_maps, hq_row, interior, semantic, stats
from model import MB, atomic, gpu_guard, read, sha
from panels import old_model
from common import config
import summarize


def phase_code():
    p = PREVIOUS.parent/"square_hole_diagnosis_20261122"/"full_phase_eval"/"run.py"
    spec = importlib.util.spec_from_file_location("phase_reference",p)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    return ref


@torch.no_grad()
def primus():
    ref = phase_code()
    net, helper = old_model()
    scores = []
    ids = set()
    for split, group, _, planes in datasets():
        if split!="cem_validation":
            continue
        items = []
        for name,z,image,gt,valid,reliable in planes:
            assert reliable and valid is None and (group,name,z) not in ids
            _, maps, _, _ = ref.measure_pass(net,helper,image,120,(0,0))
            items.append(stats(gt,cc(maps>.5)))
            ids.add((group,name,z))
        metrics = {k:summarize.mean(items,k) for k in ("AP50","AP75","PQ","MAE")}
        scores.append(dict(group=group,n=len(items),**metrics))
        print("C1 Primus CEM",group,len(ids),flush=True)
    assert len(ids)==1424
    scores.append(dict(group="ALL",n=1424,**{k:summarize.mean(scores,k) for k in ("AP50","AP75","PQ","MAE")}))
    summarize.O = OUT
    summarize.write("primus_cem_reference.csv",scores)


@torch.no_grad()
def mitonet():
    from infer import mitonet as official_engine

    opts = next(r for r in config()["runs"] if r["model"]=="Official-MitoNet-v1")
    checkpoint = opts["checkpoint_sha256"]
    assert checkpoint==sha(Path(opts["checkpoint_path"]))
    old = MB/"runs"/"M"
    can_reuse = read(old/"inference_complete.json")["checkpoint_sha256"]==checkpoint
    engine = official_engine(opts)
    directory = OUT/"mitonet_reference"
    base = directory/"eval_units"
    base.mkdir(parents=True,exist_ok=True)
    for split, group, key, planes in datasets():
        if split not in ("cem_validation","mb1","hq"):
            continue
        out = base/(split+"__"+key+".jsonl")
        if out.exists():
            continue
        cached = None
        if split=="mb1" and can_reuse:
            assert read(old/(key+"_inference.json"))["checkpoint_sha256"]==checkpoint
            cached = np.load(old/(key+"_panoptic.npy"),mmap_mode="r")
        tmp = out.with_suffix(".part")
        with tmp.open("w") as f:
            for name,z,image,gt,valid,reliable in planes:
                pan = np.asarray(engine(image) if cached is None else cached[z])
                assert pan.shape==image.shape
                fg = pan > 0
                row = dict(id=name,z=z,group=group,split=split)
                if split=="hq":
                    row.update(hq_row(fg))
                else:
                    assert reliable and valid is None
                    if split=="mb1":
                        row["cavity"] = interior(gt_maps(gt),fg)[0]
                    c = stats(gt,cc(fg))
                    row.update(semantic=semantic(confusion(fg,gt,None)),
                               native=stats(gt,pan),cc=c,cc_matched=c)
                f.write(json.dumps(row)+"\n")
        tmp.replace(out)
        print("MitoNet",split,group,key,flush=True)
    group_summary("mitonet_v1",directory)


if __name__=="__main__":
    start = time.monotonic()
    gpu = gpu_guard()
    torch.set_num_threads(2)
    name = sys.argv[1]
    {"primus":primus,"mitonet":mitonet}[name]()
    atomic(OUT/(name+"_reference.json"),dict(gpu=gpu,seconds=time.monotonic()-start))
