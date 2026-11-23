from collections import defaultdict
from functools import partial
import hashlib
from itertools import product
import json
import statistics
import sys
import time

import numpy as np
import torch

from heads_model import CONFIG, OUT, load
from heads_decode import decoded_grid, predict
from bench_inputs import cem_items, cem_read, datasets
from evaluate_new import cc, confusion, gt_maps, hq_row, interior, semantic, stats
from model import atomic, gpu_guard, read, sha
import summarize

INSTANCE_KEYS = ("AP50","AP75","PQ","MAE")
INSTANCE_COUNTS = ("touching_pairs","separated_pairs")


def choose(arm, infer, directory):
    lock = directory/"locked.json"
    lineage = dict(weights_sha256=sha(directory/"best.pt"),
                   config_sha256=sha(OUT/"config.json"),decode_sha256=sha(OUT/"heads_decode.py"))
    if lock.exists():
        receipt = read(lock)
        assert receipt["lineage"]==lineage
        return tuple(receipt["params"])
    selected = []
    settings = list(product(*CONFIG["grid"][arm]))
    table = {option: defaultdict(list) for option in settings}
    count = CONFIG["cem_selection"]["max_per_collection"]
    for group, options in cem_items():
        chosen = sorted(options,key=lambda r:hashlib.sha256(("20261014/"+r[0]).encode()).hexdigest())[:count]
        for key, _ in chosen:
            image, gt = cem_read(group,key)
            maps = tile_cached(infer,directory,group,key,image,create=True)
            for params, ids in decoded_grid(arm,maps,settings):
                result = stats(gt,ids)
                table[params][group].append((result["PQ"],result["AP75"]))
            selected.append((group,key))
        print("CEM tuning",group,len(selected),flush=True)
    rows = []
    for (t_fg, control, size), groups in table.items():
        val = [statistics.mean(stat[j] for stat in group) for j in (0,1) for group in groups.values()]
        k = len(groups)
        rows.append(dict(t_fg=t_fg,control=control,min_size=size,n=len(selected),
                         PQ=statistics.mean(val[:k]),AP75=statistics.mean(val[k:])))
    rows.sort(key=lambda r:(-r["PQ"],-r["AP75"],r["t_fg"],r["control"],r["min_size"]))
    summarize.O = OUT
    summarize.write(arm+"_cem_grid.csv",rows)
    params = tuple(rows[0][k] for k in ("t_fg","control","min_size"))
    atomic(lock,dict(params=params,examples=selected,lineage=lineage))
    print("CEM locked",params,flush=True)
    return params


def tile_cached(infer,directory,group,key,image,create=False):
    cache = directory/"tuning_predictions"
    tag = hashlib.sha256((group+"/"+key).encode()).hexdigest()
    path = cache/(tag+".npy")
    if path.is_file():
        maps = np.load(path,allow_pickle=False)
        assert maps.shape[1:]==image.shape and np.isfinite(maps).all()
        return maps
    maps = infer(image)
    if create:
        cache.mkdir(exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            np.save(f,maps,allow_pickle=False)
        tmp.replace(path)
    return maps


def score_plane(arm, maps, gt, split, params):
    fg = maps[0] > CONFIG["inference"]["semantic_threshold"]
    if split=="hq":
        return hq_row(fg)
    value = dict(semantic=semantic(confusion(fg,gt,None)))
    if split=="mb1":
        value["cavity"] = interior(gt_maps(gt),fg)[0]
    value["native"] = stats(gt,next(decoded_grid(arm,maps,[params]))[1])
    value["cc"] = stats(gt,cc(fg))
    alternate = maps[0] > params[0]
    value["cc_matched"] = value["cc"] if params[0]==0.5 else stats(gt,cc(alternate))
    return value


def run_units(arm,infer,directory,params):
    base = directory/"eval_units"
    base.mkdir(exist_ok=True)
    for split, group, key, planes in datasets():
        if split not in ("cem_validation","mb1","hq"):
            continue
        target = base/(split+"__"+key+".jsonl")
        if target.exists():
            continue
        part = target.with_suffix(".part")
        previous = len(part.read_text().splitlines()) if part.exists() else 0
        with part.open("a") as handle:
            for i,(name,z,image,gt,valid,instance) in enumerate(planes):
                if i<previous:
                    continue
                assert split=="hq" or (valid is None and instance)
                maps = tile_cached(infer,directory,group,name,image) if split=="cem_validation" else infer(image)
                row = score_plane(arm,maps,gt,split,params)
                handle.write(json.dumps(dict(id=name,z=z,group=group,split=split,**row))+"\n")
                handle.flush()
                if i % 100 == 0 and split!="hq":
                    print(arm,split,group,key,i+1,flush=True)
        part.replace(target)
        print("DONE",arm,split,group,key,flush=True)


def group_summary(arm,directory):
    grouped = defaultdict(list)
    for f in sorted((directory/"eval_units").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            grouped[r["split"],r["group"]].append(r)
    ids = defaultdict(set)
    for (split,group), rows in grouped.items():
        for row in rows:
            key = (group,row["id"],row["z"])
            assert key not in ids[split]
            ids[split].add(key)
    for split,count in (("cem_validation",1424),("mb1",2595),("hq",20)):
        assert len(ids[split]) == count
    semantic_rows, instance_rows, hq = [], [], []
    for (split, group), rows in sorted(grouped.items()):
        tag = dict(arm=arm,split=split,group=group,n=len(rows))
        if split=="hq":
            hq.extend(rows)
            continue
        s = [row["semantic"] for row in rows]
        semantic_rows.append(dict(**tag,**{k:summarize.mean(s,k) for k in summarize.S},**summarize.cavity(rows)))
        for method in ("native","cc","cc_matched"):
            v = [row[method] for row in rows]
            instance_rows.append(dict(**tag,method=method,
                **{k:summarize.mean(v,k) for k in INSTANCE_KEYS},
                **{k:sum(item[k] for item in v) for k in INSTANCE_COUNTS}))
    for split in ("cem_validation","mb1"):
        relevant = [r for r in semantic_rows if r["split"]==split]
        tag = dict(arm=arm,split=split,group="ALL",n=sum(r["n"] for r in relevant))
        semantic_rows.append(dict(**tag,**{k:summarize.mean(relevant,k) for k in relevant[0]
            if k in summarize.S or k.rsplit("_r",1)[0] in ("interior_fn_ratio","closed_share","open_share",
                "large_cavity_grid_align","missed_instance_rate","affected_instance_rate")}))
        for method in ("native","cc","cc_matched"):
            r = [v for v in instance_rows if v["split"]==split and v["method"]==method]
            instance_rows.append(dict(**tag,method=method,**{k:summarize.mean(r,k) for k in INSTANCE_KEYS},
                                      **{k:sum(v[k] for v in r) for k in INSTANCE_COUNTS}))
    summarize.O = OUT
    summarize.write(arm+"_semantic.csv",semantic_rows)
    summarize.write(arm+"_instance.csv",instance_rows)
    summarize.write(arm+"_hq.csv",[dict(arm=arm,n=len(hq),hole_ratio=summarize.mean(hq,"hole_ratio"),
                                     square_holes=sum(row["square_holes"] for row in hq))])


@torch.no_grad()
def main(arm):
    start = time.monotonic()
    gpu = gpu_guard()
    torch.set_num_threads(2)
    net = load(arm)
    directory = OUT/arm
    infer = partial(predict,net,arm=arm)
    params = choose(arm,infer,directory)
    run_units(arm,infer,directory,params)
    group_summary(arm,directory)
    atomic(directory/"evaluated.json",dict(gpu=gpu,seconds=time.monotonic()-start,params=params))


if __name__=="__main__":
    main(sys.argv[1])
