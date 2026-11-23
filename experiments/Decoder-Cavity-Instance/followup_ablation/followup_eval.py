import hashlib
from itertools import product
from collections import defaultdict
import time

import numpy as np
import torch

from followup_model import OUT, PARENT, atomic, gpu_guard, load, read
from bc_postprocess import boundary_watershed
from bench_inputs import cem_items
from evaluate_new import cc, confusion, datasets, gt_maps, hq_row, interior, semantic, stats
from inference import load as load_parent, predict
import summarize


def write(name, rows):
    summarize.O = OUT
    summarize.write(name, rows)


def splits(*wanted):
    for split, group, key, planes in datasets():
        if split in wanted:
            yield split, group, key, planes


def aggregate(rows, fields, **label):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    result = []
    for group, items in sorted(groups.items()):
        result.append(dict(group=group, n=len(items), **label,
                           **{k: summarize.mean(items, k) for k in fields}))
    result.append(dict(group="ALL", n=len(rows), **label,
                       **{k: summarize.mean(result, k) for k in fields}))
    return result


@torch.no_grad()
def boundary_eval():
    start = time.monotonic()
    cache = OUT / "bc_fg_bnd_cache"
    cache.mkdir(exist_ok=True)
    arm = "arm_b_multiscale_bcd"
    record = read(PARENT / arm / "trained.json")
    options = read(OUT / "config.json")["bc_watershed"]
    grid = list(product(options["foreground_thresholds"], options["boundary_thresholds"],
                        options["minimum_seed_areas"]))
    trial = {p: [] for p in grid}
    selected = {
        (group, name)
        for group, images in cem_items()
        for name in sorted((item[0] for item in images),
                           key=lambda x: hashlib.sha256(("20261014/" + x).encode()).hexdigest())[
            :options["max_tuning_views_per_collection"]]
    }
    net = None
    gpu = None

    def probability(split, group, name, z, image):
        nonlocal net, gpu
        tag = hashlib.sha256(f"{split}/{group}/{name}/{z}".encode()).hexdigest()
        p = cache / (record["model_sha256"] + "_" + tag + ".npy")
        if p.is_file():
            cached = np.load(p, allow_pickle=False)
            assert cached.shape == (2, *image.shape) and np.isfinite(cached).all()
            return cached
        if net is None:
            gpu = gpu_guard()
            net, checked = load_parent(arm)
            assert checked["model_sha256"] == record["model_sha256"]
        maps = predict(net, image, "stride120")[:2]
        if (group, name) in selected:
            temp = p.with_suffix(".tmp")
            with temp.open("wb") as f:
                np.save(f, maps, allow_pickle=False)
            temp.replace(p)
        return maps

    for split, group, _, planes in splits("cem_validation"):
        for name, z, image, gt, valid, reliable in planes:
            if (group, name) not in selected:
                continue
            assert reliable and valid is None
            fg, bnd = probability(split, group, name, z, image)
            for p in grid:
                values = stats(gt, boundary_watershed(fg, bnd, *p))
                trial[p].append(dict(group=group, **values))
        print("CEM tuning", group, flush=True)
    assert all(len(v) == len(selected) for v in trial.values())
    scores = []
    for p, rows in trial.items():
        result = aggregate(rows, ("PQ", "AP75"))[-1]
        scores.append(dict(t_fg=p[0], t_bnd=p[1], min_size=p[2], **result))
    scores.sort(key=lambda r: (-r["PQ"], -r["AP75"], r["t_fg"], r["t_bnd"], r["min_size"]))
    params = tuple(scores[0][k] for k in ("t_fg", "t_bnd", "min_size"))
    lock = dict(params=params, selected=sorted(selected), model_sha256=record["model_sha256"])
    atomic(OUT / "bc_locked.json", lock)
    write("bc_watershed_cem_grid.csv", scores)
    previous = summarize.readcsv(PARENT / "instance_metrics.csv")
    result = []
    for split, n in (("cem_validation", 1424), ("mb1", 2595)):
        rows = []
        for _, group, _, planes in splits(split):
            for name, z, image, gt, valid, reliable in planes:
                assert reliable and valid is None
                fg, bnd = probability(split, group, name, z, image)
                metric = stats(gt, boundary_watershed(fg, bnd, *params))
                rows.append(dict(group=group, id=name, z=z, **metric))
            print("BC score", split, group, len(rows), flush=True)
        assert len(rows) == n
        table = aggregate(rows, summarize.I, split=split, method="bc_watershed")
        for row in table:
            group = row["group"]
            for key in ("touching_pairs", "separated_pairs"):
                row[key] = sum(v[key] for v in rows if group == "ALL" or v["group"] == group)
        for r in previous:
            if r["arm"] == arm and r["window"] == "stride120" and r["split"] == split:
                table.append({k: r[k] for k in table[0]})
        result.extend(table)
        atomic(OUT / f"bc_{split}_planes.json", rows)
    write("bc_watershed_summary.csv", result)
    atomic(OUT / "bc_complete.json", dict(gpu=gpu, seconds=time.monotonic() - start, params=params))


@torch.no_grad()
def no_raw_eval():
    start = time.monotonic()
    gpu = gpu_guard()
    net, record = load()
    mb1 = defaultdict(list)
    hq = []
    for split, group, key, planes in splits("mb1", "hq"):
        for name, z, image, gt, valid, reliable in planes:
            fg = predict(net, image, "stride120")[0] > 0.5
            if split == "hq":
                hq.append(dict(group=group, id=name, **hq_row(fg)))
                continue
            assert valid is None and reliable
            mb1[group].append(dict(id=name, z=z, semantic=semantic(confusion(fg, gt, valid)),
                                   cavity=interior(gt_maps(gt), fg)[0], cc=stats(gt, cc(fg))))
        print("NoRawSkip score", split, group, key, flush=True)
    assert len(mb1) == 7 and sum(map(len, mb1.values())) == 2595 and len(hq) == 20
    groups = []
    for group, rows in mb1.items():
        sem = {k: summarize.mean([v["semantic"] for v in rows], k) for k in summarize.S}
        inst = {k: summarize.mean([v["cc"] for v in rows], k) for k in summarize.I}
        groups.append(dict(group=group, n=len(rows), **sem, **inst, **summarize.cavity(rows)))
    fields = {"interior_fn_ratio", "closed_share", "missed_instance_rate", "large_cavity_grid_align"}
    macro = {k: summarize.mean(groups, k) for k in groups[0]
             if k in (*summarize.S, *summarize.I) or k.rsplit("_r", 1)[0] in fields}
    write("arm_a_norawskip_mb1_summary.csv", [dict(group="ALL", n=2595, **macro)])
    write("arm_a_norawskip_mb1_by_group.csv", groups)
    write("arm_a_norawskip_hq5nm_summary.csv",
          [dict(n=20, hole_ratio=summarize.mean(hq, "hole_ratio"),
                square_holes=sum(r["square_holes"] for r in hq))])
    atomic(OUT / "noraw_mb1_planes.json", mb1)
    atomic(OUT / "noraw_hq_planes.json", hq)
    atomic(OUT / "noraw_complete.json",
           dict(gpu=gpu, seconds=time.monotonic() - start, checkpoint=record["model_sha256"]))


if __name__ == "__main__":
    import sys

    torch.set_num_threads(2)
    entry = {"bc": boundary_eval, "noraw": no_raw_eval}
    entry[sys.argv[1]]()
