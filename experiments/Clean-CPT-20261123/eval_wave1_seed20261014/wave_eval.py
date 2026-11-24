"""Wave 1 fixed split evaluation and external local-MitoEM overlay."""
import argparse
from functools import partial
import json
from pathlib import Path
import shutil

from wave_train_v1 import BASE, HIST, MODELS, ROOT, configure, gpu_limit


def external_planes():
    import numpy as np
    import zarr

    root = BASE.parent / "MitoVerse_data"
    volumes = json.loads((root / "splits/mitoem2.0.json").read_text())["volumes"]
    known = {r["volume_id"] for r in json.loads(
        (ROOT.parent / "clean_cpt_manifest.json").read_text())["mitoverse"]["volumes"]}
    for row in volumes:
        if row["split"] not in ("val", "test"):
            continue
        name = row["id"]
        assert name not in known
        z = zarr.open_group(str(root / row["zarr"]), mode="r")
        assert z["img"].shape == z["mito"].shape
        for i in range(z["img"].shape[0]):
            valid = np.asarray(z["mask"][i]) != 0 if "mask" in z else None
            if valid is not None and not valid.any():
                continue
            yield row, i, np.asarray(z["img"][i]), np.asarray(z["mito"][i]), valid


def install_metrics(ev):
    import evaluate_new

    def stats(gt, ids):
        x = evaluate_new.score(gt, ids)
        pairs, separated = evaluate_new.touching(gt, ids)
        return dict(PQ=float(x["PQ"]), AP50=float(x["AP_50"]), AP75=float(x["AP_75"]),
                    F1_50=float(x["F1_50"]), F1_75=float(x["F1_75"]),
                    MAE=int(abs(x["n_gt"] - x["n_pred"])),
                    touching_pairs=pairs, separated_pairs=separated)

    ev.stats = stats
    ev.INSTANCE_KEYS = (*ev.INSTANCE_KEYS, "F1_50", "F1_75")
    return stats


def external(out, scorer):
    from collections import defaultdict
    import csv
    import statistics

    path = out / "mitoem2_planes.jsonl"
    if not path.exists():
        part = path.with_suffix(".part")
        done = set()
        if part.is_file():
            with part.open() as f:
                for line in f:
                    row = json.loads(line)
                    done.add((row["volume"], row["z"]))
        with part.open("a", buffering=1) as f:
            for v, z, image, gt, valid in external_planes():
                identifier = (v["id"], z)
                if identifier not in done:
                    score = scorer(image, gt, valid)
                    row = dict(partition=v["split"], subset=v["me2"], volume=v["id"], z=z, **score)
                    f.write(json.dumps(row, allow_nan=False) + "\n")
        part.replace(path)
    with path.open() as f:
        scores = [json.loads(line) for line in f]
    cases = defaultdict(list)
    for x in scores:
        cases[x["partition"], x["subset"], x["volume"]].append(x)
    fields = ("Dice", "IoU", "PQ", "AP50", "AP75", "F1_50", "F1_75", "MAE",
              "touching_pairs", "separated_pairs")
    rows = []
    for (split, subset, volume), entries in sorted(cases.items()):
        rows.append(dict(partition=split, subset=subset, volume=volume, n=len(entries),
                         **{k:(sum(x[k] for x in entries) if k.endswith("_pairs")
                              else statistics.mean(x[k] for x in entries)) for k in fields}))
    with (out / "mitoem2_by_volume.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(model, arm, index):
    gpu_limit(index, arm)
    import numpy as np
    import torch
    import time

    trainer, build, run, cfg = configure(model, arm)
    import evaluate_heads as ev
    import bench_inputs
    import heads_data
    from wave_cem_replay import patch

    with torch.no_grad():
        torch.set_num_threads(2)
        out = run / arm
        record = trainer.read(out / "trained.json")
        assert record["step"] == 40000 and record["frozen_encoder_verified"]
        assert record["weights_sha256"] == trainer.sha(out / "best.pt")
        net, _ = build(arm)
        net.head.load_state_dict(torch.load(out / "best.pt", weights_only=True, map_location="cpu"), strict=True)
        net.eval()
        for f in ("config.json", "heads_decode.py"):
            dst = out / f
            if not dst.exists():
                shutil.copyfile(HIST / f, dst)
            assert trainer.sha(dst) == trainer.sha(HIST / f)
        ev.OUT = out
        ev.CONFIG = cfg
        if not Path("/").joinpath("opt", "cem_mitolab_20261118").is_dir():
            bench_inputs.cem_read, _ = patch(trainer, heads_data)
            ev.cem_read = bench_inputs.cem_read
        infer = partial(ev.predict, net, arm=arm)
        start = time.monotonic()
        params = ev.choose(arm, infer, out)
        examples = trainer.read(out / "locked.json")["examples"]
        assert len(examples) == len({tuple(x) for x in examples}) == 175
        stats = install_metrics(ev)
        ev.run_units(arm, infer, out, params)
        ev.group_summary(arm, out)

        def score(image, gt, valid):
            maps = infer(image)
            fg = maps[0] > cfg["inference"]["semantic_threshold"]
            ids = next(ev.decoded_grid(arm, maps, [params]))[1]
            m = ev.semantic(ev.confusion(fg, gt, valid))
            if valid is not None:
                gt, ids = np.where(valid, gt, 0), np.where(valid, ids, 0)
            return dict(Dice=m["dice"], IoU=m["iou"], **stats(gt, ids))

        external(out, score)
        trainer.atomic(out / "evaluated.json", dict(gpu=index, params=params, examples=175,
                       seconds=time.monotonic()-start, encoder_sha256=cfg["encoder_sha256"]))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("model", choices=MODELS)
    args.add_argument("arm", choices=("panoptic","cellpose"))
    args.add_argument("index", type=int, choices=(1,2,3))
    cli = args.parse_args()
    assert cli.index == list(MODELS).index(cli.model) + 1
    evaluate(cli.model, cli.arm, cli.index)
