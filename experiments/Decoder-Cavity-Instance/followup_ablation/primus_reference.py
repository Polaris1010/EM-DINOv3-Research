import csv
import importlib.util
import time

import torch

from followup_model import OUT, PARENT, atomic, gpu_guard
from followup_eval import aggregate, cc, splits, stats
from panels import old_model
import summarize


@torch.no_grad()
def main():
    start = time.monotonic()
    gpu = gpu_guard()
    torch.set_num_threads(2)
    phase = PARENT.parent / "square_hole_diagnosis_20261122" / "full_phase_eval"
    spec = importlib.util.spec_from_file_location("phase_reference", phase / "run.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    with (phase / "per_image.csv").open(newline="") as f:
        old = {(r["id"], int(r["z"])): tuple(int(r[k]) for k in ref.CHK)
               for r in csv.DictReader(f)
               if r["dataset"] == "MB1" and r["config"] == "Gauss + Stride-120"}
    net, probe = old_model()
    rows = []
    visited = set()
    for _, group, key, planes in splits("mb1"):
        for name, z, image, gt, valid, reliable in planes:
            assert reliable and valid is None
            _, probs, _, _ = ref.measure_pass(net, probe, image, 120, (0, 0))
            mask = probs > 0.5
            expected = old[name, z]
            observed = ref.sem(mask, gt > 0)
            assert tuple(observed[k] for k in ref.CHK) == expected, (name, z)
            rows.append(dict(group=group, id=name, z=z, **stats(gt, cc(mask))))
            visited.add((name, z))
        print("Primus Stride-120 CC", key, len(rows), flush=True)
    assert len(rows) == len(visited) == len(old) == 2595
    assert visited == old.keys()
    table = aggregate(rows, summarize.I, model="C1 Primus", window="stride120")
    atomic(OUT / "primus_reference.json",
           dict(gpu=gpu, seconds=time.monotonic() - start, confusion_verified=2595, rows=table))


if __name__ == "__main__":
    main()
