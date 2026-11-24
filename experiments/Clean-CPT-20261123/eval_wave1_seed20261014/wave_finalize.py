"""Combine the six Wave 1 readouts with the same-protocol MitoNet baseline."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from wave_train_v1 import HIST, MODELS, ROOT


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write(path, records):
    keys = list(dict.fromkeys(k for r in records for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(records)


def gather(directory, prefix, model, head):
    semantic = rows(directory / f"{prefix}_semantic.csv")
    instance = [r for r in rows(directory / f"{prefix}_instance.csv") if r["method"] == "native"]
    hq, = rows(directory / f"{prefix}_hq.csv")
    assert int(hq["n"]) == 20
    grouped = {}
    for r in semantic:
        key = r["split"], r["group"]
        assert key not in grouped
        grouped[key] = {k: v for k, v in r.items() if k not in ("arm","split","group")}
    for r in instance:
        key = r["split"], r["group"]
        assert grouped[key]["n"] == r["n"]
        grouped[key].update({k:v for k,v in r.items() if k not in ("arm","split","group","method","n")})
    assert int(grouped["mb1","ALL"]["n"]) == 2595 and int(grouped["cem_validation","ALL"]["n"]) == 1424
    groups = [dict(model=model, head=head, split=s, group=g, **r) for (s,g),r in grouped.items()]
    mb, muscle, cem = (grouped[key] for key in (("mb1","ALL"),("mb1","glycolytic_muscle"),
                                                ("cem_validation","ALL")))
    main = dict(model=model,head=head,Dice=mb["dice"],IoU=mb["iou"],
                interior_r3=mb["interior_fn_ratio_r3"],PQ=mb["PQ"],AP50=mb["AP50"],
                AP75=mb["AP75"],MAE=mb["MAE"],muscle_Dice=muscle["dice"],muscle_PQ=muscle["PQ"],
                CEM_val_PQ=cem["PQ"],HQ_square_holes=hq["square_holes"],
                HQ_hole_ratio=hq["hole_ratio"])
    return groups, main


def finalize():
    main, by_group, external, records = [], [], [], []
    for model in MODELS:
        for arm, head in (("panoptic","Head C"),("cellpose","Head D")):
            d = ROOT / "runs" / model / arm
            assert (d/"evaluated.json").is_file()
            train = json.loads((d/"trained.json").read_text())
            assert train["step"] == 40000 and train["frozen_encoder_verified"]
            assert train["batch"] == 4 and train["seed"] == 20261014
            groups, row = gather(d, arm, model, head)
            by_group += groups
            main.append(row)
            external.extend(dict(model=model,head=head,**v) for v in rows(d/"mitoem2_by_volume.csv"))
            records.append(dict(model=model,head=head,gpu=train["gpu"],seconds=train["seconds"],
                                peak_reserved_bytes=train["peak_reserved"],fit_steps=train["step"]))
    official = "Official-MitoNet-v1 2D"
    groups, row = gather(HIST, "mitonet_v1", official, "Native")
    by_group += groups
    main.append(row)
    d = ROOT / "mitonet_v1_2d"
    assert (d/"evaluated.json").is_file()
    external.extend(dict(model=official,head="Native",**v) for v in rows(d/"mitoem2_by_volume.csv"))
    assert len(external) == 77
    write(ROOT/"wave1_main_table.csv",main)
    write(ROOT/"wave1_mb1_by_group.csv",[r for r in by_group if r["split"]=="mb1"])
    write(ROOT/"wave1_cem_val_by_group.csv",[r for r in by_group if r["split"]=="cem_validation"])
    write(ROOT/"wave1_external_by_volume.csv",external)
    write(ROOT/"wave1_training.csv",records)
    buckets = defaultdict(list)
    for r in external:
        buckets[r["model"],r["head"],r["partition"]].append(r)
    agg = []
    fields = ("Dice","IoU","PQ","AP50","AP75","F1_50","F1_75","MAE")
    for (model,head,split), volume in buckets.items():
        agg.append(dict(model=model,head=head,partition=split,n_volumes=len(volume),
                        n_planes=sum(int(x["n"]) for x in volume),
                        **{k:statistics.mean(float(x[k]) for x in volume) for k in fields}))
    write(ROOT/"wave1_external_summary.csv",agg)
    config = dict(seed=20261014, fit_steps=40000,batch=4,backbones=MODELS,
        training="CEM-MitoLab train; frozen DINOv3; audited batch4 draw ledger",
        tuning="CEM-MitoLab val 175 views only", held_out=["MB-1", "HQ-5nm", "MitoEM 2.0 local overlay"],
        inference=dict(tiling=256,stride=120,fusion="Gauss",padding="reflect"),
        external=dict(skip_unlabeled_han_planes=True,han_mask="nonzero",
                      cellmap_test_gt="unavailable locally", mitoem2_pyra="no local test ROI"))
    (ROOT/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    lines = ["# Clean-CPT Wave 1 main table", "",
             "All six heads used seed 20261014, batch 4 and 40,000 steps. Postprocessing used 175 CEM-val "
             "views only. Main MB-1 metrics use an equal mean of seven groups. The released native MitoNet "
             "reference for MB-1, CEM-val and HQ is from the benchmark's matching evaluation.",
             "",
             "| Backbone | Head | Dice | IoU | Interior miss r3 | PQ | AP50 | AP75 | MAE | Muscle Dice | Muscle PQ | CEM-val PQ | HQ square holes |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in main:
        lines.append("| " + " | ".join([row["model"],row["head"],*(
            f"{float(row[k]):.5f}" for k in ("Dice","IoU","interior_r3","PQ","AP50","AP75","MAE","muscle_Dice",
                                           "muscle_PQ","CEM_val_PQ")),str(row["HQ_square_holes"])])+" |")
    lines.extend(["", "## External local MitoEM 2.0 overlay", "",
                  "The test summary gives each scored volume equal weight. Han24 zeros outside its annotated "
                  "ROI are ignored. These volumes are excluded by exact identifier from Clean-CPT's registered "
                  "MitoVerse training manifest. This does not establish P0 or all CEM image provenance. "
                  "No local official CellMap test GT was verified.", "",
                  "| Backbone | Head | Split | Volumes | Scored planes | Dice | PQ |",
                  "|---|---|---|---:|---:|---:|---:|"])
    for r in agg:
        lines.append(f"| {r['model']} | {r['head']} | {r['partition']} | {r['n_volumes']} | "
                     f"{r['n_planes']} | {r['Dice']:.5f} | {r['PQ']:.5f} |")
    (ROOT/"WAVE1_MAIN_TABLE_SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    finalize()
