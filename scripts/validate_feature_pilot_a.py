"""Independently reconcile completed Pilot A CSVs and server delivery hashes.

Reads small archived tables only. Does not access images, labels, weights or GPUs.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/Feature-Pilot-A"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(name):
    with (RUN / "tables" / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-10), (a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    manifest = read(ROOT / "evidence/Feature-Pilot-A/final_delivery_manifest.json")
    matched = 0
    review_only = []
    for item in manifest["files"]:
        rel = item["relative"]
        if rel.startswith("figures/") and rel.endswith(".png"):
            review_only.append(rel)
            continue
        if rel.startswith("evidence/"):
            path = ROOT / rel.replace("evidence/", "evidence/Feature-Pilot-A/", 1)
        elif rel == "Feature_Mechanism_Pilot_Report.md":
            path = RUN / "Feature_Mechanism_Pilot_Report_generated.md"
        else:
            path = RUN / rel
        assert path.is_file(), str(path)
        assert path.stat().st_size == item["bytes"], str(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], str(path)
        matched += 1

    protocol = read(RUN / "protocol.json")
    assert protocol == read(ROOT / "configs/evaluation/Feature_Pilot_A_protocol.json")
    samples = read(RUN / "samples.json")
    assert collections.Counter(x["role"] for x in samples) == {"train": 312, "eval": 321}
    assert len({x["region_id"] for x in samples if x["role"] == "train"}) == 156
    assert collections.Counter(x["source"] for x in samples if x["subset"] == "random") == {
        s: 64 for s in protocol["diagnostic_sources"]}
    assert len([x for x in samples if x["subset"] == "historical_D2_error_case"]) == 1
    models = [x["id"] for x in protocol["models"]]
    layers = [str(x) for x in protocol["layers"]] + ["Primus"]
    source_metrics = table("source_readout_metrics.csv")
    assert len(source_metrics) == 125
    by_key = {(x["model"], x["source"], x["layer"]): x for x in source_metrics}
    assert len(by_key) == 125
    pixel_count = 0
    gt_signatures = {}
    probe_batches = set()
    for model in models:
        receipt = read(RUN / "models" / model / "probe_complete.json")
        assert receipt["status"] == "complete" and receipt["steps_per_layer"] == 500
        assert receipt["evaluation_GT_read"] is False and receipt["encoder_asset_unchanged"]
        assert len(receipt["training_region_ids"]) == 156
        assert len(receipt["sources_with_both_classes"]) == 12
        probe_batches.add(receipt["coordinate_batch_hash"])
        pixels = table(f"pixel_metrics_{model}.csv")
        assert len(pixels) == 1605
        pixel_count += len(pixels)
        for source in protocol["diagnostic_sources"]:
            for layer in layers:
                rows = [x for x in pixels if x["source"] == source and x["layer"] == layer
                        and x["subset"] == "random"]
                assert len(rows) == 64
                agg = by_key[model, source, layer]
                counts = {k: sum(int(x[k]) for x in rows) for k in (
                    "tp", "fp", "fn", "valid_pixels", "far_background_pixels", "far_FP_pixels")}
                for k, v in counts.items():
                    assert int(agg[k]) == v, (model, source, layer, k)
                tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
                close(agg["dice"], 2 * tp / (2 * tp + fp + fn))
                close(agg["precision"], tp / (tp + fp))
                close(agg["recall"], tp / (tp + fn))
                close(agg["far_FP_rate"], counts["far_FP_pixels"] / counts["far_background_pixels"])
                signature = (tp + fn, counts["valid_pixels"], counts["far_background_pixels"])
                assert gt_signatures.setdefault(source, signature) == signature
    assert len(probe_batches) == 1

    sep = table("separability.csv")
    assert len(sep) == 300
    indexed = {(x["model"], x["source"], x["layer"], x["background_set"]): x for x in sep}
    assert len(indexed) == 300
    paired = table("D2_paired_AUROC.csv")
    assert len(paired) == 40
    for row in paired:
        a = indexed["D2-30k", row["source"], row["layer"], "all_far_background"]
        b = indexed["D2-100k", row["source"], row["layer"], "all_far_background"]
        field = row["score"] + "_AUROC"
        close(row["D2_30k_AUROC"], a[field])
        close(row["D2_100k_AUROC"], b[field])
        close(row["delta"], float(b[field]) - float(a[field]))
        assert int(row["bootstrap_valid_repeats"]) == 200

    spectra = table("spectrum.csv")
    assert len(spectra) == 300
    spectral_index = {(x["model"], x["source"], x["layer"], x["repeat"]): x for x in spectra}
    for model in models:
        distribution = table(f"spectrum_distribution_{model}.csv")
        assert len(distribution) == 60
        for row in distribution:
            values = json.loads(row["variance_fractions_descending"])
            assert len(values) == 512 and min(values) >= 0
            close(sum(values), 1)
            assert all(a >= b for a, b in zip(values, values[1:]))
            result = spectral_index[model, row["source"], row["layer"], row["repeat"]]
            close(result["effective_rank"], math.exp(-sum(p * math.log(p) for p in values if p > 0)))
            close(result["top1_explained_variance"], values[0])
            assert int(result["patches"]) == 512 and float(result["effective_rank"]) <= 511.000001

    registry = read(ROOT / "experiments/registry.yaml")
    old = json.loads(subprocess.check_output([
        "git", "-c", "safe.directory=" + ROOT.as_posix(), "show",
        "02a5ea093caf88bd4ca1ae30aba3d453ebeb7104:experiments/registry.yaml"], cwd=ROOT))
    old_rows = {x["experiment_id"]: x for x in old["experiments"]}
    new_rows = {x["experiment_id"]: x for x in registry["experiments"]}
    for key, row in old_rows.items():
        assert new_rows[key] == row, key
    assert set(new_rows) - set(old_rows) == {"Feature-Mechanism-Pilot-A"}
    result = {
        "status": "PASS", "archived_server_files_sha256_verified": matched,
        "PNG_review_artifacts_not_committed": review_only,
        "pixel_rows_reconciled": pixel_count, "pooled_source_rows_reconciled": len(source_metrics),
        "separability_rows": len(sep), "paired_AUROC_rows_reconciled": len(paired),
        "spectral_distributions_reconciled": len(spectra),
        "same_GT_counts_across_models_and_layers": True,
        "historical_error_case_excluded_from_pooled_metrics": True,
        "identical_train_minibatch_hash_all_models": sorted(probe_batches)[0],
        "unchanged_historical_registry_entries": len(old_rows),
        "scope": "CSV/hash reconciliation only; no new images, model computation or training"
    }
    if args.write:
        (ROOT / "evidence/Feature-Pilot-A/independent_validation.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
