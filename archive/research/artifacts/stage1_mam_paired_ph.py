"""Compute matched raw-EM and SAM-MAM PH distances after CPU map generation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from topology_raw_ph import TOP, diagrams, distance


PAIRED = TOP / "mam_sam_vit_b" / "paired"
NAMES = ["identity", "flip_horizontal", "flip_vertical", "weak_jitter", "strong_jitter", "weak_blur", "strong_blur", "mild_crop", "medium_crop", "aggressive_crop"]
MODES = ["raw_em_paired", "mam_affinity", "mam_boundary"]


def main() -> None:
    manifest = json.loads((TOP / "sample_manifest.json").read_text())
    samples = manifest["samples"]
    complete = json.loads((PAIRED / "generation_complete.json").read_text())
    assert complete["status"] == "complete" and complete["sample_count"] == len(samples) == 424
    assert complete["augmentation_names"] == NAMES
    temporary = PAIRED / "metrics_paired.csv.tmp"
    target = PAIRED / "metrics_paired.csv"
    columns = ["sample_index", "volume_id", "source", "slice_index", "input", "augmentation", "d0", "d1", "h0_count", "h1_count", "h0_aug_count", "h1_aug_count"]
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index, sample in enumerate(samples):
            with np.load(PAIRED / "maps" / f"sample_{index:06d}.npz") as package:
                raw = package["raw"]
                mam = package["mam"]
            assert raw.shape == mam.shape == (10, 64, 64)
            for mode, images in zip(MODES, (raw, mam, 1 - mam)):
                original = diagrams(images[0])
                for name, image in zip(NAMES, images):
                    current = diagrams(image)
                    writer.writerow({
                        "sample_index": index, "volume_id": sample["volume_id"], "source": sample["source"],
                        "slice_index": sample["slice_index"], "input": mode, "augmentation": name,
                        "d0": distance(original[0], current[0]), "d1": distance(original[1], current[1]),
                        "h0_count": len(original[0]), "h1_count": len(original[1]),
                        "h0_aug_count": len(current[0]), "h1_aug_count": len(current[1]),
                    })
            if (index + 1) % 10 == 0:
                stream.flush()
                (PAIRED / "ph_progress.json").write_text(json.dumps({"completed_samples": index + 1, "total_samples": len(samples)}))
    temporary.replace(target)
    (PAIRED / "ph_complete.json").write_text(json.dumps({"status": "complete", "samples": len(samples), "rows": len(samples) * len(NAMES) * len(MODES), "modes": MODES, "label_free": True}, indent=2))
    print(json.dumps({"rows": len(samples) * len(NAMES) * len(MODES), "path": str(target)}))


if __name__ == "__main__":
    main()
