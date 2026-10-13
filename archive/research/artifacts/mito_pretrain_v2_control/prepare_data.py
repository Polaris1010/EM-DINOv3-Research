"""Prepare the locked MitoVerse v2 volume manifests from the archived source split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    src = args.source_manifest.resolve(strict=True)
    out = args.output_dir.resolve(strict=True)
    if any(out.iterdir()):
        raise SystemExit(f"Output directory is not empty: {out}")

    rows = list(csv.DictReader(src.open(encoding="utf-8-sig", newline="")))
    if len(rows) != 212:
        raise SystemExit(f"Unexpected source manifest length: {len(rows)}")

    active = [r for r in rows if r["split"] in {"train", "validation", "legacy_dev_test"}]
    seen_ids: set[str] = set()
    seen_paths: dict[Path, str] = {}
    known_scans: dict[tuple[str, str], str] = {}
    for row in active:
        vid = row["volume_id"]
        if vid in seen_ids:
            raise SystemExit(f"Duplicate volume ID: {vid}")
        seen_ids.add(vid)
        p = Path(row["local_path"]).resolve(strict=True)
        if p in seen_paths:
            raise SystemExit(f"Path shared by {seen_paths[p]} and {vid}")
        seen_paths[p] = vid
        image_meta = json.loads((p / "img" / ".zarray").read_text())
        label_meta = json.loads((p / "mito" / ".zarray").read_text())
        shape = image_meta["shape"]
        if len(shape) != 3 or shape != label_meta["shape"]:
            raise SystemExit(f"Image/label shape mismatch: {vid}")
        mask_meta = p / "mask" / ".zarray"
        if mask_meta.exists() and json.loads(mask_meta.read_text())["shape"] != shape:
            raise SystemExit(f"Mask shape mismatch: {vid}")
        if row["axes"] != "ZYX":
            raise SystemExit(f"Unexpected axis order: {vid}")
        if row["original_volume_id"]:
            key = (row["original_dataset_id"], row["original_volume_id"])
            if key in known_scans and known_scans[key] != row["split"]:
                raise SystemExit(f"Known scan crosses split: {key}")
            known_scans[key] = row["split"]
        row["verified_shape_zyx"] = shape
        row["image_dtype"] = image_meta["dtype"]
        row["label_dtype"] = label_meta["dtype"]
        row["verified_mask_present"] = mask_meta.exists()

    groups = {
        "pretrain_train": [r for r in active if r["split"] == "train"],
        "seg_train": [r for r in active if r["split"] == "train" and r["original_dataset_id"] != "han24"],
        "validation": [r for r in active if r["split"] == "validation"],
        "legacy_dev_test": [r for r in active if r["split"] == "legacy_dev_test"],
    }
    expected = {"pretrain_train": 31, "seg_train": 26, "validation": 8, "legacy_dev_test": 16}
    for name, group in groups.items():
        if len(group) != expected[name]:
            raise SystemExit(f"Unexpected {name} count: {len(group)}")

    shutil.copyfile(src, out / "source_manifest_v3.csv")
    for name, group in groups.items():
        write_json(out / f"{name}.json", group)

    (out / "data_notes.md").write_text(
        "# MitoVerse v2 data notes\n\n"
        "The archived round6 source split is retained: 31 train, 8 validation, "
        "16 previously viewed legacy_dev_test and 157 excluded volumes. "
        "The 31 train images alone form the SSL pool; labels and masks are not "
        "used to choose SSL crops. All five han24 volumes are excluded from "
        "supervised training until the valid-mask production definition is "
        "confirmed. Their eligible train images remain in the SSL pool. "
        "Original specimen/scan/ROI identities remain uncertain for some volumes; "
        "this protocol supports source-grouped development comparisons only. "
        "No claim of strictly unseen historical data is made.\n\n"
        "Sampling: source uniformly, then volume uniformly within source, "
        "then a legal slice and XY crop. The training reader must implement "
        "this plan without filtering by segmentation labels.\n\n"
        "Normalization N2: input uint8 / 255 in float32, copy grayscale to "
        "three channels, then subtract (0.485, 0.456, 0.406) and divide by "
        "(0.229, 0.224, 0.225), channelwise. This matches archived "
        "round5/r4_common.py and must be shared by all new groups.\n",
        encoding="utf-8",
    )
    hashes = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    write_json(out / "data_hashes.json", hashes)
    print(json.dumps({"source_sha256": sha256(src), "counts": {k: len(v) for k, v in groups.items()},
                      "train_sources": dict(Counter(r["original_dataset_id"] for r in groups["pretrain_train"])),
                      "output_dir": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
