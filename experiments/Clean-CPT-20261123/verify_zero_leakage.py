"""Build or verify the authorized Clean-CPT split against its source manifests."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def acquisition(name):
    match = re.search(r"(?:^|_)jrc_[a-z0-9-]+", name.casefold())
    return match.group().lstrip("_") if match else None


def build(project, audit):
    codex = project / "codex"
    original = codex / "ssl_diversity_d12_20261017"
    source = read(original / "D2_manifest.json")
    mv = read(original / "mitoverse_pool.json")
    cem = read(original / "cem_pool.json")
    assert sha(original / "mitoverse_pool.json") == source["mitoverse_pool_sha256"]
    assert sha(original / "cem_pool.json") == source["cem_pool_sha256"]
    with (audit / "cellmap_official_test_crops.csv").open(newline="", encoding="utf-8") as f:
        cellmap_tests = {(r["dataset"], r["crop_name"]) for r in csv.DictReader(f)}
    cellmap_sources = {source for source, _ in cellmap_tests}
    cellmap_local = read(project / "MitoVerse_data/splits/cellmap.json")["volumes"]
    cellmap_val_test = {r["id"] for r in cellmap_local if r["split"] in {"val", "test", "validation"}}
    me = read(project / "MitoVerse_data/splits/mitoem2.0.json")["volumes"]
    me_holdout = {r["id"] for r in me if r["split"] in {"val", "test", "validation"}}
    mb = codex / "mb1_official_benchmark_20261015"
    mb1_rows = read(mb / "evidence/overlap_source_check.json")["rows"]
    mb1_ids = {r["candidate_volume_id"] for r in mb1_rows if r.get("candidate_volume_id") not in {None, "UNVERIFIED"}}
    assert len(cellmap_sources) == 6 and len(cellmap_tests) == 16
    assert len(me_holdout) == 11 and len({r["dataset"] for r in mb1_rows}) == 7
    assert len(mb1_ids) == 6
    volume_ids = {r["volume_id"] for r in mv["volumes"]}
    assert volume_ids == {r["volume_id"] for r in source["volumes"]} and len(volume_ids) == 153
    excluded = {i for i in volume_ids if acquisition(i) in cellmap_sources}
    retained = volume_ids - excluded
    intersections = {
        "MB1_registered_candidates": sorted(retained & mb1_ids),
        "MB1_fly_acquisition": sorted(i for i in retained if i.startswith("openorganelle_jrc_fly")),
        "Lucchi_family": sorted(i for i in retained if "lucchi" in i.casefold()),
        "MitoEM2_val_test_exact": sorted(retained & me_holdout),
        "CellMap_val_test_exact": sorted(retained & cellmap_val_test),
        "CellMap_official_test_acquisitions": sorted(i for i in retained if acquisition(i) in cellmap_sources),
    }
    assert not any(intersections.values()), intersections
    assert len(excluded) == 34 and len(retained) == 119
    validation = set(read(codex / "cem_mitolab_three_20261118/manifest.json")["validation"])
    publisher = read(audit / "cem_publisher_identity.json")
    groups = {r["name"]: r for r in cem["groups"]}
    assert groups.keys() == publisher.keys() and len(groups) == 627 and validation <= groups.keys()
    old = read(audit / "phase0_audit_98.json")
    old_cem_excluded = set(old["cem"]["known_excluded_ids"])
    assert validation <= old_cem_excluded and len(old_cem_excluded - validation) == 2
    kept_groups = [r for r in cem["groups"] if r["name"] not in validation]
    assert len(validation) == 30 and sum(groups[k]["images"] for k in validation) == 34564
    assert len(kept_groups) == 597 and sum(r["images"] for r in kept_groups) == 1390869
    assert not {r["name"] for r in kept_groups} & validation
    internal = [r for r in kept_groups if publisher[r["name"]]["internal"] and not publisher[r["name"]]["external_reference"]]
    assert len(internal) == 432 and sum(r["images"] for r in internal) == 427093
    keep_volume_rows = []
    for row in mv["volumes"]:
        if row["volume_id"] in retained:
            row = dict(row, local_path=str(Path(row["local_path"]).relative_to(project)))
            path = project / row["local_path"]
            assert path.is_dir() and (path / row["image_array_path"]).is_dir(), path
            keep_volume_rows.append(row)
    sampling_groups = {}
    for source_name, acquisitions in mv["sampling_groups"].items():
        by_acquisition = {key: [i for i in ids if i in retained] for key, ids in acquisitions.items()}
        by_acquisition = {key: ids for key, ids in by_acquisition.items() if ids}
        if by_acquisition:
            sampling_groups[source_name] = by_acquisition
    assert {i for group in sampling_groups.values() for ids in group.values() for i in ids} == retained
    archive = Path(cem["archive"])
    index = Path(cem["index"])
    assert archive.stat().st_size == cem["archive_bytes"]
    assert archive.stat().st_mtime_ns == cem["archive_mtime_ns"]
    assert sha(index) == cem["index_sha256"]
    with sqlite3.connect(index.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        for row in kept_groups:
            for image_id in (row["first_id"], row["last_id"]):
                assert db.execute("SELECT collection FROM images WHERE id=?", (image_id,)).fetchone() == (row["name"],)
    return {
        "protocol": "Clean-CPT-20261123-final",
        "cem_fraction": 0.5,
        "excluded_mitoverse": sorted(excluded),
        "excluded_cem": sorted(validation),
        "restored_previously_excluded_cem": sorted(old_cem_excluded - validation),
        "exact_identifier_intersections": intersections,
        "cellmap_official_test_crops": sorted(cellmap_tests),
        "holdout_evidence": {
            "MV3": "retired; existing historical legal Z intervals are inherited",
            "MB1": "known IDs and official acquisition metadata checked; supplementary dHash audit sampled all seven groups",
            "HQ5nm": "separate HQ-5nm acquisition/workspace; original metadata lacks a common cross-corpus acquisition ID",
            "CEM_NCI": "432 internal collections, 427093 images. Conrad and Narayan (2023) declare the benchmark excluded from CEM; internal designation supports institutional separation from external MitoEM/CellMap. No per-image origin crosswalk is provided for these collections.",
        },
        "source_hashes": {
            "D2_manifest": sha(original / "D2_manifest.json"),
            "mitoverse_pool": source["mitoverse_pool_sha256"],
            "cem_pool": source["cem_pool_sha256"],
            "cellmap_official_test_crops": sha(audit / "cellmap_official_test_crops.csv"),
        },
        "mitoverse": {
            "original_volume_count": 153,
            "volume_count": len(retained),
            "volumes": keep_volume_rows,
            "sampling_groups": sampling_groups,
            "heldout_regions": {i: v for i, v in mv["heldout_regions"].items() if i in retained},
        },
        "cem": {
            "original_collections": 627,
            "collections": len(kept_groups),
            "images": sum(r["images"] for r in kept_groups),
            "archive": str(archive.relative_to(project)),
            "index": str(index.relative_to(project)),
            "archive_bytes": cem["archive_bytes"],
            "archive_mtime_ns": cem["archive_mtime_ns"],
            "index_sha256": cem["index_sha256"],
            "groups": kept_groups,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()
    actual = build(args.project.resolve(), args.audit.resolve())
    data = (json.dumps(actual, indent=2, ensure_ascii=False, sort_keys=False) + "\n").encode()
    if args.create:
        with args.manifest.open("xb") as handle:
            handle.write(data)
    else:
        assert args.manifest.read_bytes() == data, "Manifest differs from independently derived split"
    print(f"MitoVerse 119/153; CEM 597/627, 1390869 images; manifest SHA256={sha(args.manifest)}")
