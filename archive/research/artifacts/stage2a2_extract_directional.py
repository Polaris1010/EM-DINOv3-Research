"""Re-extract native directional SAM affinities for the fixed 424 samples."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

RUN = Path(__file__).resolve().parents[1]
REF = json.loads((RUN / "references/paths.json").read_text())
STAGE1 = Path(REF["stage1_root"])
sys.path[:0] = [str(STAGE1 / "scripts"), str(STAGE1 / "topology_feasibility/mam_sam_vit_b")]
from mam_sam_vit_b_smoke import read_patch  # noqa: E402
from mam_utils import embeddings_to_affinities, nearest_neighbor_resize  # noqa: E402

WEIGHT = Path(REF["sam_vit_h_weight"])
OLD = Path(REF["stage2_vit_h_paired_maps"])
OUT = RUN / "branch_S/directional_affinity"
MAPS = OUT / "maps"
EXPECTED_SHA = "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=424)
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in {"1", "2", "3"}
    assert WEIGHT.stat().st_size == 2564550879 and sha256(WEIGHT) == EXPECTED_SHA
    samples = json.loads((RUN / "references/sample_manifest_424.json").read_text())["samples"]
    assert len(samples) == 424 and 1 <= args.limit <= 424
    torch.set_num_threads(4)
    MAPS.mkdir(parents=True, exist_ok=True)
    model = sam_model_registry["vit_h"](checkpoint=str(WEIGHT)).to("cuda").eval()
    predictor = SamPredictor(model)
    t0 = time.monotonic()
    max_old_diff = 0.0
    generated = 0
    for i, sample in enumerate(samples[:args.limit]):
        target = MAPS / f"sample_{i:06d}.npz"
        if target.exists():
            with np.load(target) as pack:
                assert pack["ay"].shape == pack["ax"].shape == (64, 64)
            continue
        rgb = read_patch(sample)
        with torch.inference_mode():
            predictor.set_image(rgb)
            embedding = predictor.features.detach().cpu().numpy().squeeze()
        assert embedding.shape == (256, 64, 64)
        affs = embeddings_to_affinities(embedding, delta_v=.5, delta_d=1.5)
        ay, ax = affs[0].copy(), affs[1].copy()
        reconstructed = nearest_neighbor_resize(np.minimum(ay, ax)[1:, 1:], (64, 64)).astype(np.float32)
        with np.load(OLD / f"sample_{i:06d}.npz") as old:
            prior = old["mam"][0]
        difference = float(np.max(np.abs(reconstructed - prior)))
        max_old_diff = max(max_old_diff, difference)
        if difference > 1e-5:
            raise ValueError(f"Prior reconstruction mismatch at sample {i}: {difference}")
        ay[0] = 0
        ax[:, 0] = 0
        assert np.isfinite(ay).all() and np.isfinite(ax).all()
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, ay=ay, ax=ax)
        temporary.replace(target)
        generated += 1
        if i % 25 == 0 or i == 423:
            progress = {"status": "running", "last_sample_index": i, "files_present": len(list(MAPS.glob("sample_*.npz"))),
                        "max_prior_reconstruction_abs_difference": max_old_diff,
                        "elapsed_seconds": time.monotonic() - t0, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}
            (OUT / "progress.json").write_text(json.dumps(progress, indent=2))
            print(json.dumps(progress), flush=True)
    if args.limit == 424:
        assert len(list(MAPS.glob("sample_*.npz"))) == 424
        complete = {"status": "complete", "samples": 424, "sam_weight_sha256": EXPECTED_SHA,
                    "max_prior_reconstruction_abs_difference": max_old_diff,
                    "affinity_definition": "OrgMIM embedding neighbor affinity, delta_v=0.5, delta_d=1.5",
                    "ay_semantics": "ay[r,c] joins (r-1,c) to (r,c), r>0",
                    "ax_semantics": "ax[r,c] joins (r,c-1) to (r,c), c>0",
                    "input": "Stage1 1/99 percentile-normalized central 512 patch; no labels",
                    "elapsed_seconds": time.monotonic() - t0}
        (OUT / "complete.json").write_text(json.dumps(complete, indent=2))
        print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
