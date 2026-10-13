"""Benchmark raw-uint8 ViT-H cache tiles without writing a full cache."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zarr
from segment_anything import SamPredictor, sam_model_registry

RUN = Path(__file__).resolve().parents[1]
STAGE1 = Path(json.loads((RUN / "stage1_reference/stage1_paths.json").read_text())["stage1_root"])
sys.path[:0] = [str(STAGE1 / "scripts"), str(STAGE1 / "topology_feasibility/mam_sam_vit_b")]
from mam_sam_vit_b_smoke import mam_from_rgb, sha256  # noqa: E402

WEIGHT = RUN / "topology_calibration/sam_vit_h/sam_vit_h_4b8939.pth"
EXPECTED_SHA = "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"
OUT = RUN / "topology_calibration/cache_benchmark/vit_h_raw_tile_10.json"


def main() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in {"1", "2", "3"}
    assert sha256(WEIGHT) == EXPECTED_SHA
    samples = json.loads((RUN / "topology_calibration/sample_manifest.json").read_text())["samples"]
    selected = []
    seen = set()
    for sample in samples:
        h, w = sample["image_shape"]
        if h >= 1024 and w >= 1024 and sample["volume_id"] not in seen:
            selected.append(sample)
            seen.add(sample["volume_id"])
        if len(selected) == 10:
            break
    assert len(selected) == 10
    torch.set_num_threads(4)
    model = sam_model_registry["vit_h"](checkpoint=str(WEIGHT)).to("cuda").eval()
    predictor = SamPredictor(model)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for sample in selected:
        t0 = time.monotonic()
        array = zarr.open(sample["local_path"], mode="r")[sample["image_array_path"]]
        gray = np.asarray(array[int(sample["slice_index"]), :1024, :1024], dtype=np.uint8)
        assert gray.shape == (1024, 1024)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        read_seconds = time.monotonic() - t0
        torch.cuda.synchronize()
        t1 = time.monotonic()
        mam, embedding_shape = mam_from_rgb(predictor, rgb)
        torch.cuda.synchronize()
        encode_seconds = time.monotonic() - t1
        assert mam.shape == (64, 64) and np.isfinite(mam).all()
        rows.append({"volume_id": sample["volume_id"], "source": sample["source"],
                     "read_seconds": read_seconds, "encode_seconds": encode_seconds,
                     "mam_min": float(mam.min()), "mam_max": float(mam.max()), "mam_std": float(mam.std())})
        print(json.dumps(rows[-1]), flush=True)
    times = np.array([r["encode_seconds"] for r in rows[1:]], dtype=np.float64)
    estimate = json.loads((RUN / "topology_calibration/offline_prior/slice_geometry_report.json").read_text())["estimated_1024_overlap25_tiles"]
    result = {"status": "complete", "n": len(rows), "sam_weight_sha256": EXPECTED_SHA,
              "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "embedding_shape": embedding_shape,
              "median_encode_seconds_excluding_first": float(np.median(times)),
              "p95_encode_seconds_excluding_first": float(np.quantile(times, .95)),
              "estimated_tile_count": estimate,
              "estimated_3_gpu_encode_hours_excluding_io": float(estimate * np.median(times) / 3 / 3600),
              "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "rows": rows}
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
