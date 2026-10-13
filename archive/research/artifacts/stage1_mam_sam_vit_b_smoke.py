"""CPU-only OrgMIM-style MAM feasibility using official SAM ViT-B embeddings."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zarr
from segment_anything import SamPredictor, sam_model_registry


ROOT = Path(__file__).resolve().parents[1]
TOP = ROOT / "topology_feasibility"
MAM_DIR = TOP / "mam_sam_vit_b"
sys.path.insert(0, str(MAM_DIR))
from mam_utils import embeddings_to_affinities, nearest_neighbor_resize  # noqa: E402


WEIGHT = MAM_DIR / "weights" / "sam_vit_b_01ec64.pth"
EXPECTED_SHA = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_patch(sample: dict) -> np.ndarray:
    arr = zarr.open(sample["local_path"], mode="r")[sample["image_array_path"]]
    h, w = map(int, sample["image_shape"])
    side = min(h, w, 512)
    y, x = (h - side) // 2, (w - side) // 2
    gray = np.asarray(arr[int(sample["slice_index"]), y:y + side, x:x + side], dtype=np.float32)
    low, high = np.percentile(gray, (1, 99))
    gray = np.clip((gray - low) / max(high - low, 1e-6), 0, 1)
    return np.repeat((gray * 255).astype(np.uint8)[:, :, None], 3, axis=2)


def mam_from_rgb(predictor: SamPredictor, rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    with torch.inference_mode():
        predictor.set_image(rgb)
        embedding = predictor.features.detach().cpu().numpy().squeeze()
    affinity = embeddings_to_affinities(embedding, delta_v=0.5, delta_d=1.5)
    mam = np.minimum(affinity[0], affinity[1])[1:, 1:]
    mam = nearest_neighbor_resize(mam, (64, 64)).astype(np.float32)
    return mam, tuple(embedding.shape)


def main() -> None:
    started = time.monotonic()
    torch.set_num_threads(4)
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "CPU-only trial requires CUDA_VISIBLE_DEVICES empty"
    assert sha256(WEIGHT) == EXPECTED_SHA
    samples = json.loads((TOP / "sample_manifest.json").read_text())["samples"]
    sample = samples[0]
    rgb = read_patch(sample)
    model = sam_model_registry["vit_b"](checkpoint=str(WEIGHT)).to("cpu").eval()
    predictor = SamPredictor(model)
    before_forward = time.monotonic()
    mam, embedding_shape = mam_from_rgb(predictor, rgb)
    forward_seconds = time.monotonic() - before_forward
    assert mam.shape == (64, 64) and np.isfinite(mam).all()
    assert 0 <= float(mam.min()) <= float(mam.max()) <= 1
    np.save(MAM_DIR / "maps" / "smoke_first_sample_mam.npy", mam)
    np.save(MAM_DIR / "maps" / "smoke_first_sample_raw.npy", rgb[:, :, 0])
    result = {
        "status": "pass", "sample": {k: sample[k] for k in ("volume_id", "source", "slice_index")},
        "model": "Meta SAM ViT-B 01ec64, frozen, CPU", "model_sha256": EXPECTED_SHA,
        "affinity": "OrgMIM embeddings_to_affinities, delta_v=0.5, delta_d=1.5, min vertical/horizontal",
        "rgb_shape": rgb.shape, "embedding_shape": embedding_shape, "mam_shape": mam.shape,
        "mam_min": float(mam.min()), "mam_max": float(mam.max()), "mam_std": float(mam.std()),
        "forward_and_affinity_seconds": forward_seconds, "total_seconds": time.monotonic() - started,
        "uses_labels": False, "training_gpu_used": False,
    }
    (MAM_DIR / "smoke_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
