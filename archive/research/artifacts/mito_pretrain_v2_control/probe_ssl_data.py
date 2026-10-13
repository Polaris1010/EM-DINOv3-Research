"""Small CPU check of source-balanced Zarr sampling and DINO collate shapes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov3-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=["B", "L", "T", "LT"], default="B")
    args = parser.parse_args()
    sys.path.insert(0, str(args.dinov3_root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dinov3.configs import get_default_config
    from ssl_data import Collate, Images

    cfg = OmegaConf.merge(get_default_config(), OmegaConf.load(args.config))
    cfg.crops.global_crops_size = 256
    cfg.crops.local_crops_size = 112
    cfg.crops.local_crops_number = 8
    cfg.student.patch_size = 16
    dataset = Images(args.manifest, cfg, 0, 0, 2, args.arm)
    one = dataset[0]
    two = dataset[1]
    batch = Collate(cfg, args.arm)([one, two])
    shape = {k: list(v.shape) for k, v in batch.items() if hasattr(v, "shape")}
    result = {"status": "complete", "sample_sources": [one[1]["source"], two[1]["source"]],
              "sample_locations": [one[1], two[1]], "tensor_shapes": shape,
              "first_index": batch["_first_index"], "sample_digest": batch["_sample_digest"]}
    assert shape["collated_global_crops"] == [4, 3, 256, 256], shape
    assert shape["collated_local_crops"] == [16, 3, 112, 112], shape
    if args.arm in {"T", "LT"}:
        assert shape["collated_teacher_global_crops"] == [4, 3, 256, 256], shape
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
