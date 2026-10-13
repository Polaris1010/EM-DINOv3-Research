"""CPU check of the natural ViT-L/16 checkpoint and native patch-token wrapper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov3-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.dinov3_root))
    from dinov3.models.vision_transformer import vit_large

    state = torch.load(args.weights, map_location="cpu", mmap=True, weights_only=True)
    model = vit_large(
        patch_size=16,
        n_storage_tokens=4,
        mask_k_bias=True,
        layerscale_init=1e-5,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_dtype="bf16",
    )
    loaded = model.load_state_dict(state, strict=True)
    model.eval()
    torch.set_num_threads(4)
    image = torch.linspace(0, 1, 3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32)
    with torch.inference_mode():
        native = model.forward_features(image)
        layers = model.get_intermediate_layers(image, n=[5, 11, 17, 23], reshape=True)
    final = layers[-1].flatten(2).transpose(1, 2)
    delta = float((final - native["x_norm_patchtokens"]).abs().max())
    result = {
        "status": "complete",
        "weight": str(args.weights),
        "missing_keys": list(loaded.missing_keys),
        "unexpected_keys": list(loaded.unexpected_keys),
        "storage_tokens": model.n_storage_tokens,
        "attention_mask_k_bias": True,
        "patch_shape": list(native["x_norm_patchtokens"].shape),
        "storage_shape": list(native["x_storage_tokens"].shape),
        "intermediate_shapes": [list(t.shape) for t in layers],
        "last_layer_patch_max_abs_error": delta,
        "expected_patch_grid": [2, 2],
    }
    if delta > 1e-6 or result["patch_shape"] != [1, 4, 1024]:
        raise AssertionError(result)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
