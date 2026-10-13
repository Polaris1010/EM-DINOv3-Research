"""Adapter for audited server DINOv3; never resolves guessed paths or downloads.

The diagnosis plan must give actual roots, resolved config and their hashes.
Existing B_full checkpoint files are opened read-only with weights_only=True.
"""
import hashlib
from pathlib import Path
import random
import sys
import numpy as np
import torch


def verified(path,digest):
    path=Path(path)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError(f"Missing or changed source/config: {path}")
    return path


def build_encoder(spec):
    cfg_path=verified(spec["resolved_config_path"],spec["resolved_config_sha256"])
    sys.path.insert(0,spec["dinov3_root"])
    from omegaconf import OmegaConf
    from dinov3.models import build_model_from_cfg
    cfg=OmegaConf.load(cfg_path)
    enc,_=build_model_from_cfg(cfg,only_teacher=True)
    if enc.embed_dim!=1024 or enc.patch_size!=16 or enc.n_storage_tokens!=4:
        raise ValueError("Expected official ViT-L/16 architecture")
    saved=torch.load(spec["checkpoint_path"],map_location="cpu",weights_only=True)
    if spec["id"]=="P0":
        state=saved
    else:
        expected={"B100k":100000,"B150k":150000}[spec["id"]]
        if saved["step"]!=expected or saved["manifest_sha256"]!=spec["ssl_manifest_sha256"]:
            raise ValueError("Wrong B_full step or training manifest")
        state=saved["backbone"]
    enc.load_state_dict(state,strict=True,assign=True)
    return enc.float().eval().requires_grad_(False)


def build_primus_head(spec,encoder,seed):
    verified(spec["primus_source_path"],spec["primus_source_sha256"])
    sys.path.insert(0,spec["nnunet_root"])
    from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import Primus_Multiscale
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=Primus_Multiscale(embed_dim=1024,patch_embed_size=16,num_classes=2,
                           dino_encoder=encoder,interaction_indices=[5,11,17,23])
    import inspect
    if Path(inspect.getfile(Primus_Multiscale)).resolve()!=Path(spec["primus_source_path"]).resolve():
        raise ValueError("Python imported another Primus implementation")
    has_gates=hasattr(model,"last_gates")
    if has_gates != (spec["readout_variant"]=="historical_primus_frequency_gated"):
        raise ValueError("Gate variant differs from the registered decoder")
    return model
