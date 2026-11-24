"""Wave 1 frozen-backbone training using the original C/D training loop."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CPT = ROOT.parent
BASE = CPT.parent
HIST = BASE / "decoder_heads_benchmark_20261123"
REGISTRY = BASE / "mv_generalist_stage1_20261013"
MODELS = {
    "P0": (None, "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"),
    "Clean-C0_100k": ("Clean_C0_100k", "a5e4d0fcc804ed21db23b1e70292b9b89fe4722cac16c369d6e4711f8f4f1260"),
    "Clean-C1_100k": ("Clean_C1_30k_to_100k", "22c3104f954b15f1259c52ae969898034e8cae61469cf183bea3615b0b34d250"),
}


def read(path):
    return json.loads(path.read_text())


def configure(model, arm):
    import torch
    sys.path.insert(0, str(HIST))
    import heads_model
    import heads_targets
    import train_heads
    sys.path.insert(0, str(REGISTRY / "src"))
    import mv_probe
    p0 = next(m for m in read(REGISTRY / "model_registry.json")["models"] if m["id"] == "P0")
    sys.path.insert(0, p0["dinov3_root"])
    from dinov3.models import build_model_from_cfg
    from omegaconf import OmegaConf
    run, digest = MODELS[model]
    checkpoint = (Path(p0["checkpoint_path"]) if run is None else
                  CPT / "runs" / run / "checkpoints/step000100000/teacher_backbone.pth")
    assert train_heads.sha(checkpoint) == digest
    out = ROOT / "runs" / model
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(heads_model.CONFIG, encoder=str(checkpoint), encoder_sha256=digest)
    for module in (heads_model, train_heads):
        module.CONFIG = cfg
    train_heads.OUT = out

    def build(actual_arm):
        assert actual_arm == arm
        with torch.device("meta"):
            enc, dim = build_model_from_cfg(OmegaConf.load(p0["resolved_config_path"]), only_teacher=True)
        state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        enc.load_state_dict(state if run is None else state["backbone"], strict=True, assign=True)
        assert dim == 1024 and enc.n_storage_tokens == 4 and enc.patch_size == 16
        torch.manual_seed(cfg["seed"])
        head = heads_model.PanopticHead() if arm == "panoptic" else heads_model.FlowHead()
        return heads_model.System(enc.cuda(), head.cuda()), mv_probe

    train_heads.build = build
    assert (heads_targets.OUT / "ank_diffusion.so").is_file()
    return train_heads, build, out, cfg


def gpu_limit(index, arm):
    expected_host = {"panoptic": "mira-4090-8", "cellpose": "mira-4090-7"}
    assert socket.gethostname().split(".")[0] == expected_host[arm]
    assert index in (1, 2, 3)
    row = subprocess.check_output(
        ["nvidia-smi", "-i", str(index), "--query-gpu=uuid,memory.free",
         "--format=csv,noheader,nounits"], text=True).strip().split(",")
    uuid, available = row[0].strip(), int(row[1].strip())
    assert available >= 6144, f"GPU {index} has {available} MiB free"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == uuid
    import torch
    assert torch.cuda.device_count() == 1
    mib = 2**20
    cap = min(4608, available - 2048)
    torch.cuda.set_per_process_memory_fraction(cap * mib / torch.cuda.get_device_properties(0).total_memory)
    print(f"GPU {index}: {available} MiB initially free, {cap} MiB CUDA allocator ceiling", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("arm", choices=("panoptic", "cellpose"))
    parser.add_argument("gpu", type=int, choices=(1, 2, 3))
    parser.add_argument("mode", choices=("smoke", "train"))
    args = parser.parse_args()
    assert args.gpu == list(MODELS).index(args.model) + 1
    gpu_limit(args.gpu, args.arm)
    trainer, _, _, _ = configure(args.model, args.arm)
    trainer.train(args.arm, args.mode == "smoke")
