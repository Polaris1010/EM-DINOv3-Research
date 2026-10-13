"""CPU-first paired representation diagnostics; no training launcher or SSL entrypoint.

Inputs are trusted local NPZ files and an audited JSON plan. Final test/external
ROIs are deliberately rejected by this development diagnostic entrypoint.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "vendor").exists():
    sys.path.insert(0, str(ROOT / "vendor"))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(module):
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        a = v.detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(tuple(a.shape)).encode())
        h.update(a.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_module(path):
    spec = importlib.util.spec_from_file_location("audited_diagnosis_factory", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate_plan(plan):
    if plan.get("status") != "locked_development_diagnosis":
        raise ValueError("Diagnosis requires a locked development plan")
    if plan.get("automatic_ssl_training") is not False:
        raise ValueError("Automatic SSL training must be false")
    models = plan["models"]
    if {x["id"] for x in models} != {"P0", "B100k", "B150k"} or len(models) != 3:
        raise ValueError("Exactly one P0/B100k/B150k is required")
    if any(len(x.get("sha256", "")) != 64 for x in models):
        raise ValueError("Checkpoint digests are required")
    seeds = plan["decoder_seeds"]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Seeds must be nonempty and unique")
    ids = set()
    for roi in plan["rois"]:
        if roi.get("split") not in {"train", "dev"}:
            raise ValueError("Final test/external/unknown ROI cannot select a diagnostic method")
        if not all(roi.get(k) for k in ("id", "source_id", "acquisition_id", "image_path",
                                        "label_path", "valid_mask_path", "annotation_evidence")):
            raise ValueError("Source, acquisition, annotation and mask evidence are required")
        if roi.get("label_kind") not in {"binary_mito", "instance_mito"}:
            raise ValueError("Labels must explicitly describe mitochondria")
        if roi["id"] in ids:
            raise ValueError("ROI IDs must be unique")
        ids.add(roi["id"])
        spacing = roi["spacing_yx_nm"]
        bounds = roi["roi_yx_nm"]
        if len(spacing) != 2 or any(not np.isfinite(x) or x <= 0 for x in spacing):
            raise ValueError("Physical spacing must be finite and positive")
        if len(bounds) != 4 or not all(np.isfinite(bounds)) or not (bounds[0] < bounds[1] and bounds[2] < bounds[3]):
            raise ValueError("Physical ROI bounds are required")
        for k in ("image_sha256", "label_sha256", "valid_mask_sha256"):
            if len(roi.get(k, "")) != 64:
                raise ValueError("Every data input requires a SHA256")
    if not ids:
        raise ValueError("No audited ROIs")
    return plan


def read_roi(roi):
    arrays = []
    for key in ("image", "label", "valid_mask"):
        path = Path(roi[key + "_path"])
        if sha(path) != roi[key + "_sha256"]:
            raise ValueError(f"Changed input: {key}")
        arrays.append(np.load(path, allow_pickle=False))
    image, label, valid = arrays
    if image.ndim != 2 or image.shape != label.shape or image.shape != valid.shape:
        raise ValueError("Aligned 2D image/label/valid mask required")
    if not np.isfinite(image).all() or not np.isfinite(label).all():
        raise ValueError("Nonfinite inputs")
    if not np.isin(valid, [0, 1]).all() or not np.equal(label, np.floor(label)).all():
        raise ValueError("Valid mask must be binary and labels integral")
    if roi["label_kind"] == "binary_mito" and not np.isin(label[label >= 0], [0, 1]).all():
        raise ValueError("Other semantic classes cannot be treated as instance IDs")
    y0, y1, x0, x1 = roi["roi_yx_nm"]
    physical = np.array(image.shape) * np.array(roi["spacing_yx_nm"])
    if not np.allclose(physical, [y1 - y0, x1 - x0], atol=1e-5, rtol=1e-6):
        raise ValueError("Image dimensions and physical ROI disagree")
    return (torch.tensor(image, dtype=torch.float32)[None, None],
            torch.tensor(label.astype(np.int64))[None],
            torch.tensor(valid.astype(bool))[None] & torch.tensor(label >= 0)[None])


def resize_view(image, label, valid, size, mode):
    if mode == "fixed_fov":
        # The SAME native ROI is resampled. Upsampling adds no native information.
        out = F.interpolate(image, (size, size), mode="bilinear", align_corners=False, antialias=True)
        lab = F.interpolate(label[:, None].float(), (size, size), mode="nearest")[:, 0].long()
        # Conservative validity: any ignored native pixel in a reduced cell invalidates it.
        if image.shape[-2] >= size and image.shape[-1] >= size:
            mask = F.adaptive_max_pool2d((~valid[:, None]).float(), (size, size))[:, 0] == 0
        else:
            mask = F.interpolate(valid[:, None].float(), (size, size), mode="nearest")[:, 0] > .5
        return out, lab, mask
    if mode != "native_fov" or min(image.shape[-2:]) < size:
        raise ValueError("Native views require genuine native pixels at least as large as the crop")
    h, w = image.shape[-2:]
    y, x = (h - size) // 2, (w - size) // 2
    return image[..., y:y+size, x:x+size], label[..., y:y+size, x:x+size], valid[..., y:y+size, x:x+size]


class FrozenReadout(nn.Module):
    def __init__(self, encoder, head, indices=(5, 11, 17, 23)):
        super().__init__()
        self.encoder, self.head, self.indices = encoder, head, indices
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def features(self, image):
        with torch.no_grad():
            x = image.repeat(1, 3, 1, 1) if image.shape[1] == 1 else image
            feats = self.encoder.get_intermediate_layers(x, n=self.indices, reshape=True)
            return [f.detach().float() for f in feats]

    def forward(self, image):
        return self.head(self.features(image), image.shape[-2:])


class LinearProbe(nn.Module):
    def __init__(self, embed_dim, use_multilayer=False):
        super().__init__()
        self.use_multilayer = use_multilayer
        self.projection = nn.Conv2d(embed_dim * (4 if use_multilayer else 1), 2, 1)

    def forward(self, features, output_size):
        x = torch.cat(features, 1) if self.use_multilayer else features[-1]
        return F.interpolate(self.projection(x), output_size, mode="bilinear", align_corners=False)


class PrimusHead(nn.Module):
    """Uses the actual supplied Primus module, including any historical frequency gates."""
    def __init__(self, primus):
        super().__init__()
        self.up_projection = primus.up_projection
        self.last_gates = getattr(primus, "last_gates", None)

    def forward(self, features, output_size):
        if self.last_gates is not None:
            features = [gate(f) for gate, f in zip(self.last_gates, features)]
        y = self.up_projection(torch.cat(features, 1))
        if y.shape[-2:] != tuple(output_size):
            raise ValueError("Primus shape mismatch; pad to a multiple of patch size")
        return y


def masked_loss(logits, label, valid):
    target = (label > 0).long().masked_fill(~valid, -1)
    if not valid.any():
        raise ValueError("No valid annotation")
    ce = F.cross_entropy(logits, target, ignore_index=-1)
    p = logits.softmax(1)[:, 1] * valid
    gt = (target == 1).float()
    return ce + 1 - (2 * (p * gt).sum() + 1e-6) / (p.sum() + gt.sum() + 1e-6)


def segmentation_counts(logits, label, valid):
    pred, gt = logits.argmax(1) == 1, label > 0
    tp = int((pred & gt & valid).sum())
    fp = int((pred & ~gt & valid).sum())
    fn = int((~pred & gt & valid).sum())
    return {"tp": tp, "fp": fp, "fn": fn,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None}


def fit_readout(head, train_batches, dev_batches, steps=3000, seed=20260907):
    """Shared readout-only fitter for detached features (no encoder or SSL update).

    This is a NEW paired-ROI diagnostic, not a replay of historical nnU-Net
    sampling/sliding/TTA. Batch caches must come from locked train/dev ROIs.
    A caller supplies the same caches, step budget and seed to all three models.
    """
    if not 1 <= steps <= 3000 or not train_batches or not dev_batches:
        raise ValueError("Bounded readout budget and train/dev batches required")
    for features, label, valid in list(train_batches) + list(dev_batches):
        if any(f.requires_grad or f.grad_fn is not None for f in features):
            raise ValueError("Frozen, detached feature caches required")
    torch.manual_seed(seed)
    initial_digest = digest(head)
    optimizer = torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.05,betas=(.9,.98))
    best, best_state, curve = -1.0, None, []
    for step in range(steps):
        head.train()
        features, label, valid = train_batches[step % len(train_batches)]
        lr=1e-4*min(1.,(step+1)/50)*(.1+.9*.5*(1+np.cos(np.pi*step/steps)))
        for group in optimizer.param_groups: group["lr"]=lr
        optimizer.zero_grad(set_to_none=True)
        loss=masked_loss(head(features,label.shape[-2:]),label,valid)
        if not torch.isfinite(loss): raise ValueError("Nonfinite readout loss")
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(),12,error_if_nonfinite=True)
        optimizer.step()
        if (step+1)%1000==0 or step+1==steps:
            head.eval()
            with torch.no_grad():
                metrics=[segmentation_counts(head(f,y.shape[-2:]),y,v) for f,y,v in dev_batches]
            dice=[m["dice"] for m in metrics if m["dice"] is not None]
            score=sum(dice)/len(dice) if dice else -1.0
            curve.append({"step":step+1,"macro_roi_dice":score,"per_roi":metrics})
            if score>best:
                best=score;best_state={k:v.detach().clone() for k,v in head.state_dict().items()}
    if best_state is None: raise ValueError("No evaluable foreground/background dev ROI")
    head.load_state_dict(best_state)
    return {"protocol":"paired_roi_cached_features_v1","initial_head_sha256":initial_digest,
            "steps":steps,"seed":seed,"curve":curve,"encoder_updated":False}


def dense_diagnostic(feature, label, valid, patch_size=16, instance_labels=True):
    """Pure, valid patches; horizontal/vertical neighbour pairs have matched distance.

    Binary labels support same-class analysis only: no invented instance identities.
    Instance size is reported as a stratum, rather than averaged across all objects.
    """
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError("One aligned feature map per ROI")
    _, c, h, w = feature.shape
    if label.shape != (1, h * patch_size, w * patch_size) or valid.shape != label.shape:
        raise ValueError("Labels must align exactly with the feature patch grid")
    blocks = label[0].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size).reshape(h, w, -1)
    vb = valid[0].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size).reshape(h, w, -1).all(-1)
    pure = (blocks.min(-1).values == blocks.max(-1).values) & vb
    ids = blocks[..., 0]
    f = feature[0].permute(1, 2, 0)
    if not torch.isfinite(f).all():
        raise ValueError("Nonfinite dense features")
    vectors = f[pure]
    if len(vectors) < 2:
        return {"valid_pure_patches": len(vectors), "status": "INSUFFICIENT"}
    centered = vectors - vectors.mean(0)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    rank_probability = singular / singular.sum().clamp_min(1e-12)
    effective_rank = float(torch.exp(-(rank_probability * rank_probability.clamp_min(1e-12).log()).sum())) if singular.sum() > 0 else 0.0
    normalized = F.normalize(f, dim=-1)
    fg, bg = f[pure & (ids > 0)], f[pure & (ids == 0)]
    separation = float(torch.linalg.vector_norm(fg.mean(0) - bg.mean(0))) if len(fg) and len(bg) else None
    instance_sizes = {int(k): int((label == k).sum()) for k in torch.unique(label[label > 0])}
    strata = {}
    for dy, dx, axis in ((0, 1, "x"), (1, 0, "y")):
        for y in range(h - dy):
            for x in range(w - dx):
                if not (pure[y, x] and pure[y + dy, x + dx]):
                    continue
                a, b = int(ids[y, x]), int(ids[y + dy, x + dx])
                group = "within" if a > 0 and a == b else "boundary" if (a > 0) != (b > 0) else None
                if group is None:
                    continue
                obj = a if a > 0 else b
                area = instance_sizes[obj]
                size_bin = ("lt4096" if area < 4096 else "4096to16383" if area < 16384 else "ge16384") if instance_labels else "instance_size_UNVERIFIED"
                key = f"{axis}:{size_bin}"
                strata.setdefault(key, {"within": [], "boundary": []})[group].append(float((normalized[y,x] * normalized[y+dy,x+dx]).sum()))
    pairs = {}
    for key, values in strata.items():
        means = {k: sum(v) / len(v) if v else None for k, v in values.items()}
        pairs[key] = {"counts": {k: len(v) for k,v in values.items()}, "mean_cosine": means,
                      "delta_boundary": means["within"] - means["boundary"] if all(v is not None for v in means.values()) else None}
    return {"status": "computed", "valid_pure_patches": len(vectors), "foreground_patches": len(fg),
            "background_patches": len(bg), "effective_rank_singular_entropy": effective_rank,
            "mean_channel_variance": float(centered.square().mean()), "total_spectral_energy": float(energy.sum()),
            "foreground_background_centroid_l2": separation, "adjacent_pair_strata": pairs,
            "within_interpretation": "same instance" if instance_labels else "same foreground class; instance identity unavailable",
            "limitations": "Pure-patch filtering misses subpatch weak boundaries; paired cosine is descriptive, not linear separability or proof of degradation."}


def exposure_rows(plan):
    rows = []
    for model in plan["models"]:
        for roi in plan["rois"]:
            rows.append({"model": model["id"], "model_sha256": model["sha256"],
                         "roi_id": roi["id"], "source_id": roi["source_id"],
                         "acquisition_id": roi["acquisition_id"], "split": roi["split"],
                         "ssl_exposure": roi.get("ssl_exposure", {}).get(model["id"], "UNVERIFIED"),
                         "supervised_readout_exposure": "training" if roi["split"] == "train" else "validation_selection",
                         "method_development_exposure": True,
                         "roi_yx_nm": roi["roi_yx_nm"]})
    return rows


def run_features(plan, factory_file, output, device="cpu"):
    validate_plan(plan)
    if device != "cpu" and not (device == "cuda:0" and os.environ.get("CUDA_VISIBLE_DEVICES") in {"1", "2", "3"}):
        raise ValueError("Only CPU or one explicitly masked physical GPU 1/2/3 is permitted")
    if sha(factory_file) != plan["factory_sha256"]:
        raise ValueError("Factory changed")
    factory = load_module(factory_file)
    output = Path(output)
    if output.exists():
        raise ValueError("New output directory required; historical results never overwritten")
    output.mkdir(parents=True)
    results = []
    for model in plan["models"]:
        if sha(model["checkpoint_path"]) != model["sha256"]:
            raise ValueError("Checkpoint changed")
        enc = factory.build_encoder(model).to(device).eval().requires_grad_(False)
        if any(p.requires_grad for p in enc.parameters()):
            raise ValueError("Encoder must be frozen")
        for roi in plan["rois"]:
            image, label, valid = read_roi(roi)
            # One intensity normalization per native ROI, shared by every model/view.
            low, high = plan["intensity_range"]
            if high <= low:
                raise ValueError("Invalid fixed intensity range")
            image = ((image - low) / (high - low)).clamp(0, 1)
            mean = torch.tensor(plan["rgb_mean"], device=device)[None,:,None,None]
            std = torch.tensor(plan["rgb_std"], device=device)[None,:,None,None]
            for mode in ("fixed_fov", "native_fov"):
                for size in (256, 384, 512):
                    x, y, mask = resize_view(image, label, valid, size, mode)
                    with torch.no_grad():
                        fs = enc.get_intermediate_layers((x.to(device).repeat(1,3,1,1)-mean)/std,
                                                        n=[5,11,17,23], reshape=True)
                    if len(fs) != 4 or any(f.shape[1] != 1024 for f in fs):
                        raise ValueError("Expected ViT-L four 1024-dimensional feature maps")
                    if mode == "fixed_fov":
                        bounds = roi["roi_yx_nm"]
                        native_pixels_added = False
                    else:
                        h, w = image.shape[-2:]
                        ystart, xstart = (h-size)//2, (w-size)//2
                        sy, sx = roi["spacing_yx_nm"]
                        b = roi["roi_yx_nm"]
                        bounds = [b[0]+ystart*sy,b[0]+(ystart+size)*sy,b[2]+xstart*sx,b[2]+(xstart+size)*sx]
                        native_pixels_added = size > 256
                    results.append({"model":model["id"],"roi_id":roi["id"],"source_id":roi["source_id"],
                                    "acquisition_id":roi["acquisition_id"],"input_size":size,"mode":mode,
                                    "view_roi_yx_nm":bounds,"native_pixels_added_vs256":native_pixels_added,
                                    "effective_spacing_yx_nm":[(bounds[1]-bounds[0])/size,(bounds[3]-bounds[2])/size],
                                    "checkpoint_sha256":model["sha256"],"input_sha256":roi["image_sha256"],
                                    "patch_tokens":(size//16)**2,
                                    "layers": {str(index):dense_diagnostic(f.cpu().float(),y,mask,instance_labels=roi["label_kind"]=="instance_mito")
                                               for index,f in zip([5,11,17,23],fs)}})
        del enc
    (output/"dense_resolution_results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    (output/"data_exposure_records.json").write_text(json.dumps(exposure_rows(plan),indent=2),encoding="utf-8")
    (output/"executed_plan.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--factory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execute-features", action="store_true")
    args = parser.parse_args()
    plan = validate_plan(json.loads(args.plan.read_text(encoding="utf-8")))
    if args.execute_features:
        if not args.factory or not args.output:
            parser.error("Feature execution requires --factory and --output")
        run_features(plan,args.factory,args.output,args.device)
    else:
        print(json.dumps({"status":"validated_only","models":3,"rois":len(plan["rois"]),"training_started":False}))


if __name__ == "__main__":
    main()
