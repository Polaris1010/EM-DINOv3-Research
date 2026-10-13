"""Check that the natural-init control preserves the existing frozen probe."""
from __future__ import annotations

import ast
import gc
import inspect
import json
import os
from pathlib import Path

import torch

import frozen_probe as original
import frozen_probe_control as control

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "branch_E/control_20261011"


def main() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in {"1", "2", "3"}
    OUT.mkdir(parents=True, exist_ok=True)
    old_path = RUN / "src/frozen_probe.py"
    assert control.digest(old_path) == control.ORIGINAL_SOURCE_SHA
    trees = [ast.parse(p.read_text()) for p in (old_path, RUN / "src/frozen_probe_control.py")]
    for name in ("seed", "Samples", "loss", "validation"):
        nodes = [next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name) for tree in trees]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1]), name + " changed"
    loops = [next(n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "step") for tree in trees]
    assert ast.dump(loops[0]) == ast.dump(loops[1]), "Training loop drift"
    result = {"status": "running", "original_source_sha256": control.digest(old_path),
              "official_sha256": control.digest(control.OFFICIAL), "training_loop_AST_identical": True,
              "sampler_loss_validation_seed_AST_identical": True, "datasets": {}}
    assert result["official_sha256"] == control.OFFICIAL_SHA
    for ds in control.DATASETS:
        old_metrics = json.loads((RUN / f"branch_E/runs/B_full_{ds}/metrics.json").read_text())
        split = control.ROOT / "nnUNet_preprocessed" / ds / "splits_final.json"
        assert old_metrics["source_sha256"] == control.ORIGINAL_SOURCE_SHA
        assert old_metrics["seed"] == 20260907 and old_metrics["steps"] == 3000
        assert old_metrics["split_sha256"] == control.digest(split)
        result["datasets"][ds] = {"split_sha256": control.digest(split), "seed": 20260907, "steps": 3000}
    a = original.model(original.ENCODER, 20260907).eval()
    original_head = control.head_digest(a)
    b = control.model(control.ENCODER, 20260907, "B_full").eval()
    assert control.head_digest(b) == original_head
    for left, right in zip(a.dino_encoder.parameters(), b.dino_encoder.parameters()):
        assert torch.equal(left, right)
    sample_a = original.Samples("Dataset111_CMCC", 20260907).sample(0)
    sample_b = control.Samples("Dataset111_CMCC", 20260907).sample(0)
    assert sample_a[2] == sample_b[2] and torch.equal(sample_a[0], sample_b[0]) and torch.equal(sample_a[1], sample_b[1])
    x = sample_a[0]
    def logits(net):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            features = net.dino_encoder.get_intermediate_layers(x.repeat(1, 3, 1, 1), n=net.interaction_indices, reshape=True)
        with torch.no_grad():
            return net.up_projection(torch.cat([gate(f.float()) for gate, f in zip(net.last_gates, features)], dim=1))
    first, second = logits(a), logits(b)
    difference = float((first - second).abs().max())
    assert difference == 0.0
    import evaluate
    dependencies = [Path(inspect.getfile(type(a))), Path(evaluate.__file__), Path(control.CONFIG)]
    previous_eval_time = min((RUN / f"branch_E/runs/B_full_{ds}/metrics.json").stat().st_mtime for ds in control.DATASETS)
    result["dependency_snapshots"] = [{"path": str(p), "sha256": control.digest(p), "mtime": p.stat().st_mtime,
                                       "modified_after_previous_B_eval": p.stat().st_mtime > previous_eval_time} for p in dependencies]
    assert not any(p["modified_after_previous_B_eval"] for p in result["dependency_snapshots"]), "An inference dependency changed since B evaluation; rerun B for a matched comparison"
    del a, b, first, second
    gc.collect()
    torch.cuda.empty_cache()
    natural = control.model(control.OFFICIAL, 20260907, "official_natural").eval()
    assert control.head_digest(natural) == original_head
    assert all(not p.requires_grad for p in natural.dino_encoder.parameters())
    result.update({"status": "pass", "strict_official_backbone_keys": len(natural.dino_encoder.state_dict()),
                   "decoder_initial_sha256": original_head, "official_and_B_decoder_initialization_identical": True,
                   "B_compatibility_logits_max_abs_diff": difference, "first_training_sample_identical": True,
                   "encoder_frozen": True})
    (OUT / "protocol_match.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
