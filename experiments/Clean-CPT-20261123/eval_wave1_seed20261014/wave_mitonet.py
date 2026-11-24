"""Score the released Official-MitoNet-v1 2D engine on the local held-out overlay."""
import sys
import numpy as np
import torch
from wave_train_v1 import HIST, ROOT, gpu_limit
from wave_eval import external, install_metrics


@torch.no_grad()
def main():
    gpu_limit(1, "panoptic")
    torch.set_num_threads(2)
    sys.path.insert(0, str(HIST))
    import evaluate_heads as ev
    import references as reference
    from infer import mitonet as official_engine
    options = next(r for r in reference.config()["runs"] if r["model"] == "Official-MitoNet-v1")
    from pathlib import Path
    assert reference.sha(Path(options["checkpoint_path"])) == options["checkpoint_sha256"]
    engine = official_engine(options)
    instance_stats = install_metrics(ev)
    out = ROOT / "mitonet_v1_2d"
    out.mkdir(exist_ok=True)

    def score(image, gt, valid):
        ids = np.asarray(engine(image))
        assert ids.shape == image.shape
        s = ev.semantic(ev.confusion(ids > 0, gt, valid))
        if valid is not None:
            gt, ids = np.where(valid, gt, 0), np.where(valid, ids, 0)
        return dict(Dice=s["dice"], IoU=s["iou"], **instance_stats(gt, ids))

    external(out, score)
    reference.atomic(out / "evaluated.json", dict(checkpoint_sha256=options["checkpoint_sha256"], gpu=1))


if __name__ == "__main__":
    main()
