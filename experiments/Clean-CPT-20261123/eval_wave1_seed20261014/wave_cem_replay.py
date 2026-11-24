"""Replay the frozen batch-4 CEM draw ledger from original TIFFs."""
from functools import lru_cache
import json
from pathlib import Path


def patch(trainer, hist_data):
    import numpy as np
    import tifffile

    from heads_targets import instance_target
    from model import BASE
    from mixture import C, ROOT

    source = ROOT / "MitoNet/dataset/EMPIAR-11037/cem_mitolab"
    ledger = BASE / "batch4/cem_draws.jsonl"

    @lru_cache(maxsize=512)
    def source_images(group, name):
        image = tifffile.imread(source / group / "images" / name)
        label = tifffile.imread(source / group / "masks" / name)
        assert image.ndim == 2 and image.dtype == np.uint8
        assert label.shape == image.shape and label.dtype.kind in "uib"
        return image, label

    class Replay(hist_data.CEMDraws):
        def __init__(self):
            self.prefix_sha = trainer.sha(ledger)
            assert self.prefix_sha == trainer.read(BASE / "batch4/ready.json")["draw_sha"]["cem"]
            data = trainer.read(C / "manifest.json")
            allowed = set(data["groups"]) - set(data["validation"])
            self.image_count = sum(len(data["groups"][g]) for g in allowed)
            image_sets = {g: {row[0] for row in data["groups"][g]} for g in allowed}
            self.draws = [json.loads(line) for line in ledger.read_text().splitlines()]
            assert len(self.draws) == 160000
            assert all(d == "cem" and z == 0 and g in allowed and n in image_sets[g]
                       for d, g, n, z, *_ in self.draws)

        def crop(self, i):
            _, group, name, _, y, x, flip_y, flip_x = self.draws[i]
            raw, ids = source_images(group, name)
            raw, ids = raw[y:y+256, x:x+256], ids[y:y+256, x:x+256]
            pads = tuple((0, 256-k) for k in raw.shape)
            raw = np.pad(raw, pads, constant_values=127)
            vals, inv = np.unique(ids, return_inverse=True)
            ids = np.pad((inv.reshape(ids.shape) + int(vals[0] != 0)).astype(np.int32),
                         pads, constant_values=-1)
            for axis, flag in enumerate((flip_y, flip_x)):
                if flag:
                    raw, ids = np.flip(raw, axis), np.flip(ids, axis)
            return raw, ids

        def sample(self, arm, i):
            raw, ids = self.crop(i)
            return raw, ids, instance_target(arm, ids)

    trainer.CEMDraws = Replay
    return source_images, Replay
