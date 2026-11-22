"""Native-grid semantic inference for the available manifest-matched HQ fields."""
import csv
import sys
from pathlib import Path
import numpy as np
import tifffile
import torch

OUT = Path(__file__).resolve().parent
BASE = OUT.parent / "mix_mitolab_mv_20261118"
DATA = Path("/opt/data/3_117/zhangyc/Large-Field-Workspace/mitos/HQ-5nm_20261120")
sys.path.insert(0, str(BASE))
from mixture import MB, read, sha, atomic
sys.path.insert(0, str(MB / "code"))
from common import gpu_guard, config

@torch.inference_mode()
def run(model):
    card = gpu_guard()
    torch.set_num_threads(2)
    seed = 20261014
    if model == "Official-MitoNet-v1":
        from infer import mitonet
        spec = next(r for r in config()["runs"] if r["model"] == model)
        ck = Path(spec["checkpoint_path"])
        assert sha(ck) == spec["checkpoint_sha256"]
        encoder = None
        predict = mitonet(spec)
    else:
        from fit import primus
        folder = BASE / "batch4/primus_cem" / model
        ck = folder / "model_40000.pt"
        meta = read(folder / "trained.json")
        assert meta["seed"] == seed and meta["steps"] == 40000 and sha(ck) == meta["checkpoint_sha"]
        net, p, encoder = primus(model)
        assert encoder == meta["init_encoder_sha"]
        missing, extra = net.load_state_dict(torch.load(ck,map_location="cpu",weights_only=True),strict=False)
        assert not extra and all(k.startswith("dino_encoder.") for k in missing)
        net.eval()
        predict = lambda a: p.sliding(net,a,tile_batch_size=16)
    folder = OUT / model
    folder.mkdir(exist_ok=True)
    manifest = DATA / "manifest.csv"
    assert sha(manifest) == read(DATA/"verification.json")["manifest_sha256"]
    rows = []
    for r in csv.DictReader(manifest.open(encoding="utf-8-sig")):
        path = DATA / r["file"]
        record = dict(id=r["id"],organ=r["organ"],file=r["file"],annotations=r["annotations"])
        if path.is_file():
            assert sha(path) == r["file_sha256"] and float(r["pixel_size_nm"]) == 5
            image = tifffile.imread(path)
            assert image.shape == (2048,2048) and image.dtype == np.uint8
            foreground = np.asarray(predict(image)) > 0
            assert foreground.shape == image.shape
            output = folder / (r["id"] + ".tif")
            tifffile.imwrite(output,foreground.astype(np.uint8),compression="deflate")
            assert np.array_equal(tifffile.imread(output),foreground)
            record.update(status="predicted",foreground_pixels=int(foreground.sum()),pixels=image.size)
        else:
            record.update(status="missing")
        rows.append(record)
    atomic(folder/"receipt.json",dict(model=model,gpu=card,checkpoint_sha=sha(ck),encoder_sha=encoder,
           inference="official Engine2d" if encoder is None else "Primus native-grid sliding",
           source_sha=sha(Path(__file__)),rows=rows))

if __name__ == "__main__":
    for name in sys.argv[1:]:
        run(name)
