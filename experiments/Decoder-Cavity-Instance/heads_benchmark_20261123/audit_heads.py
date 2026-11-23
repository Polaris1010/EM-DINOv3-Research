import hashlib
import json
from pathlib import Path
import tarfile


def load(p):
    return json.loads(p.read_text())


def hash_file(p):
    with p.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def collect(directory):
    keys = {"mb1":set(),"cem_validation":set(),"hq":set()}
    for p in directory.glob("*.jsonl"):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            split = r["split"]
            k = (r["group"],r["id"],r["z"])
            assert k not in keys[split],k
            keys[split].add(k)
    assert {k:len(v) for k,v in keys.items()}=={"mb1":2595,"cem_validation":1424,"hq":20}
    return keys


def main():
    here = Path(__file__).resolve().parent
    previous = here.parent/"decoder_cavity_inst_20261123"
    assert (here/"evaluation_complete.txt").is_file()
    config = load(here/"config.json")
    h = hash_file(here/config["encoder"])
    assert h==config["encoder_sha256"]
    expected = {"mb1":set(),"cem_validation":set(),"hq":set()}
    for directory in (previous/"arm_a_multiscale_sem"/"evaluations"/"stride120").iterdir():
        if directory.name not in expected:
            continue
        for f in directory.glob("*.jsonl"):
            for r in map(json.loads,f.read_text().splitlines()):
                expected[directory.name].add((r["group"],r["id"],r["z"]))
    runs = {}
    names = []
    for arm in config["heads"]:
        directory = here/arm
        assert collect(directory/"eval_units")==expected
        smoke = load(directory/"smoke_pass.json")
        train = load(directory/"trained.json")
        eval_receipt = load(directory/"evaluated.json")
        assert train["weights_sha256"]==hash_file(directory/"best.pt")
        assert load(directory/"locked.json")["params"]==eval_receipt["params"]
        assert all(x["gpu"] in (1,2,3) for x in (smoke,train,eval_receipt))
        assert len(list((directory/"smoke_targets").glob("*.png")))==5
        runs[arm] = dict(smoke=smoke,train=train,eval=eval_receipt)
        names.extend(arm+"_"+kind+".csv" for kind in ("cem_grid","semantic","instance","hq"))
    assert collect(here/"mitonet_reference"/"eval_units")==expected
    names += ["primus_cem_reference.csv",*(f"mitonet_v1_{tag}.csv" for tag in ("semantic","instance","hq"))]
    metadata = dict(encoder_sha256=h,matched_ids=True,runs=runs,
                    primus_reference=load(here/"primus_reference.json"),
                    mitonet_reference=load(here/"mitonet_reference.json"))
    receipt = here/"delivery_receipts.json"
    receipt.write_text(json.dumps(metadata,indent=2))
    names.append(receipt.name)
    tmp = here/"lightweight_transfer.tmp"
    server_prefix = "/".join(("", "opt", "data", ""))
    with tarfile.open(tmp,"w:gz") as package:
        for name in names:
            p = here/name
            assert 0 < p.stat().st_size < 2000000
            assert server_prefix not in p.read_text(encoding="utf-8")
            package.add(p,arcname=name)
    final = here/"lightweight_transfer.tar.gz"
    tmp.replace(final)
    print(hash_file(final),flush=True)


if __name__=="__main__":
    main()
