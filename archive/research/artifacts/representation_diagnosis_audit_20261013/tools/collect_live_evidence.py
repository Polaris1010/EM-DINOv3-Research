"""Read-only server collection. No torch, CUDA, process control or training.

Run with explicit audit roots; stdout is the JSON artifact, for local redirection.
Paths discovered through symlinks outside declared roots are skipped.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,action="append",required=True)
    p.add_argument("--source-file",type=Path,action="append",default=[])
    p.add_argument("--hash-checkpoints",action="store_true")
    a=p.parse_args()
    result={"mode":"read_only_no_gpu","roots":[],"files":[],"checkpoints":[],"errors":[]}
    for rootarg in a.root:
        try: root=rootarg.resolve(strict=True)
        except OSError as e:result["errors"].append({"path":str(rootarg),"error":str(e)});continue
        result["roots"].append(str(root))
        candidates=set()
        for pattern in ["**/*resolved*.yaml","**/*resolved*.yml","**/B_full.yaml","**/metadata.json",
                        "**/*protocol*.json","**/*metrics.json","**/training_result*.json","**/initialization.json",
                        "**/steps_rank*.jsonl","**/*.pth"]:
            candidates.update(root.glob(pattern))
        for path in sorted(candidates):
            if not path.is_file() or not path.resolve().is_relative_to(root):continue
            stat=path.stat();row={"path":str(path),"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns}
            if path.suffix==".pth":
                if a.hash_checkpoints:
                    h=hashlib.sha256()
                    with path.open("rb") as f:
                        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
                    row["sha256"]=h.hexdigest()
                else:row["sha256"]="UNVERIFIED: hashing not requested"
                result["checkpoints"].append(row);continue
            if path.suffix==".jsonl":
                gram_keys=set();nonzero=0;lines=0;first=None;last=None;crop_keys=set()
                h=hashlib.sha256()
                with path.open("rb") as f:
                    for raw in f:
                        h.update(raw);lines+=1
                        try:data=json.loads(raw)
                        except ValueError:continue
                        first=data if first is None else first;last=data
                        for k,v in data.items():
                            if "gram" in k.lower():
                                gram_keys.add(k)
                                if isinstance(v,(int,float)) and not isinstance(v,bool) and v!=0:nonzero+=1
                            if "crop" in k.lower():crop_keys.add(k)
                        for k,v in data.get("losses",{}).items():
                            if "gram" in k.lower():
                                gram_keys.add("losses."+k)
                                if isinstance(v,(int,float)) and v!=0:nonzero+=1
                row.update(sha256=h.hexdigest(),line_count=lines,first=first,last=last,
                           gram_keys=sorted(gram_keys),nonzero_gram_entries=nonzero,crop_keys=sorted(crop_keys),
                           interpretation="No Gram key is not proof of disabled loss; inspect resolved config and trainer")
            elif stat.st_size<=2*1024*1024:
                raw=path.read_bytes();row.update(sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode("utf-8",errors="replace"))
            else:row["content"]="UNVERIFIED: metadata exceeded 2 MiB limit"
            result["files"].append(row)
    for path in a.source_file:
        try:
            raw=path.read_bytes()
            result["files"].append({"path":str(path),"sha256":hashlib.sha256(raw).hexdigest(),"text":raw.decode("utf-8")})
        except OSError as e:result["errors"].append({"path":str(path),"error":str(e)})
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
