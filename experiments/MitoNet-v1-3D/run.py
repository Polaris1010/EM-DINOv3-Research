"""Official MitoNet-v1 orthoplane inference and matched native-grid semantic scoring."""
import ast
import csv
import inspect
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import tifffile
import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from empanada.data import VolumeDataset
from empanada.inference import filters
from empanada.inference.engines import PanopticDeepLabRenderEngine3d
from empanada.inference.tracker import InstanceTracker
from empanada.inference.patterns import (create_matchers, forward_matching, backward_matching, update_trackers,
    finish_tracking, get_axis_trackers_by_class, create_instance_consensus, create_semantic_consensus, fill_volume)
OUT=Path(__file__).resolve().parent
BASE=OUT.parent/"mix_mitolab_mv_20261118"
sys.path.insert(0,str(BASE))
from mixture import M, MB, read, sha, arrays
sys.path.insert(0,str(MB/"code"))
from common import gpu_guard, config, M4, atomic, Planes
from infer import mitonet
from evaluate import metrics

def official(spec):
    assets=M4/"assets"
    locks=read(assets/"official_assets.json")["files"]
    ns=dict(globals())
    for filename,names in [("utils_upstream.py",{"normalize","to_tensor","Preprocessor","load_model_to_device"}),
                           ("inference_upstream.py",{"Engine3d","tracker_consensus"})]:
        f=assets/filename
        assert sha(f)==locks[filename]["sha256"]
        tree=ast.parse(f.read_text())
        nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        assert {n.name for n in nodes}==names
        for n in nodes:
            if n.name=="tracker_consensus":
                assert len(n.decorator_list)==1 and n.decorator_list[0].id=="thread_worker"
                n.decorator_list=[]  # consume official generator synchronously
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(f),"exec"),ns)
    f=assets/"model_config.yaml"
    assert sha(f)==locks[f.name]["sha256"]
    mc=yaml.safe_load(f.read_text())
    mc["model"]=str(spec["checkpoint_path"])
    engine=ns["Engine3d"](mc)
    return engine,ns["tracker_consensus"],mc

def counts(pred,gt,valid=None):
    a=torch.as_tensor(np.asarray(pred)>0,device="cuda")
    b=torch.as_tensor(np.asarray(gt)>0,device="cuda")
    assert a.shape==b.shape
    if valid is not None:
        v=torch.as_tensor(np.array(valid),device="cuda")>0
        a,b=a[v],b[v]
    tp,ng,npred=torch.stack(((a&b).sum(),b.sum(),a.sum())).tolist()
    return np.array((tp,npred-tp,ng-tp,a.numel()-npred-ng+tp),dtype=np.int64)

def jobs():
    entries=read(MB/"MB1_benchmark_manifest.json")["entries"]
    volumes=[("mb1",r) for r in entries if r["dataset"]!="TEM-100"]
    mv=[("mv3",r) for r in read(M/"MV-Generalist.json")["entries"] if r["supervised_role"]=="dev"]
    tem=[("tem",r) for r in entries if r["dataset"]=="TEM-100"]
    assert (len(volumes),len(mv),len(tem))==(6,38,100)
    return volumes+mv+tem

@torch.inference_mode()
def worker(rank):
    gpu=gpu_guard()
    torch.set_num_threads(2)
    spec=next(r for r in config()["runs"] if r["model"]=="Official-MitoNet-v1")
    assert sha(Path(spec["checkpoint_path"]))==spec["checkpoint_sha256"]
    engine,consensus,mc=official(spec)
    settings=dict(model_sha=spec["checkpoint_sha256"],axes=list(engine.axes),engine=str(inspect.signature(type(engine))),
                  consensus=str(inspect.signature(consensus)),source_sha=sha(Path(__file__)),gpu=gpu)
    atomic(OUT/f"protocol_{rank}.json",settings)
    tem_model=None
    for mode,r in jobs()[rank::2]:
        key=r["id"]
        path=OUT/"scores"/(mode+"_"+key+".json")
        if path.exists():
            continue
        print("RUN",mode,key,flush=True)
        if mode=="tem":
            if tem_model is None:
                tem_model=mitonet(spec)
            a,b=[Planes(r[k],r["shape_zyx"]) for k in ("image_path","label_path")]
            rows=[metrics(counts(tem_model(a[0]),b[0]))]
            a.close()
            b.close()
        else:
            if mode=="mb1":
                a=tifffile.imread(r["image_path"],out="memmap")
                assert tuple(a.shape)==tuple(r["shape_zyx"])
            else:
                a,b,v,j=arrays(key)
            trackers={}
            for axis in engine.axes:
                _,trackers[axis]=engine.infer_on_axis(a,axis)
            pred,_,_=next(consensus(trackers,None,mc))
            assert pred.shape==a.shape
            np.savez_compressed(OUT/"pred"/(mode+"_"+key+".npz"),foreground_bits=np.packbits(pred>0,axis=-1),shape=pred.shape)
            if mode=="mb1":
                b=Planes(r["label_path"],r["shape_zyx"])
                rows=[metrics(counts(pred[z],b[z])) for z in range(len(a))]
                b.close()
            else:
                c=sum((counts(pred[z],b[z],v[z] if v is not None else None) for z in j["valid_z"]),np.zeros(4,np.int64))
                rows=[metrics(c)]
            del pred,trackers,a
        atomic(path,dict(mode=mode,id=key,group=r.get("dataset",r.get("source_id")),rows=rows))
        print("DONE",key,flush=True)

def summarize():
    gs=defaultdict(list)
    for mode,r in jobs():
        d=read(OUT/"scores"/(mode+"_"+r["id"]+".json"))
        gs["mb1" if mode=="tem" else mode,d["group"]].extend(d["rows"])
    keys=("dice","iou","precision","recall")
    cols=["model","split","group","mode",*keys]
    table=[]
    def reduce(rows):
        return {k:float(np.mean(vals)) if (vals:=[x[k] for x in rows if x[k] is not None]) else None for k in keys}
    for (split,group),rows in gs.items():
        table.append(dict(model="Official-MitoNet-v1",split=split,group=group,mode="2D" if group=="TEM-100" else "3D",**reduce(rows)))
    ref=BASE/"zeroshot_repeat_20261120"
    for name in ("Official-MitoNet-v1","P0","B100","Barrier100"):
        shared=defaultdict(list)
        for r in read(ref/f"{name}_20261014.json")["rows"]:
            shared[r["split"],r["group"]].append(r)
        for (split,group),rows in shared.items():
            table.append(dict(model=name,split=split,group=group,mode="2D-reference",**reduce(rows)))
    parts=defaultdict(list)
    for r in table:
        parts[r["model"],r["split"],r["mode"]=="2D-reference"].append(r)
    for (model,split,ref),rows in parts.items():
        for label,subset in [("ALL",rows),("VOLUMES",[r for r in rows if r["group"]!="TEM-100"] if split=="mb1" else [])]:
            if subset:
                table.append(dict(model=model,split=split,group=label,mode="2D-reference" if ref else ("3D+2D" if split=="mb1" and label=="ALL" else "3D"),**reduce(subset)))
    with (OUT/"comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols)
        w.writeheader()
        w.writerows(table)

if __name__=="__main__":
    if sys.argv[1]=="report":
        summarize()
    else:
        worker(int(sys.argv[1]))
