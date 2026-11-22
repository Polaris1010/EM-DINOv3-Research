from pathlib import Path
from collections import defaultdict
import csv
import sys
import numpy as np
import tifffile
from scipy import ndimage
from PIL import Image
from model import BASE, OUT, build, read, sha, torch
from mixture import M, MB, C, ROOT, arrays, instance, atomic
from common import Planes
from evaluate import metrics
from instance_repeat import score
from inference import predict

def touching(gt, pred):
    # A touching pair shares a pixel edge; each ID votes by greatest overlapping nonzero prediction.
    pairs=[]
    for a,b in ((gt[:-1],gt[1:]),(gt[:,:-1],gt[:,1:])):
        active=(a>0)&(b>0)&(a!=b)
        if active.any():
            pairs.append(np.sort(np.stack((a[active],b[active]),1),axis=1))
    if not pairs:
        return 0,0
    pair=np.unique(np.concatenate(pairs),axis=0)
    mapping={}
    for label in np.unique(pair):
        ids,counts=np.unique(pred[gt==label],return_counts=True)
        valid=ids>0
        mapping[label]=ids[valid][counts[valid].argmax()] if valid.any() else 0
    separated=sum(bool(mapping[a] and mapping[b] and mapping[a]!=mapping[b]) for a,b in pair)
    return len(pair),separated

def example(path,image,gt,sem,pred,base):
    palette=np.random.default_rng(20261014).integers(35,255,(4096,3),dtype=np.uint8)
    def color(ids):
        return np.where((ids>0)[...,None],palette[ids.astype(int)%4096],0)
    gray=np.asarray(image,np.float32)
    low,high=np.percentile(gray,[1,99])
    gray=np.clip((gray-low)/max(high-low,1)*255,0,255).astype(np.uint8)
    panels=[np.repeat(gray[:,:,None],3,axis=2),color(gt),np.repeat((sem*255).astype(np.uint8)[:,:,None],3,axis=2),color(base),color(pred)]
    Image.fromarray(np.concatenate(panels,axis=1)).save(path.with_suffix(".png"))
    np.savez_compressed(path.with_suffix(".npz"),semantic=sem,instance=pred)

@torch.inference_mode()
def main():
    net,p,stage,head,receipt=build()
    ck=OUT/"instance_40000.pt"
    assert sha(ck)==read(OUT/"trained_40000.json")["sha256"]
    head.load_state_dict(torch.load(ck,weights_only=True))
    head.eval()
    dest=OUT/"scores_40000"
    dest.mkdir(exist_ok=True)
    old={(r["split"],r["name"],r["z"]):r for r in read(BASE/"batch4/primus_cem/Barrier100/evaluation.json")}
    def block(split,group,key,planes,region=False):
        fn=dest/(split+"_"+key+".json")
        if fn.exists():
            return
        rows=[]
        conf=np.zeros(4,np.int64)
        for j,(name,z,image,gt,valid,reliable) in enumerate(planes):
            sem,pred,fallback=predict(net,p,stage,head,image)
            if j==0:
                assert np.array_equal(sem,p.sliding(net,image,tile_batch_size=16))
            mask=np.ones_like(sem) if valid is None else valid.astype(bool)
            s=sem[mask]
            b=gt[mask]>0
            c=np.array([np.count_nonzero(s&b),np.count_nonzero(s&~b),np.count_nonzero(~s&b),np.count_nonzero(~s&~b)])
            conf+=c
            row=dict(name=name,z=z,**metrics(c),fallback=fallback)
            if not region:
                original=old[split,name,z]
                assert all(original[k]==row[k] for k in ("tp","fp","fn","tn"))
            if reliable:
                gt=np.where(mask,gt,0)
                pred=np.where(mask,pred,0)
                base=ndimage.label(sem & mask,structure=np.ones((3,3),int))[0]
                for method,ids in (("head",pred),("cc",base)):
                    result=score(gt,ids)
                    n,sep=touching(gt,ids)
                    row.update({method+"_"+k:float(v) for k,v in result.items()})
                    row[method+"_count_mae"]=abs(result["n_pred"]-result["n_gt"])
                    row[method+"_touch_pairs"]=n
                    row[method+"_separated"]=sep
                if j==0 and (split!="mb1" or group!="tem" or not list(dest.glob("mb1_tem*.png"))):
                    example(fn,image,gt,sem,pred,base)
            rows.append(row)
        if region:
            original=old[split,key,0]
            assert conf.tolist()==[original[k] for k in ("tp","fp","fn","tn")]
        atomic(fn,dict(split=split,group=group,key=key,semantic=metrics(conf),rows=rows))
    for r in read(MB/"MB1_benchmark_manifest.json")["entries"]:
        a,b=[Planes(r[k],r["shape_zyx"]) for k in ("image_path","label_path")]
        block("mb1",r["dataset"],r["id"],((r["id"],z,a[z],b[z],None,True) for z in range(r["shape_zyx"][0])))
        a.close()
        b.close()
    cem=read(C/"manifest.json")
    for group in cem["validation"]:
        def planes():
            for name,_,_ in cem["groups"][group]:
                image,sem=np.load(Path("/opt/cem_mitolab_20261118")/group/(name+".npy"))
                gt=tifffile.imread(ROOT/"MitoNet/dataset/EMPIAR-11037/cem_mitolab"/group/"masks"/name)
                assert np.array_equal(gt>0,sem>0)
                yield name,0,image,gt,None,True
        block("cem_validation",group,group,planes())
    for r in read(M/"MV-Generalist.json")["entries"]:
        if r["supervised_role"]!="dev":
            continue
        a,b,v,j=arrays(r["id"])
        oz,oy,ox=(q[0] for q in r["region_zyx"])
        reliable=r.get("annotation_contract")=="instance_proofread"
        ids=instance(r["image_path"]) if reliable else None
        def mvplanes():
            for z in j["valid_z"]:
                h,w=a[z].shape
                mask=v[z]>0 if v is not None else None
                gt=b[z] if ids is None else ids[z+oz,oy:oy+h,ox:ox+w]
                ok=(gt>0)==(b[z]>0)
                assert np.all(ok if mask is None else ok[mask])
                yield r["id"],int(z),a[z],gt,mask,reliable
        block("mv3",r["source_id"],r["id"],mvplanes(),True)
    summarize(dest)

def summarize(dest):
    grouped=defaultdict(lambda:defaultdict(list))
    contact=defaultdict(lambda:np.zeros(4,dtype=np.int64))
    keys=["dice","iou"]+[m+"_"+k for m in ("head","cc") for k in ("AP_50","AP_75","PQ","F1_50","F1_75","count_mae")]
    for path in sorted(dest.glob("*.json")):
        d=read(path)
        region=d["split"]=="mv3"
        for row in d["rows"]:
            if "head_touch_pairs" in row:
                contact[d["split"]]+=np.asarray([row["head_touch_pairs"],row["head_separated"],row["cc_separated"],row["fallback"]],dtype=np.int64)
        for k in keys:
            vals=[r[k] for r in d["rows"] if k in r]
            if region and k in ("dice","iou"):
                vals=[d["semantic"][k]]
            elif region and vals:
                vals=[np.mean(vals)]
            grouped[d["split"],d["group"]][k].extend(vals)
    rows=[dict(split=s,group=g,**{k:np.mean(v) for k,v in cols.items() if v}) for (s,g),cols in grouped.items()]
    assert len(rows)==49
    for split in ("mb1","mv3","cem_validation"):
        part=[r for r in rows if r["split"]==split]
        rows.append(dict(split=split,group="ALL",**{k:np.mean([r[k] for r in part if k in r]) for k in keys if any(k in r for r in part)}))
    with (OUT/"metrics_40000.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["split","group"]+keys)
        writer.writeheader()
        writer.writerows(rows)
    atomic(OUT/"evaluation_complete.json",{"summaries":[r for r in rows if r["group"]=="ALL"]})
    lines=["# Barrier100-CEM 二维实例分支 40k","",
           "固定语义前景，逐切片比较学习的中心与偏移分组和八连通域。下表按组等权汇总，MV3 实例指标仅覆盖拥有合格实例标注的来源。","","| 数据 | Dice | IoU | AP50 实例头 / 连通域 | AP75 实例头 / 连通域 | PQ 实例头 / 连通域 | 实例数 MAE 实例头 / 连通域 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["group"]=="ALL":
            values=[r["split"],f"{r['dice']:.4f}",f"{r['iou']:.4f}"]
            for metric in ("AP_50","AP_75","PQ","count_mae"):
                values.append(" / ".join(f"{r[m+'_'+metric]:.4f}" if m+"_"+metric in r else "NA" for m in ("head","cc")))
            lines.append("| "+" | ".join(values)+" |")
    lines+=["","## 接触实例","","按切片累计，共享像素边界的两个真值实例分别获得不同主预测 ID 计为分开。"]
    for split,(pairs,learned,cc,fallback) in sorted(contact.items()):
        lines.append(f"- {split} 有 {pairs} 对接触实例；实例头分开 {learned} 对，连通域分开 {cc} 对。有前景但无中心的切片共 {fallback} 张。")
    lines+=["","完整逐来源结果见 metrics_40000.csv。scores_40000 保存逐切片结果及预测图。语义混淆矩阵逐项对照已有 Barrier100-CEM 结果通过。"]
    (OUT/"final_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
