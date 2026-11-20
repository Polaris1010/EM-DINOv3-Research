"""Summarize validated 3D volumes and the paired 2D reference using the same units."""
import csv
from collections import defaultdict
from run import BASE, OUT, jobs, metrics, np, read, summarize

def report():
    grouped=defaultdict(list)
    original=read(BASE/"zeroshot_repeat_20261120/Official-MitoNet-v1_20261014.json")["rows"]
    ref={(r["split"],r["name"],r["z"]):r for r in original}
    seen=set()
    pooled=[]
    for mode,r in jobs():
        split="mv3" if mode=="mv3" else "mb1"
        d=read(OUT/"scores"/(mode+"_"+r["id"]+".json"))
        for z,new in enumerate(d["rows"]):
            key=(split,r["id"],z)
            old=ref[key]
            assert new["tp"]+new["fn"]==old["tp"]+old["fn"]
            assert new["tn"]+new["fp"]==old["tn"]+old["fp"]
            assert key not in seen
            seen.add(key)
        if mode=="mb1":
            v=metrics([sum(x[k] for x in d["rows"]) for k in ("tp","fp","fn","tn")])
            pooled.append(dict(dataset=d["group"],**v))
    assert seen==set(ref) and len(seen)==2633
    summarize()
    with (OUT/"volumetric_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(pooled[0]))
        w.writeheader()
        w.writerows(pooled)
    with (OUT/"comparison.csv").open() as f:
        for r in csv.DictReader(f):
            grouped[(r["split"],r["group"])].append(r)
    text=["# Official MitoNet-v1 三轴推理语义评测","","MV3 使用 38 个开发区域的原有效真值，按 12 来源等权。MB-1 的 6 个体数据先求逐切片均值。TEM-100 独立二维推理，七组汇总按组等权。", "",
          "| 数据集 | 分组 | 3D Dice¹ | IoU | Precision | Recall | 原版 2D Dice | P0-cem | B100-cem | Barrier100-cem |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (split,g),rows in sorted(grouped.items()):
        new=next(r for r in rows if r["mode"]!="2D-reference")
        old={r["model"]:r for r in rows if r["mode"]=="2D-reference"}
        vals=[new[k] for k in ("dice","iou","precision","recall")]+[old[m]["dice"] for m in ("Official-MitoNet-v1","P0","B100","Barrier100")]
        text.append("| "+" | ".join([split,g]+[f"{float(v):.4f}" if v else "NA" for v in vals])+" |")
    text+=["","¹ TEM-100 的该列使用官方 2D 推理；MB-1 ALL 包含该组。VOLUMES 单列六个体数据。DINOv3 对照固定 CEM-MitoLab 40k 语义头 seed 20261014。逐体素池化结果见 volumetric_metrics.csv。","",
           "推理使用发布版 MitoNet_v1.pth、官方 Engine3d、三个正交方向与官方实例共识。所有体统一使用库默认参数。二维与三维默认阈值及后处理不同，因此差值反映完整推理流程的变化。"]
    (OUT/"report.md").write_text("\n".join(text)+"\n",encoding="utf-8")

if __name__=="__main__":
    report()
