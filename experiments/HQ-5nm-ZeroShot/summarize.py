import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
models = ("Official-MitoNet-v1","Barrier100","B100","P0")
rows = []
summary = []
reference = None
for model in models:
    receipt = json.loads((root/model/"receipt.json").read_text())
    fields = [(r["id"],r["status"],r["annotations"]) for r in receipt["rows"]]
    if reference is None:
        reference = fields
    assert fields == reference and all(a=="none" for _,_,a in fields)
    valid = [r for r in receipt["rows"] if r["status"]=="predicted"]
    for r in valid:
        rows.append(dict(model=model,**r,foreground_fraction=r["foreground_pixels"]/r["pixels"]))
    for organ in ["ALL"] + sorted({r["organ"] for r in valid}):
        selected = [r for r in valid if organ=="ALL" or r["organ"]==organ]
        summary.append(dict(model=model,organ=organ,images=len(selected),
                       foreground_fraction=sum(r["foreground_pixels"] for r in selected)/sum(r["pixels"] for r in selected)))
for name,records in [("per_field.csv",rows),("summary.csv",summary)]:
    with (root/name).open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
lines = ["# HQ-5nm 零样本二维语义预测","",
         f"四组共处理 {len(rows)} 个模型－图像组合。源清单 {len(reference)} 张，实际可读 {len(valid)} 张，像素尺寸 5 nm。缺失 ID 为 "+", ".join(i for i,status,_ in reference if status=="missing")+"。",
         "源清单将全部视野标记为无标注。Dice、IoU、Precision、Recall 均缺少逐像素真值，留空。下表仅描述预测前景占比。","",
         "| 模型 | 图像数 | 预测前景占比 |","|---|---:|---:|"]
for s in summary:
    if s["organ"]=="ALL":
        lines.append(f"| {s['model']} | {s['images']} | {s['foreground_fraction']:.2%} |")
lines+=["","MitoNet 使用官方发布版权重和 Engine2d。Barrier100、B100、P0 分别搭配已有 CEM-MitoLab 40k Primus 语义头，固定 decoder seed 20261014。原始网格推理，未使用目标域标签、未调参。每张预测以原始 2048×2048 的 0/1 TIFF 保存。"]
(root/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
