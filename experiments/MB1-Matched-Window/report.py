from pathlib import Path
from collections import defaultdict
import json,csv
p=Path(__file__).resolve().parent
models=("P0","B100","C0_Vanilla_100k","Barrier100")
sources=defaultdict(list)
reference=None
for name in models:
    rows=json.loads((p/(name+".json")).read_text())["rows"]
    keys={(r["group"],r["id"],r["z"],r["window"],r["n"],r["gt"],r["inner"],r["padding"]) for r in rows}
    assert len(rows)==len(keys)==156
    if reference is None:
        reference=keys
    assert keys==reference
    for row in rows:
        sources[name,row["group"],row["window"],"all"].append(row)
        if not row["padding"] and not any(q["padding"] for q in rows if q["id"]==row["id"] and q["z"]==row["z"]):
            sources[name,row["group"],row["window"],"native_1024"].append(row)
result=[]
for (m,g,w,subset),rows in sorted(sources.items()):
    total={k:sum(r[k] for r in rows) for k in ("tp","gt","fp","inner","inner_missed","enclosed_missed","fg_margin_sum")}
    div=lambda a,b:a/b if b else None
    result.append(dict(model=m,group=g,window=w,subset=subset,n=len(rows),padding=sum(r["padding"] for r in rows),
          dice=div(2*total["tp"],total["gt"]+total["tp"]+total["fp"]),
          recall=div(total["tp"],total["gt"]),inner_miss=div(total["inner_missed"],total["inner"]),
          enclosed_miss=div(total["enclosed_missed"],total["inner"]),fg_margin=div(total["fg_margin_sum"],total["gt"])))
with (p/"summary.csv").open("w",newline="") as f:
    out=csv.DictWriter(f,fieldnames=list(result[0]))
    out.writeheader()
    out.writerows(result)
for subset in ("all","native_1024"):
    print(subset, sorted({r["group"] for r in result if r["subset"]==subset}))
    for model in models:
        out=[]
        for w in (256,512,1024):
            selected=[r for r in result if r["model"]==model and r["window"]==w and r["subset"]==subset]
            out.append([round(sum(s)/len(s),4) for k in ("dice","recall","inner_miss","enclosed_miss","fg_margin")
                        if (s:=[r[k] for r in selected if r[k] is not None])])
        print(model,out)
