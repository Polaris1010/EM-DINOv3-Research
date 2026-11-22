from collections import defaultdict
from pathlib import Path
import csv,json,statistics as st,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parent
configs=('C1 Baseline','C1 Stride-120','C1 Dual-Phase','MitoNet-v1')
keys=('instances','missed_instances','detected_instances','eligible_instances','skipped_instances','affected_instances','interior_px','interior_fn_px','closed_hole_px','open_cavity_px','large_cavities','aligned_edges','total_edges')
rates=('missed_instance_rate','skipped_instance_rate','interior_fn_ratio','closed_share','open_share','affected_instance_rate','affected_eligible_rate','large_cavity_grid_align')
def div(a,b):return a/b if b else None

def summaries(rows):
    m={k:sum(r[k] for r in rows) for k in keys}
    result=dict(missed_instance_rate=div(m['missed_instances'],m['instances']),
         skipped_instance_rate=div(m['skipped_instances'],m['detected_instances']),
         interior_fn_ratio=div(m['interior_fn_px'],m['interior_px']),
         closed_share=div(m['closed_hole_px'],m['interior_fn_px']),
         open_share=div(m['open_cavity_px'],m['interior_fn_px']),
         affected_instance_rate=div(m['affected_instances'],m['detected_instances']),
         affected_eligible_rate=div(m['affected_instances'],m['eligible_instances']),
         large_cavity_grid_align=div(m['aligned_edges'],m['total_edges']))
    return dict(planes=len(rows),**m,**result)

def write(name,rows):
    if not rows:return
    p=O/name;tmp=p.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    tmp.replace(p)
def avg(a,k):
    v=[r[k] for r in a if r[k] is not None]
    return st.mean(v) if v else None
def fm(v):return 'NA' if v is None else f'{v:.2%}'

def render(top):
    colors=np.array([[0,0,0],[30,225,75],[255,225,0],[255,42,42],[10,125,255],[175,175,175]],np.uint8)
    panel_list=[]
    for group,values in sorted(top.items()):
        for ratio,n,id,z in values:
            path=O/'examples'/f'{group}__{id}__{z}.npz'
            with np.load(path) as d:raw,overlay=d['raw'],d['overlays']
            gray=Image.fromarray(raw).convert('RGB')
            width,height=560,round(raw.shape[0]/raw.shape[1]*560)
            if height>720:width,height=max(1,round(width*720/height)),720
            pad=10; x0=pad
            im=Image.new('RGB',(5*(width+pad)+pad,height+85),'#151515')
            d=ImageDraw.Draw(im)
            font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
            title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',19)
            d.text((pad,5),f'{group} | {id}  z={z}  |  Baseline GT-interior FN: {ratio:.1%}',font=title,fill='white')
            for j,(legend,c) in enumerate(zip(('TP','Boundary FN','Interior FN','Background FP','Whole instance missed'),colors[1:])):
                x=pad+j*(width+pad)
                d.rectangle((x,38,x+15,53),fill=tuple(int(v) for v in c))
                d.text((x+22,36),legend,font=font,fill='white')
            ims=[gray.resize((width,height),Image.Resampling.LANCZOS)]
            ims+=[Image.fromarray(colors[x]).resize((width,height),Image.Resampling.NEAREST) for x in overlay]
            for label,i in zip(('Original EM',*configs),ims):
                d.text((x0,62),label,font=font,fill='white')
                im.paste(i,(x0,84));x0+=width+pad
            dest=O/'visual_panels'/f'{group}__{id}__z{z}.png'
            im.save(dest)
            panel_list.append(dict(group=group,id=id,z=z,baseline_r3_interior_fn_ratio=ratio,png=dest.relative_to(O).as_posix()))
    assert len(panel_list)==12
    write('visual_index.csv',panel_list)
    return panel_list

def main():
    data=[]
    complete=(O/'receipt.json').is_file()
    for f in (O/'units').glob('*'):
        for line in f.read_text().splitlines():
            try:unit=json.loads(line)
            except json.JSONDecodeError:
                if '--partial' in sys.argv:continue
                raise
            assert len(unit)==8
            data+=unit
    if not data:return
    a=defaultdict(list)
    for r in data:a[r['config'],r['group'],r['radius']].append(r)
    by=[dict(config=c,group=g,radius=r,**summaries(v)) for (c,g,r),v in sorted(a.items())]
    allrows=[]
    for c in configs:
        for r in (3,5):
            rows=[v for v in by if v['config']==c and v['radius']==r]
            if not rows:continue
            n=sum(x['planes'] for x in rows)
            allrows.append(dict(config=c,radius=r,scope='ALL' if n==2595 and len(rows)==7 else 'PARTIAL',planes=n,
                groups=len(rows),**{k:avg(rows,k) for k in rates},**{k:sum(x[k] for x in rows) for k in keys}))
    if complete:
        assert len(data)==2595*8 and len({(r['config'],r['id'],r['z'],r['radius']) for r in data})==len(data)
        assert all(v['planes']==2595 and v['groups']==7 for v in allrows)
        write('per_plane.csv',data)
    write('summary_by_group.csv',by);write('summary_all.csv',allrows)
    receipt=json.loads((O/'receipt.json').read_text()) if complete else {}
    panels=render(json.loads((O/'examples'/'top.json').read_text())) if complete and '--partial' not in sys.argv else []
    p=json.loads((O/'protocol.json').read_text())
    text=['# MB-1 去边界内部漏洞评测','',
          f'状态 **{"完整" if complete else "进行中"}**。物理 GPU {p["gpu"]} 复算三种 C1 二维推理并评分，CPU 计算 GT 形态与统计。实际总耗时 {receipt.get("seconds",0):.2f} 秒。所有 C1 掩膜逐图复核与原全量评测的混淆计数一致。',
          'MitoNet-v1 的六个体数据直接使用已有的官方三轴 Engine3d 共识掩膜。TEM-100 使用官方 Engine2d 复算。表中 MitoNet 是三维+二维混合参照。此轮未纳入可选四相位。',
          '保留逐切片 GT 实例 ID。实例 recall <10% 单独计为整颗漏检。已检出实例逐个以 disk(3/5) 腐蚀。内部漏分率在组内按符合条件的内部核像素汇总。封闭孔以预测掩膜的 4 连通背景填孔判断，其他内部漏分计为开放掏空。',
          '受影响实例分母采用全部已检出实例，腐蚀后无核的实例不计为受影响。CSV 另给仅在可评估实例中的受影响率以及无核跳过率。网格对齐统计的是面积至少 64 像素的内部漏分连通域的四条外接框边缘，距 16 像素格线不超过 1 像素。组间七组等权，分母为零的率记 NA。',
          '', '| 配置 | 半径 | 视野 | 内部漏分 | 封闭 / 开放 | 受影响实例 | 整颗漏检 | 无核跳过 | 大漏洞网格对齐 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in allrows:
        text.append(f'| {r["config"]} | {r["radius"]} | {r["planes"]} | {fm(r["interior_fn_ratio"])} | {fm(r["closed_share"])} / {fm(r["open_share"])} | {fm(r["affected_instance_rate"])} | {fm(r["missed_instance_rate"])} | {fm(r["skipped_instance_rate"])} | {fm(r["large_cavity_grid_align"])} |')
    text+=['','## 七组 r=3','',
           '| 组 | C1 Baseline 漏分 / 受影响 | C1 Stride-120 | C1 Dual-Phase | MitoNet-v1 |','|---|---:|---:|---:|---:|']
    for g in sorted({r['group'] for r in by}):
        cells=[]
        for c in configs:
            r=next((r for r in by if r['group']==g and r['config']==c and r['radius']==3),None)
            cells.append(f'{fm(r["interior_fn_ratio"])} / {fm(r["affected_instance_rate"])}' if r else 'NA')
        text.append('| '+' | '.join([g]+cells)+' |')
    text+=['','## 人工核对图','',
           '原图加四列预测。绿色为真阳性，黄色为检测到的实例边界带漏分，红色为其内部漏分，蓝色为背景假阳性。灰色单列整颗漏检实例。六个指定来源分别选 Baseline r=3 最高的两张视野，四个配置显示相同视野。']
    for r in panels:text.append(f'- [{r["group"]} · {r["id"]} z={r["z"]}]({r["png"]})')
    text+=['','[汇总 CSV](summary_all.csv) · [逐组 CSV](summary_by_group.csv) · [逐切片 CSV](per_plane.csv)']
    (O/'GT_INTERIOR_HOLES_SUMMARY.md').write_text('\n'.join(text)+'\n',encoding='utf-8')

if __name__=='__main__':
    main()
