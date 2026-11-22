from collections import defaultdict,Counter
from pathlib import Path
import csv,json,statistics as st
O=Path(__file__).resolve().parent
DIAG=O.parent/'square_hole_diagnosis_20261122'
OLD=O.parent/'barrier_primus_instance_20261120'
ARMS=('arm_a_multiscale_sem','arm_b_multiscale_bcd')
RUNS=('stride120','baseline128')
C=('instances','missed_instances','detected_instances','eligible_instances','skipped_instances','affected_instances','interior_px','interior_fn_px','closed_hole_px','open_cavity_px','large_cavities','aligned_edges','total_edges')
I=('AP50','AP75','PQ','MAE')
S=('dice','iou','precision','recall')

def readcsv(p):
    with p.open() as h:return list(csv.DictReader(h))
def read(p):return json.loads(p.read_text())
def write(p,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (O/p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def mean(rows,k):
    vals=[float(v) for r in rows if (v:=r.get(k)) is not None and v!='']
    return st.mean(vals) if vals else None
def div(a,b):return a/b if b else None

def cavity(rows):
    out={}
    for rad in (3,5):
        part=[r for row in rows for r in row.get('cavity',[]) if r['radius']==rad]
        if not part:continue
        m={k:sum(r[k] for r in part) for k in C}
        ratio=dict(interior_fn_ratio=div(m['interior_fn_px'],m['interior_px']),closed_share=div(m['closed_hole_px'],m['interior_fn_px']),
              open_share=div(m['open_cavity_px'],m['interior_fn_px']),affected_instance_rate=div(m['affected_instances'],m['detected_instances']),
              missed_instance_rate=div(m['missed_instances'],m['instances']),skipped_instance_rate=div(m['skipped_instances'],m['detected_instances']),
              large_cavity_grid_align=div(m['aligned_edges'],m['total_edges']))
        out.update({f'{k}_r{rad}':v for k,v in (m|ratio).items()})
    return out

def aggregate(arm,window):
    root=O/arm/'evaluations'/window
    groups=defaultdict(list)
    counts=Counter()
    for f in sorted(root.glob('*/*.jsonl')):
        r=[json.loads(x) for x in f.read_text().splitlines()]
        split,group=r[0]['split'],r[0]['group']
        counts[split]+=len(r)
        if split=='mv3':
            tp,fp,fn=[sum(x['semantic'][k] for x in r) for k in ('tp','fp','fn')]
            r[0]['mv3_region']=dict(dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,precision=div(tp,tp+fp),recall=div(tp,tp+fn))
        groups[split,group]+=r
    assert counts['mb1']==2595 and counts['cem_validation']==1424 and counts['hq']==20,counts
    sem=[];inst=[];hq=[]
    for (split,group),r in sorted(groups.items()):
        tag=dict(arm=arm,window=window,split=split,group=group,n=len(r))
        if split=='hq':
            hq.extend(r);continue
        d=([x['mv3_region'] for x in r if 'mv3_region' in x] if split=='mv3' else [x['semantic'] for x in r])
        sem.append(dict(**tag,**{k:mean(d,k) for k in S},**cavity(r)))
        for method in ('cc','watershed'):
            v=[x[method] for x in r if method in x]
            if v:inst.append(dict(**(tag|dict(n=len(v))),method=method,**{k:mean(v,k) for k in I},
                                  touching_pairs=sum(x['touching_pairs'] for x in v),separated_pairs=sum(x['separated_pairs'] for x in v)))
    for split in ('cem_validation','mb1','mv3'):
        z=[r for r in sem if r['split']==split]
        sem.append(dict(arm=arm,window=window,split=split,group='ALL',n=sum(r['n'] for r in z),
            **{k:mean(z,k) for k in z[0] if k in S or k.rsplit('_r',1)[0] in ('interior_fn_ratio','closed_share','open_share','affected_instance_rate','missed_instance_rate','skipped_instance_rate','large_cavity_grid_align')}))
        for method in ('cc','watershed'):
            z=[r for r in inst if r['split']==split and r['method']==method]
            if z:inst.append(dict(arm=arm,window=window,split=split,group='ALL',method=method,n=sum(r['n'] for r in z),
                              **{k:mean(z,k) for k in I},touching_pairs=sum(r['touching_pairs'] for r in z),separated_pairs=sum(r['separated_pairs'] for r in z)))
    return sem,inst,hq

def old_stats():
    old=dict()
    cavity0=readcsv(DIAG/'gt_interior_holes/summary_all.csv')
    qc=readcsv(DIAG/'full_phase_eval/mb1_summary.csv')
    mp=readcsv(O.parent/'mitonet_v1_orthoplane_20261120/comparison.csv')
    for k,conf,s in [('C1 Primus (Stride-120)','C1 Stride-120',next(x for x in qc if x['config']=='Gauss + Stride-120' and x['scope']=='ALL')),
                     ('MitoNet-v1 (3D+2D)','MitoNet-v1',next(x for x in mp if x['mode']=='3D+2D' and x['split']=='mb1' and x['group']=='ALL'))]:
        out={x:float(s[x]) for x in S}
        for r in (3,5):
            row=next(x for x in cavity0 if x['config']==conf and int(x['radius'])==r)
            out.update({f'{x}_r{r}':float(row[x]) for x in ('interior_fn_ratio','large_cavity_grid_align','closed_share','missed_instance_rate')})
        old[k]=out
    ii=readcsv(OLD/'metrics_40000.csv')
    raw=defaultdict(Counter)
    for p in (OLD/'scores_40000').glob('*.json'):
        d=read(p)
        if d['split'] not in ('mb1','cem_validation'):continue
        for r in d['rows']:
            for met in ('cc','head'):
                if f'{met}_touch_pairs' in r:
                    raw[d['split'],met].update(touching_pairs=r[f'{met}_touch_pairs'],separated_pairs=r[f'{met}_separated'])
    result=[]
    for split in ('mb1','cem_validation'):
        row=next(r for r in ii if r['split']==split and r['group']=='ALL')
        for tag,method in [('Old Primus CC','cc'),('Old center-offset head','head')]:
            result.append(dict(arm=tag,window='legacy128',split=split,group='ALL',method=method,
             **{k:float(row[f'{method}_{x}']) for k,x in zip(I,('AP_50','AP_75','PQ','count_mae'))},**raw[split,method]))
    return old,result

def f(v,percent=False):
    return 'NA' if v is None else (f'{v:.2%}' if percent else f'{v:.5f}')

def main():
    sem=[];inst=[];hq=[]
    for a in ARMS:
        for w in RUNS:
            x,y,z=aggregate(a,w)
            sem+=x;inst+=y;hq+=z
    old,oi=old_stats()
    inst+=oi
    packed=[]
    for a in ARMS:
        for w in RUNS:
            z=[r for r in hq if r['arm']==a and r['window']==w]
            assert len(z)==20
            packed.append(dict(arm=a,window=w,n=len(z),hole_ratio=mean(z,'hole_ratio'),square_holes=sum(r['square_holes'] for r in z)))
    old_group=readcsv(DIAG/'gt_interior_holes/summary_by_group.csv')
    old_base=[r for r in old_group if r['radius']=='3']
    eq=lambda a,s,g,w:next(r for r in sem if r['arm']==a and r['split']==s and r['group']==g and r['window']==w)
    write('semantic_and_cavity.csv',sem)
    write('instance_metrics.csv',inst)
    write('hq_per_image.csv',hq)
    write('hq_summary.csv',packed)
    refs={}
    for a in ARMS:
        for w in RUNS:refs[a+'_'+w]=eq(a,'mb1','ALL',w)
    write('core_mb1_comparison.csv',[dict(system=k,**r) for k,r in (old|refs).items()])
    report=['# Frozen C1 100k · MultiScale-Sem 与 MultiScale-BCD','',
     '两组使用同一 CEM-MitoLab 裁块序列、seed 20261014、batch 4、40k 步和匹配 AdamW 日程。B 的边界与距离损失权重均为 1，距离背景权重为 0.05。B 的分水岭参数依据预先固定的 CEM 验证样本与 collection 等权 PQ 锁定，MB-1 与 MV3 仅应用锁定值。','',
     '| ARM | GPU | 100 步前 20/末 20 损失中位数 | 40k 秒 | 峰值分配 GiB | 峰值保留 GiB | 编码器不变 |',
     '|---|---:|---|---:|---:|---:|---:|']
    for a in ARMS:
        t=read(O/a/'trained.json');s=read(O/a/'smoke_pass.json')
        report.append(f'| {a} | {t["gpu"]} | {s["first20_median"]:.4f} → {s["last20_median"]:.4f} | {t["seconds"]:.1f} | {t["peak_allocated"]/1024**3:.2f} | {t["peak_reserved"]/1024**3:.2f} | {t["encoder_unchanged"]} |')
    w=read(O/ARMS[1]/'watershed_locked.json')
    report+=['',f'BCD 五张训练目标的索引见冒烟回执。锁定参数 (tau_seed, tau_bnd, min_size) = {w["parameters"]}，CEM 验证搜索使用 {len(w["examples"])} 个视野。实例分割在核心掩膜中执行，语义评分使用独立的 fg > 0.5。','',
     '## MB-1 七组等权','',
     '| 系统 | Dice | IoU | r3 内部漏分率 | r5 封闭孔占比 | r3 16px 网格对齐 | 整颗漏检 |',
     '|---|---:|---:|---:|---:|---:|---:|']
    order=(('C1 Primus (Stride-120)',old['C1 Primus (Stride-120)']),
           ('方案 A · Stride-120',refs[ARMS[0]+'_stride120']),('方案 B · Stride-120',refs[ARMS[1]+'_stride120']),
           ('MitoNet-v1 (3D+2D)',old['MitoNet-v1 (3D+2D)']),
           ('方案 A · Baseline-128',refs[ARMS[0]+'_baseline128']),('方案 B · Baseline-128',refs[ARMS[1]+'_baseline128']))
    for name,r in order:
        keys=('dice','iou','interior_fn_ratio_r3','closed_share_r5','large_cavity_grid_align_r3','missed_instance_rate_r3')
        report.append('| '+' | '.join([name]+[f(r[k],j>1) for j,k in enumerate(keys)])+' |')
    report+=['','## MB-1 逐组 r3 内部漏分率','',
             '| 来源 | 旧 Primus | A | B | MitoNet 3D+2D |','|---|---:|---:|---:|---:|']
    for g in sorted({r['group'] for r in sem if r['split']=='mb1' and r['group']!='ALL'}):
        oldv=lambda c:float(next(r for r in old_base if r['group']==g and r['config']==c)['interior_fn_ratio'])
        report.append('| '+' | '.join([g]+[f(x,True) for x in [oldv('C1 Stride-120'),eq(ARMS[0],'mb1',g,'stride120')['interior_fn_ratio_r3'],eq(ARMS[1],'mb1',g,'stride120')['interior_fn_ratio_r3'],oldv('MitoNet-v1')]])+' |')
    report+=['','## CEM 与 MB-1 2D 实例指标','',
             '| 数据 | 模型 | 后处理 | 推理 | AP50 | AP75 | PQ | MAE | 接触对分开/总计 |',
             '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in inst:
        if r['group']!='ALL' or r['split'] not in ('cem_validation','mb1'):continue
        report.append('| '+' | '.join([r[k] for k in ('split','arm','method','window')]+[f(r[k]) for k in I]+[f"{r['separated_pairs']}/{r['touching_pairs']}"])+' |')
    report+=['','旧 Primus 与旧中心偏移头采用历史 256/128 推理。A、B 提供 128 与 120 两种窗口结果，CC 对照直接作用于相同语义掩膜。CEM 验证曾用于 B 的分水岭选择。MB-1 已参与此前开发诊断。','',
             '## HQ-5nm 无标注 20 FOV','',
             '| 模型 | 推理 | 封闭空洞面积占比 | 方孔数 |','|---|---|---:|---:|']
    for r in packed:report.append(f'| {r["arm"]} | {r["window"]} | {f(r["hole_ratio"],True)} | {r["square_holes"]} |')
    report+=['','[逐组语义/内部漏洞](semantic_and_cavity.csv) · [实例对照](instance_metrics.csv) · [HQ 逐图](hq_per_image.csv)。MB-1 有 GT 的复核图在 visual_panels，HQ 输出为预测掩膜。']
    report+=['','## 本轮结果裁决','']
    old0=old['C1 Primus (Stride-120)']
    for a,label in zip(ARMS,('A','B')):
        r=refs[a+'_stride120']
        report.append(f'- {label} 相对旧 Primus 的 MB-1 Dice 变化 {r["dice"]-old0["dice"]:+.5f}，r3 去边界内部漏分率变化 {(r["interior_fn_ratio_r3"]-old0["interior_fn_ratio_r3"])*100:+.2f} 个百分点。')
    for split in ('cem_validation','mb1'):
        r=[x for x in inst if x['group']=='ALL' and x['window']=='stride120' and x['arm']==ARMS[1] and x['split']==split]
        z={x['method']:x for x in r}
        report.append(f'- B 在 {split} 的分水岭 PQ {z["watershed"]["PQ"]:.5f}，同一语义前景的 CC PQ {z["cc"]["PQ"]:.5f}。')
    report+=['','HQ 的方孔形态统计无 GT；MB-1 的内部漏分和整颗漏检仍需分开判读。试验仅包含一个 decoder seed。']
    (O/'EXPERIMENT_SUMMARY.md').write_text('\n'.join(report)+'\n',encoding='utf-8')

if __name__=='__main__':
    main()
