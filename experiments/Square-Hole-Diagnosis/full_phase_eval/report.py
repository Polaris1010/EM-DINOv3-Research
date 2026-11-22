from pathlib import Path
import csv,json,statistics as st,sys
p=Path(__file__).resolve().parent
mbref=p.parent.parent/'mix_mitolab_mv_20261118/zeroshot_repeat_20261120/Barrier100_20261014.json'
old=json.loads(mbref.read_text())['rows']
old={(r['name'],int(r['z'])):r for r in old if r['split']=='mb1'}
z=list(csv.DictReader((p.parent/'phase_shift_test/per_image.csv').open()))
oldz={r['sample']:float(r['dice']) for r in z if r['model']=='Barrier100' and r['config']=='Baseline' and r['domain']=='MB1'}
deltas=[(r['sample'],oldz[r['sample']],float(r['dice'])) for r in z if r['model']=='Barrier100' and r['config']=='Shift-8' and r['domain']=='MB1']
configs=('Baseline','Gauss + Stride-120','Gauss + Dual-Phase','Gauss + Quad-Phase')
metrics=('dice','iou','precision','recall')
hq=[r for f in (p/'units').glob('HQ_*.json') for r in json.loads(f.read_text())]
mb=[]
for f in (p/'units').glob('MB1_*'):
    for line in f.read_text().splitlines():
        try:row=json.loads(line)
        except json.JSONDecodeError:
            if '--partial' in sys.argv:continue
            raise
        mb.extend(row)
if not hq:exit()
complete=(p/'receipt.json').exists()
protocol=json.loads((p/'protocol.json').read_text())
receipt=json.loads((p/'receipt.json').read_text()) if complete else {}
mean=lambda a,k:st.mean(v for r in a if (v:=r[k]) is not None) if any(r[k] is not None for r in a) else None
def write(name,rows):
    if not rows:return
    f=p/name
    t=f.with_suffix('.tmp')
    with t.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    t.replace(f)
def agg(a,ref):
    w=sum(r['windows'] for r in a)
    bw=sum(r['windows'] for r in ref)
    return dict(n=len(a),hole_ratio_mean=mean(a,'hole_ratio'),square_holes_total=sum(r['squares'] for r in a),
                square_holes_per_image=mean(a,'squares'),enclosed_holes_total=sum(r['holes'] for r in a),
                windows=w,baseline_windows=bw,window_ratio=w/bw,seconds=sum(r['seconds'] for r in a))
h=[]
for c in configs:
    a=[r for r in hq if r['config']==c]
    b=[r for r in hq if r['config']=='Baseline']
    if a:h.append(dict(config=c,**agg(a,b)))
write('hq5nm_summary.csv',h)
mbby=[]; mball=[]
for c in configs:
    allrows=[r for r in mb if r['config']==c]
    if not allrows:continue
    key=lambda r:(r['id'],r['z'])
    keys={key(r) for r in allrows}
    assert len(keys)==len(allrows)
    ref=[r for r in mb if r['config']=='Baseline' and key(r) in keys]
    by=[]
    for g in sorted({r['group'] for r in allrows}):
        a=[r for r in allrows if r['group']==g]
        b=[r for r in ref if r['group']==g]
        by.append(dict(config=c,group=g,**agg(a,b),**{k:mean(a,k) for k in metrics},**{'delta_'+k:mean(a,k)-mean(b,k) if mean(a,k) is not None and mean(b,k) is not None else None for k in metrics}))
    mbby.extend(by)
    n=len(allrows)
    scope='ALL' if n==2595 and len(by)==7 else 'TEM-100+glycolytic_muscle' if n==402 and len(by)==2 else 'PARTIAL'
    d=agg(allrows,ref)
    d['hole_ratio_mean']=mean(by,'hole_ratio_mean')
    mball.append(dict(config=c,scope=scope,groups=len(by),**d,**{k:mean(by,k) for k in metrics},**{'delta_'+k:mean(by,'delta_'+k) for k in metrics}))
write('mb1_by_group.csv',mbby)
write('mb1_summary.csv',mball)
if complete:
    assert len(h)==4 and all(r['n']==20 for r in h)
    assert len(mball)==4 and [r['n'] for r in mball[:3]]==[2595]*3
    assert mball[3]['n']==(2595 if protocol['quad_full'] else 402)
    write('per_image.csv',hq+mb)
base=[r for r in mb if r['config']=='Baseline']
diffs=[r for r in base if r['baseline_replay_diff']]
assert not any(not r['reflect_small_image'] for r in diffs)
mref=[dict(group=r['group'],**{k:old[r['id'],r['z']][k] for k in metrics}) for r in base]
gb=sorted({r['group'] for r in base})
macro=lambda a,k:st.mean(x for g in gb if (x:=mean([r for r in a if r['group']==g],k)) is not None)
out=['# Full Patch-Phase-Decoupled Inference', '',
 f'状态 **{"完整" if complete else "在运行"}**。物理 GPU {protocol["gpu"]}。编码器 `{protocol["encoder_path"]}`，解码器 `{protocol["decoder_path"]}`。SHA256 见 [protocol.json](protocol.json)。',
 f'推理窗口 256、阈值 >0.5、reflect 补边。原图边缘从每相位的最少覆盖窗口取预测。每个视野先计算 Baseline 并核验，再依次计算三个候选；同相位前向结果复用。表中秒数是各配置所用 pass 之和，可相互重叠，包含 tile 组装与融合。拓扑统计另外计算。全程墙钟耗时 {receipt.get("seconds_wall",0):.1f} 秒（未完成时为 0）。',
 '', '## 步骤 0 · 前次 Shift-8 掉点', '',
 '旧脚本使用 127.5 灰度常数补边，归一化后为 0。Reflect 是本次统一的新补边规则。掉点由四张图共同造成，Fly brain 是最大项。现有对照不能将旧掉点单独归因于补边。',
 '', '| 视野 | 基线 Dice | Shift-8 Dice | 差值 |', '|---|---:|---:|---:|']
for k,a,b in deltas:out.append(f'| {k} | {a:.5f} | {b:.5f} | {b-a:+.5f} |')
out+=['','## HQ-5nm · 共同 20 FOV','',
 '| 配置 | FOV | 空洞面积占比 | 方孔总数 / 每张 | 封闭孔总数 | 窗口数 / 倍率 | 秒 |',
 '|---|---:|---:|---:|---:|---:|---:|']
for r in h:
    out.append(f'| {r["config"]} | {r["n"]} | {r["hole_ratio_mean"]:.3%} | {r["square_holes_total"]} / {r["square_holes_per_image"]:.2f} | {r["enclosed_holes_total"]} | {r["windows"]} / {r["window_ratio"]:.3f}× | {r["seconds"]:.1f} |')
def val(v):
    return f'{v:.5f}' if v is not None else 'NA'
out+=['','## MB-1 汇总','',
 f'Quad-Phase 全量估算 {protocol.get("quad_full_estimated_seconds",0):.0f} 秒。超过 900 秒时四相位仅应用于 TEM-100 和 Glycolytic muscle，倍率用同子集 Baseline 作分母。',
 '',f'Baseline 已核对 {len(base)} planes，其间 {len(diffs)} 个具有 reflect 补边的样本与旧四项混淆计数不同。无需补边的样本要求严格一致。',
 '', '| 配置 | 范围 | N | Dice | IoU | Precision | Recall | 空洞占比 | 方孔 | 窗口数 / 倍率 | 秒 |',
 '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
if base:
    out.extend([f'相同已完成视野按组等权的历史 Baseline Dice 为 {macro(mref,"dice"):.5f}，本次为 {macro(base,"dice"):.5f}；历史 IoU 为 {macro(mref,"iou"):.5f}，本次为 {macro(base,"iou"):.5f}。',''])
for r in mball:
    out.append(f'| {r["config"]} | {r["scope"]} | {r["n"]} | '+ ' | '.join(val(r[k]) for k in metrics)+f' | {r["hole_ratio_mean"]:.3%} | {r["square_holes_total"]} | {r["windows"]} / {r["window_ratio"]:.3f}× | {r["seconds"]:.1f} |')
out+=['','### MB-1 逐组','', '| 子组 | 配置 | N | Dice | IoU | Precision | Recall | 方孔 |', '|---|---|---:|---:|---:|---:|---:|---:|']
for r in sorted(mbby,key=lambda r:(r['group'],configs.index(r['config']))):
    out.append(f'| {r["group"]} | {r["config"]} | {r["n"]} | '+' | '.join(val(r[k]) for k in metrics)+f' | {r["square_holes_total"]} |')
out+=['','MB-1 各组先对切片或图像算均值，再对数据组等权。拓扑指标描述预测形状，Dice/IoU 使用人工真值。','',
 '[HQ CSV](hq5nm_summary.csv) · [MB-1 汇总 CSV](mb1_summary.csv) · [MB-1 逐组 CSV](mb1_by_group.csv) · [逐图 CSV](per_image.csv)']
(p/'FULL_PHASE_EVAL_SUMMARY.md').write_text('\n'.join(out)+'\n',encoding='utf-8')
