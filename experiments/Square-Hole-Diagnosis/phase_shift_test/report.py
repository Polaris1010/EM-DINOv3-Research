from pathlib import Path
import csv,json,statistics as s
p=Path(__file__).resolve().parent
rows=list(csv.DictReader((p/'per_image.csv').open()))
receipt=json.loads((p/'receipt.json').read_text())
models=('P0','Barrier100')
configs=('Baseline','Shift-8','Gauss-Center','Gauss+Dual-Phase')
def subset(m,c,d):
    a=[r for r in rows if all(r[k]==v for k,v in [('model',m),('config',c),('domain',d)])]
    assert len(a)==(6 if d=='HQ' else 4)
    return a
def avg(a,k):return s.mean(float(r[k]) for r in a)
def total(a,k):return sum(int(r[k]) for r in a)
def frac(a,n,d):
    x,y=total(a,n),total(a,d)
    return f'{x}/{y} ({x/y:.2%})' if y else '0/0'
out=['# Patch 相位偏移及无训练融合对照','',
 f'物理 GPU {receipt["gpu"]}，推理及评分 {receipt["seconds"]:.2f} 秒。6 张 HQ-5nm 原图覆盖两模型，每张原始基线掩膜逐像素复现。MB-1 使用原诊断的 4 张有 GT 图（三张 TEM-100 加一张 Fly brain），以全二维视野逐图计算 Dice 和 IoU，再等权平均。四图是孔洞导向样本。',
 '窗口 256，步幅 128，原生输入网格。移相窗口以原图坐标 8 (mod 128) 排列，超出边界用原训练流程的灰度常数 127.5 填充。高斯 sigma 64，每相位内部按权重归一化，双相位两张概率图各占一半。所有方案同用阈值 >0.5。',
 '', '| 模型 | 配置 | HQ 空洞面积占比 | HQ 方孔总数 | 每张方孔 | MB-1 Dice | Δ Dice | MB-1 IoU | Δ IoU |',
 '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for m in models:
    g0=subset(m,'Baseline','MB1')
    for c in configs:
        a=subset(m,c,'HQ')
        b=subset(m,c,'MB1')
        d,i=avg(b,'dice'),avg(b,'iou')
        out.append(f'| {m} | {c} | {avg(a,"hole_ratio"):.3%} | {total(a,"squares")} | {avg(a,"squares"):.2f} | {d:.5f} | {d-avg(g0,"dice"):+.5f} | {i:.5f} | {i-avg(g0,"iou"):+.5f} |')
out+=['','## Shift-8 后的孔边界坐标','',
      '新格线采用 (coord−8) % 16 ≤1，旧格线采用 coord % 16 ≤1。这里按指令用单侧判据，与上一轮两侧各容差 1 像素的百分比口径不同。新孔以和任一 Baseline 类正方形孔 IoU <0.5 为操作性定义。',
      '', '| 模型 | 全部 Shift-8 孔边界 · 新 / 旧 | 新孔边界 · 新 / 旧 | 双相位残留方孔中同时与 Baseline、Shift-8 方孔 IoU ≥0.5 |',
      '|---|---|---|---|']
for m in models:
    a=subset(m,'Shift-8','HQ')
    dual=subset(m,'Gauss+Dual-Phase','HQ')
    out.append(f'| {m} | {frac(a,"shift_edges","edges")} / {frac(a,"old_edges","edges")} | {frac(a,"new_hole_shift_edges","new_hole_edges")} / {frac(a,"new_hole_old_edges","new_hole_edges")} | {total(dual,"persistent_squares")}/{total(dual,"squares")} |')
out+=['','## 双相位的实际变化','']
for m in models:
    b=subset(m,'Baseline','HQ')
    d=subset(m,'Gauss+Dual-Phase','HQ')
    old,new=total(b,'squares'),total(d,'squares')
    mb0={r['sample']:float(r['dice']) for r in subset(m,'Baseline','MB1')}
    mb={r['sample']:float(r['dice']) for r in subset(m,'Gauss+Dual-Phase','MB1')}
    out.append(f'- {m} 的 HQ 方孔由 {old} 变为 {new}（{(new-old)/old:+.1%}）。MB-1 逐图 Dice 差值为 '+', '.join(f'{key} {mb[key]-mb0[key]:+.5f}' for key in mb)+'。')
out+=['','HQ-5nm 未提供 GT。类正方形数量下降可伴随孔形状变化，并不能单凭计数判断语义修复。四张 MB-1 只用于筛查，后续稳定性以全量 MB-1 和其他独立有 GT 的数据为准。孔与孔 IoU 匹配属于描述性定位，无法单凭持久性归因于某个特征层。','',
      '[逐图指标](per_image.csv) · [逐孔坐标、两相位对齐与重合](per_hole.csv)']
(p/'PHASE_SHIFT_SUMMARY.md').write_text('\n'.join(out)+'\n',encoding='utf-8')
print('\n'.join(out))
