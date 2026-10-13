"""Render the measured frozen comparison as a standalone table image."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT=Path(__file__).resolve().parent/'mitoverse_boundary_metrics_20261012'
d=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
bold=FontProperties(fname='C:/Windows/Fonts/msyhbd.ttc')
fig,ax=plt.subplots(figsize=(13.7,5.2),dpi=200)
fig.patch.set_facecolor('white');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
ax.text(.02,.94,'冻结评测：B_full 与官方 DINOv3 自然图像权重',fontproperties=bold,fontsize=16,va='top')
ax.text(.02,.84,'完整验证体积 · 同一 decoder 协议 · 单随机种子 · 均值 ± 体积间样本标准差',fontproperties=font,fontsize=10,color='#444444')
cols=[.075,.33,.575,.745,.915]
heads=['数据集（n）','冻结编码器','Dice ↑','HD95 ↓','NSD ↑（τ=1）']
ax.plot([.02,.98],[.77,.77],color='black',lw=1.5)
for x,label in zip(cols,heads):ax.text(x,.715,label,ha='center',va='center',fontproperties=font,fontsize=11)
ax.plot([.02,.98],[.67,.67],color='black',lw=.6)
for i,(ds,kind) in enumerate([(ds,kind) for ds in ['Dataset131_CMCT','Dataset111_CMCC'] for kind in ['B_full','official_natural']]):
    row=next(r for r in d['summary'] if r['dataset']==ds and r['encoder_kind']==kind)
    y=.61-i*.105
    vals=[f"{ds.split('_')[-1]}（{row['n']}）",'B_full（EM 100k）' if kind=='B_full' else '官方自然图像权重']
    for m in ['dice','hd95','nsd']:
        places=2 if m=='hd95' else 4
        vals.append(f"{row[m+'_mean']:.{places}f} ± {row[m+'_std']:.{places}f}")
    other=next(r for r in d['summary'] if r['dataset']==ds and r['encoder_kind']!=kind)
    for j,(x,label) in enumerate(zip(cols,vals)):
        prop=font
        if j>=2:
            m=['dice','hd95','nsd'][j-2]
            better=row[m+'_mean']<other[m+'_mean'] if m=='hd95' else row[m+'_mean']>other[m+'_mean']
            if better:prop=bold
        ax.text(x,y,label,ha='center',va='center',fontproperties=prop,fontsize=10.5)
    if i==1:ax.plot([.02,.98],[y-.052,y-.052],color='black',lw=.6)
ax.plot([.02,.98],[.235,.235],color='black',lw=1.1)
notes=['std：病例间样本标准差（ddof=1），不是重复训练 std；CMCT n=2，CMCC n=3。',
       'HD95：voxel 等效单位；文件 spacing=1×1×1。NSD：沿用官方阈值 1.0（等效 1 voxel）。',
       'MedDINOv3 默认测试入口 / MONAI 1.5.0；真实 EM 物理尺度尚未校准；本表为验证集结果。']
for j,n in enumerate(notes):ax.text(.02,.18-j*.055,n,fontproperties=font,fontsize=9,color='#444444')
fig.savefig(OUT/'comparison_table.png',bbox_inches='tight',pad_inches=.18)
fig.savefig(OUT/'comparison_table.svg',bbox_inches='tight',pad_inches=.18)
plt.close(fig)
print(OUT/'comparison_table.png')
