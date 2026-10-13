from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
BASE=Path(__file__).resolve().parent
from summarize import RUNS,DATASETS
m=json.loads((BASE/'comparison.json').read_text())
labels={'em':'EM initial','gram200k':'Gram 200k (old LR)','b2_w0':'b2 / lambda 0','b2_w1':'b2 / lambda 1','b2_w2':'b2 / lambda 2','b4_w0':'b4 / lambda 0','b4_w2':'b4 / lambda 2','b2_w0_lr1':'b2 / lambda 0 / old LR','b2_w2_lr1':'b2 / lambda 2 / old LR'}
colors={'em':'#456990','gram200k':'#9f79a8','b2_w0':'#777777','b2_w1':'#e2a03c','b2_w2':'#3d9c81','b4_w0':'#aaaaaa','b4_w2':'#246c59','b2_w0_lr1':'#a2b7bd','b2_w2_lr1':'#63a5b3'}
keys=[k for k in RUNS if k in m['models']]
fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True,constrained_layout=True)
for ax,ds in zip(axes,DATASETS):
 vals=[m['models'][k]['datasets'][ds]['macro_dice'] for k in keys];y=np.arange(len(keys))
 ax.barh(y,vals,color=[colors[k] for k in keys]);ax.set_yticks(y,[labels[k] for k in keys]);ax.set_xlim(0,1);ax.set_ylim(len(keys)-.5,-.5);ax.set_title(ds);ax.set_xlabel('Full-volume macro Dice');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
 for i,v in enumerate(vals):ax.text(v+.008,i,f'{v:.4f}',va='center',fontsize=9)
fig.suptitle('Matched frozen-backbone probes: same decoder seed and 3,000 supervised updates\nSingle seed; validation-set selection; batch-4 SSL sees twice as many images',fontsize=11)
fig.savefig(BASE/'probe_comparison.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
for ax,ds in zip(axes,DATASETS):
 for k in keys:
  f=BASE/'jobs'/RUNS[k][0]/ds/'curve.json'
  if f.exists():
   curve=json.loads(f.read_text());ax.plot([r['step'] for r in curve],[r['macro_dice'] for r in curve],marker='o',label=labels[k],color=colors[k],linewidth=1.5)
 ax.set_title(ds);ax.set_xlabel('Decoder updates');ax.set_ylabel('Stride-20 validation Dice (no TTA)');ax.grid(alpha=.2);ax.set_xticks([1000,2000,3000]);ax.legend(fontsize=8)
fig.suptitle('Head selection curves: these are subsampled validation scores, not the final full-volume scores',fontsize=11)
fig.savefig(BASE/'probe_learning_curves.png',dpi=150);plt.close(fig)
print('saved comparison and learning curve plots')
