"""Scientific plots from completed CSVs; no thresholds or models are fitted."""
from pathlib import Path
import argparse,csv
import matplotlib
matplotlib.use('Agg');matplotlib.rcParams['svg.fonttype']='none'
import matplotlib.pyplot as plt
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--review-dir',required=True);a=p.parse_args()
    r=Path(a.run_dir);review=Path(a.review_dir);out=r/'figures';out.mkdir(exist_ok=True);review.mkdir(parents=True,exist_ok=True)
    def rows(n):
        with (r/'tables'/n).open(newline='') as f:return list(csv.DictReader(f))
    rank=rows('final_pixel_discrimination.csv');scores=rows('final_source_metrics.csv');curves=rows('PR_curves.csv')
    sources=['openorganelle','guay21','muller21','glycolytic_muscle','hela_cell'];names=['OpenOrganelle','Guay','Mueller','Glycolytic muscle','HeLa'];colors={'D2-30k':'#dc9735','D2-100k':'#bd4346'}
    def save(fig,name):
        fig.savefig(out/(name+'.svg'),bbox_inches='tight');fig.savefig(review/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.6))
    for ax,key in zip(axes,['AUROC','AP']):
        ds=[]
        for s in sources:
            v={x['model']:float(x[key]) for x in rank if x['source']==s and x['scope']=='all_valid'};ds.append(v['D2-100k']-v['D2-30k'])
        ax.bar(range(5),ds,color=['#417b64' if v>=0 else '#bd4346' for v in ds]);ax.axhline(0,color='#777',lw=.7);ax.set_xticks(range(5),names,rotation=20,ha='right');ax.set_title('100k - 30k: pixel '+key);ax.set_ylabel('Change');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();save(fig,'A2_pixel_ranking_changes')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key,title,mult in zip(axes,['dice','recall','far_FP_rate'],['Dice','Recall','Remote background FP (%)'],[1,1,100]):
        for op,marker,color in [('raw','o','#bd4346'),('calibrated','^','#3a7585')]:
            values=[float(next(x[key] for x in scores if x['source']==s and x['model']=='D2-100k' and x['operation']==op))*mult for s in sources]
            ax.plot(range(5),values,marker+'-',label='Raw 0.5' if op=='raw' else 'Train-only bias',color=color)
        ax.set_xticks(range(5),names,rotation=25,ha='right');ax.set_title('D2-100k '+title);ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();save(fig,'A2_Dice_Recall_FP_tradeoff')
    fig,axes=plt.subplots(1,5,figsize=(17,3.7),sharex=True,sharey=True)
    for ax,s,name in zip(axes,sources,names):
        for model in ['D2-30k','D2-100k']:
            c=[x for x in curves if x['source']==s and x['model']==model and x['scope']=='all_valid']
            ax.plot([float(x['recall']) for x in c],[float(x['precision']) for x in c],color=colors[model],lw=1.4,label=model)
            for op,marker in [('raw','o'),('calibrated','^')]:
                x=next(x for x in scores if x['source']==s and x['model']==model and x['operation']==op)
                ax.plot(float(x['recall']),float(x['precision']),marker,color=colors[model],markersize=7,markeredgecolor='white')
        ax.set_title(name);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel('Recall');ax.grid(alpha=.2)
    axes[0].set_ylabel('Precision');axes[0].legend(fontsize=8);fig.suptitle('Same native pixels: circles = raw 0.5; triangles = train-only bias. Curves retain original ranking.',fontsize=11);fig.tight_layout();save(fig,'A2_PR_curves')
if __name__=='__main__':main()
