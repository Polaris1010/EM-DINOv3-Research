"""Plot the fixed 45k/100k endpoints from the completed seven-system summary."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'experiments/Pilot-C100'

def main():
    models=json.loads((DATA/'PilotC100_summary.json').read_text(encoding='utf-8'))['models']
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    fields=[('muscle_Primus_Dice','Glycolytic muscle: Dice',False),
            ('muscle_Primus_AP_mean_per_plane','Glycolytic muscle: AP (mean per plane)',False),
            ('muscle_Primus_Recall','Glycolytic muscle: Recall',False),
            ('muscle_Primus_far_FP_rate','Glycolytic muscle: distant-background FP (%)',True)]
    for ax,(key,title,percent) in zip(axes.flat,fields):
        factor=100 if percent else 1
        for prefix,label,color in [('C0_Vanilla_','Vanilla','#2864B4'),('C1_Barrier_','Barrier','#D55E00')]:
            ys=[models[prefix+str(step)+'k'][key]*factor for step in [45,100]]
            ax.plot([45,100],ys,'o-',label=label,color=color,lw=2,markersize=6)
            for step,y in zip([45,100],ys):
                ax.annotate(f'{y:.3f}',(step,y),xytext=(5,5),textcoords='offset points',fontsize=9,color=color)
        ax.axhline(models['Trajectory-Lerp05'][key]*factor,color='#17875D',linestyle='--',lw=1,label='Fixed midpoint')
        ax.set(title=title,xlabel='SSL global step (thousands)',xticks=[45,100],xlim=(40,110))
        ax.grid(axis='y',alpha=.22);ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle('Fixed paired continuation: frozen Primus, one matched decoder seed',fontsize=13)
    out=DATA/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'PilotC100_muscle_endpoints.svg')
    plt.close(fig)

if __name__=='__main__':main()
