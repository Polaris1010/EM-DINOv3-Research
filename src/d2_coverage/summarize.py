"""Summarize CPU simulations and metadata-only I/O estimates; render design figures."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(evidence,artifacts):
    summary=json.loads((evidence/'simulation_summary.json').read_text())
    inspection=json.loads((evidence/'checkpoint_index_inspection.json').read_text())
    with (evidence/'collection_metrics.csv').open() as f:rows=list(csv.DictReader(f))
    with (evidence/'coverage_curve.csv').open() as f:curves=list(csv.DictReader(f))
    with (evidence/'repeat_histogram.csv').open() as f:repeats=list(csv.DictReader(f))
    sizes={r['collection']:r for r in inspection['cem_collections']}
    estimated={}
    for strategy in ['A','B','C']:
        expected=sum(int(r[f'{strategy}_future_draws'])*sizes[r['collection']]['compressed_bytes']/int(r['images']) for r in rows)
        decoded=sum(int(r[f'{strategy}_future_draws'])*sizes[r['collection']]['decoded_bytes']/int(r['images']) for r in rows)
        estimated[strategy]={'expected_future_CEM_compressed_read_bytes':expected,'expected_future_CEM_decoded_bytes':decoded}
    a=estimated['A']['expected_future_CEM_compressed_read_bytes']
    for v in estimated.values():v['compressed_read_ratio_vs_A']=v['expected_future_CEM_compressed_read_bytes']/a
    first_missing_share=max((int(r['missing30k'])/summary['missing_at30k'],r['collection']) for r in rows)
    top=sorted(rows,key=lambda r:int(r['missing30k'])-int(r['A_future_draws']),reverse=True)[:18]
    estimates={'method':'Per-collection mean compressed/decoded bytes from actual read-only SQLite, multiplied by simulated collection counts. Expectations, not measured image-loader throughput.',
        'no_image_reads':True,'runs':estimated,'retained_pool_compressed_bytes':sum(r['compressed_bytes'] for r in sizes.values()),
        'C_first_phase_max_collection_share':first_missing_share,
        'deficit_collections':[{'collection':r['collection'],'images':int(r['images']),'missing30k':int(r['missing30k']),
                              'A_future_draws':int(r['A_future_draws']),'C_future_draws':int(r['C_future_draws'])} for r in top],
        'full_checkpoint_bytes':inspection['checkpoint_bytes'],'backbone_export_bytes':inspection['milestone_files']['teacher_backbone.pth'],
        'SSL_disk_final_bytes_3full_4backbones':3*inspection['checkpoint_bytes']+4*inspection['milestone_files']['teacher_backbone.pth'],
        'SSL_disk_transient_bytes_4full_4backbones':4*inspection['checkpoint_bytes']+4*inspection['milestone_files']['teacher_backbone.pth']}
    (evidence/'cost_estimates.json').write_text(json.dumps(estimates,indent=2)+'\n',encoding='utf-8')
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    colors={'A':'#6b7280','B':'#2378b6','C':'#dc7726'}
    for strategy in ['A','B','C']:
        vals=[r for r in curves if r['strategy']==strategy]
        axs[0,0].plot([int(v['optimizer_step'])/1000 for v in vals],[float(v['coverage'])*100 for v in vals],marker='.',label=strategy,color=colors[strategy])
        cov=np.sort([float(r[f'{strategy}_coverage'])*100 for r in rows])
        axs[0,1].plot(cov,np.arange(1,len(cov)+1)/len(cov),label=strategy,color=colors[strategy])
        rr=[r for r in repeats if r['strategy']==strategy]
        axs[1,0].plot([int(r['image_visits']) for r in rr],[int(r['number_of_images']) for r in rr],'.-',label=strategy,color=colors[strategy])
    axs[0,0].axvline(summary['runs']['C']['full_coverage_step']/1000,color=colors['C'],linestyle='--',alpha=.6)
    axs[0,0].set(xlabel='Optimizer step (thousands)',ylabel='Unique CEM coverage (%)',title='Same 30k prefix; fixed remaining domain slots')
    axs[0,1].set(xlabel='Per-collection image coverage (%)',ylabel='Fraction of 627 collections',title='Final collection coverage distribution')
    axs[1,0].set(xlabel='Visits per image (including 30k prefix)',ylabel='Number of image IDs (log)',yscale='log',title='Repetition distribution')
    selected=sorted(range(len(rows)),key=lambda i:int(rows[i]['images']),reverse=True)[:15]
    xx=np.arange(len(selected))
    for k,strategy in enumerate(['A','B','C']):
        axs[1,1].bar(xx+(k-1)*.25,[float(rows[i][f'{strategy}_future_share'])*100 for i in selected],width=.25,label=strategy,color=colors[strategy])
    axs[1,1].set(xlabel='15 largest collections, size rank',ylabel='Share of 70k CEM slots (%)',title='Full coverage changes collection exposure')
    axs[1,1].set_xticks(xx,range(1,16))
    for ax in axs.flat:ax.legend();ax.grid(alpha=.15)
    fig.suptitle('D2-Coverage CPU simulation | C: full coverage + minimum-TV quota compensation',fontsize=14)
    path=artifacts/'coverage_design.png';fig.savefig(path,dpi=140);plt.close(fig)
    manifest={'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (evidence/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(estimates,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True)
    a=p.parse_args();main(a.evidence,a.artifacts)
