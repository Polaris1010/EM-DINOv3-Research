"""Actual-index CPU simulation. No labels, images, training or GPU access."""
import argparse,collections,csv,importlib.util,json,platform,sys,time
from pathlib import Path
import numpy as np
from core import START,END,BATCH,HistoricalSampler,CoveragePlan,compile_plan,save_plan,sha

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main(repo,artifacts,out):
    out.mkdir(parents=True,exist_ok=True);artifacts.mkdir(parents=True,exist_ok=True)
    poolp=repo/'experiments/D12/mitoverse_pool.json';cemp=repo/'experiments/D12/cem_pool.json'
    pool=json.loads(poolp.read_text());cem=json.loads(cemp.read_text());sampler=HistoricalSampler(pool,cem)
    spec=importlib.util.spec_from_file_location('old',repo/'src/d12/sampling.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    checks=sorted(set(range(100))|{int(v) for v in np.linspace(0,END*BATCH-1,401)})
    for i in checks:assert sampler.choose(i)==old.choose(pool,cem,.5,0,i)
    t=time.perf_counter();ids=np.full(END*BATCH,-1,np.int32);counts=collections.Counter()
    for i in range(END*BATCH):
        loc=sampler.choose(i);counts[loc['domain']+'::'+loc['source']]+=1
        if loc['domain']=='CEM1.5M':ids[i]=loc['image_id']
        if i+1 in [START*BATCH,END*BATCH]:
            stage,prefix=('D12','D12') if i+1==START*BATCH else ('D12-100k','D12_100k')
            expected=json.loads((repo/f'experiments/{stage}/{prefix}_sampling_observed.json').read_text())['D2']
            assert dict(counts)==expected
            print('historical replay',i+1,'source counts exact',flush=True)
    historical_seconds=time.perf_counter()-t
    np.save(artifacts/'historical_ids.npy',ids,allow_pickle=False)
    prefix_counts=np.bincount(ids[:START*BATCH][ids[:START*BATCH]>=0],minlength=cem['images'])
    assert int((prefix_counts>0).sum())==485179
    owner=np.empty(cem['images'],np.int32)
    n=np.array([g['images'] for g in cem['groups']])
    for i,g in enumerate(cem['groups']):owner[g['first_id']:g['last_id']+1]=i
    prior_group=np.bincount(owner[ids[:START*BATCH][ids[:START*BATCH]>=0]],minlength=len(n))
    future=ids[START*BATCH:];slot=np.flatnonzero(future>=0)
    a_group=np.bincount(owner[future[slot]],minlength=len(n))
    visited30=np.bincount(owner[prefix_counts>0],minlength=len(n))
    missing=n-visited30
    p=np.array([g['probability'] for g in cem['groups']])
    summary={'status':'simulation_only','gpu_used':False,'image_reads':False,'training_started':False,
             'historical_replay_seconds':historical_seconds,'old_sampler_full_record_equivalence_checks':len(checks),
             'prior_CEM_draws':int(prefix_counts.sum()),'future_CEM_slots':int(len(slot)),
             'future_MV_slots':int(len(future)-len(slot)),'missing_at30k':int(missing.sum()),
             'A_fixed_collection_budget_deficit':int(np.maximum(missing-a_group,0).sum()),
             'minimum_required_TV_from_A_future_counts':float(np.maximum(missing-a_group,0).sum()/len(slot)),
             'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,'runs':{}}
    collection_rows=[{'collection':g['name'],'images':g['images'],'p_sqrt':float(p[i]),
                     'visited30k':int(visited30[i]),'missing30k':int(missing[i]),'draws30k':int(prior_group[i])} for i,g in enumerate(cem['groups'])]
    curves=[];repeats=[]
    for strategy in ['A','B','C']:
        started=time.perf_counter();compiled=compile_plan(strategy,ids,cem)
        compile_seconds=time.perf_counter()-started
        started=time.perf_counter();meta=save_plan(artifacts,strategy,compiled,poolp,cemp);save_seconds=time.perf_counter()-started
        plan=CoveragePlan(artifacts,strategy,poolp,cemp)
        assert np.array_equal(compiled<0,future<0)
        # Actual simulation goes through the same random-access core as a future loader.
        started=time.perf_counter()
        consumed=np.fromiter((plan.image_id(i) for i in range(START*BATCH,END*BATCH)),dtype=np.int32,count=len(future))
        assert np.array_equal(consumed,compiled)
        lookup_seconds=time.perf_counter()-started
        assert np.array_equal(consumed<0,future<0)
        runtime_indices=sorted(set(int(x) for x in np.linspace(START*BATCH,END*BATCH-1,6000)))
        started=time.perf_counter()
        for index in runtime_indices:
            new=plan.choose(index);base=sampler.choose(index)
            assert new['index']==base['index'] and new['domain']==base['domain'] and new['view_seed']==base['view_seed']
            if base['domain']=='MitoVerse':assert new==base
            else:assert new['image_id']==plan.image_id(index)
        choose_seconds=time.perf_counter()-started
        cemids=consumed[consumed>=0];groupcounts=np.bincount(owner[cemids],minlength=len(n))
        imagecounts=prefix_counts+np.bincount(cemids,minlength=cem['images'])
        visited=np.bincount(owner[imagecounts>0],minlength=len(n))
        if strategy=='A':assert int((imagecounts>0).sum())==955756
        if strategy=='B':
            assert np.array_equal(groupcounts,a_group)
            assert np.array_equal(visited,visited30+np.minimum(missing,a_group))
        if strategy=='C':
            assert (imagecounts>0).all()
            assert int(np.abs(groupcounts-a_group).sum())==2*summary['A_fixed_collection_budget_deficit']
        for step in range(START,END+1,5000):
            part=consumed[:(step-START)*BATCH];c=prefix_counts+np.bincount(part[part>=0],minlength=cem['images'])
            curves.append({'strategy':strategy,'optimizer_step':step,'unique_CEM_images':int((c>0).sum()),
                           'coverage':float((c>0).mean()),'CEM_draws':int(c.sum())})
        hist=np.bincount(imagecounts)
        for visits,count in enumerate(hist):
            if count:repeats.append({'strategy':strategy,'image_visits':visits,'number_of_images':int(count)})
        for j,row in enumerate(collection_rows):
            row.update({f'{strategy}_future_draws':int(groupcounts[j]),f'{strategy}_total_draws':int(groupcounts[j]+prior_group[j]),
                        f'{strategy}_unique':int(visited[j]),f'{strategy}_coverage':float(visited[j]/n[j]),
                        f'{strategy}_future_share':float(groupcounts[j]/len(slot)),
                        f'{strategy}_share_minus_sqrt':float(groupcounts[j]/len(slot)-p[j]),
                        f'{strategy}_future_draw_delta_vs_A':int(groupcounts[j]-a_group[j])})
        fullstep=None
        if strategy=='C':fullstep=int((START*BATCH+slot[int(missing.sum())-1])//BATCH+1)
        run={'unique_images':int((imagecounts>0).sum()),'coverage':float((imagecounts>0).mean()),
             'unseen':int((imagecounts==0).sum()),'full_coverage_step':fullstep,
             'full_collections':int((visited==n).sum()),'collection_coverage_quantiles':np.quantile(visited/n,[0,.25,.5,.75,1]).tolist(),
             'TV_vs_A_future_counts':float(np.abs(groupcounts-a_group).sum()/2/len(slot)),
             'TV_vs_sqrt_future_probability':float(np.abs(groupcounts/len(slot)-p).sum()/2),
             'max_future_collection_share':float((groupcounts/len(slot)).max()),
             'compile_seconds':compile_seconds,'plan_save_seconds':save_seconds,
             'plan_bytes':(artifacts/meta['id_file']).stat().st_size,'id_lookup_seconds_all_future':lookup_seconds,
             'choose_and_base_compare_seconds_6000':choose_seconds,'plan_sha256':meta['id_sha256'],
             'repeat_visit_quantiles_all_images':np.quantile(imagecounts,[0,.25,.5,.75,.95,.99,1]).tolist()}
        summary['runs'][strategy]=run
        print(strategy,json.dumps(run),flush=True)
    summary['pool_hashes']={'mv':sha(poolp),'cem':sha(cemp)}
    summary['plan_directory']=str(artifacts)
    write_csv(out/'coverage_curve.csv',curves);write_csv(out/'collection_metrics.csv',collection_rows);write_csv(out/'repeat_histogram.csv',repeats)
    (out/'simulation_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('SIMULATION_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.repo,a.artifacts,a.out)
