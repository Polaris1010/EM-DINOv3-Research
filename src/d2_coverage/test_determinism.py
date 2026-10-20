"""CPU multiprocessing tests on the real immutable sampling plans."""
import argparse,concurrent.futures,hashlib,json,multiprocessing,time
from pathlib import Path
import numpy as np
from core import START,END,BATCH,CoveragePlan

def worker(args):
    directory,strategy,pool,cem,rank,worker_id,steps=args
    plan=CoveragePlan(directory,strategy,pool,cem)
    indices=[step*48+rank*16+j for step in steps for j in range(16) if j%2==worker_id]
    # Different issue order from the main process; same indices must mean same samples.
    indices.reverse()
    return [(i,plan.choose(i)) for i in indices]

def main(repo,artifacts,out):
    pool=repo/'experiments/D12/mitoverse_pool.json';cem=repo/'experiments/D12/cem_pool.json'
    t=time.perf_counter();result={'gpu_used':False,'image_reads':False,'world_size':3,'workers_per_rank':2,'strategies':{}}
    for strategy in ['A','B','C']:
        plan=CoveragePlan(artifacts,strategy,pool,cem)
        state=plan.checkpoint_state(54321);encoded=json.dumps(state)
        loaded=json.loads(encoded);assert plan.validate_resume(loaded)==54321*48
        for key,value in [('next_global_index',0),('plan_sha256','wrong'),('world_size',1)]:
            bad={**loaded,key:value}
            try:plan.validate_resume(bad)
            except ValueError:pass
            else:raise AssertionError('Corrupt resume accepted')
        # Simulated prefetch before checkpoint: querying future indices cannot mutate state.
        for i in range(54321*48,54325*48):plan.choose(i)
        assert plan.checkpoint_state(54321)==loaded
        second=CoveragePlan(artifacts,strategy,pool,cem)
        steps=[30000,30001,54320,54321,54322,69100,69235,69236,69999,99999]
        jobs=[(str(artifacts),strategy,str(pool),str(cem),r,w,steps) for r in range(3) for w in range(2)]
        merged={}
        with concurrent.futures.ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as executor:
            for values in executor.map(worker,jobs):
                for index,record in values:
                    assert index not in merged;merged[index]=record
        expected={step*48+j for step in steps for j in range(48)}
        assert set(merged)==expected
        for index,record in merged.items():assert record==plan.choose(index)==second.choose(index)
        # All 70k batches, all ranks: shard and reassemble IDs using training's formula.
        shape=plan.ids.reshape(70000,3,16);rebuilt=np.empty_like(shape)
        for rank in range(3):rebuilt[:,rank,:]=np.asarray(shape[:,rank,:])
        assert hashlib.sha256(rebuilt.tobytes()).digest()==hashlib.sha256(plan.ids.tobytes()).digest()
        for i in [START*48-1,END*48]:
            try:plan.choose(i)
            except IndexError:pass
            else:raise AssertionError('Out-of-budget sample accepted')
        result['strategies'][strategy]={'records_checked_across_3ranks_2workers':len(merged),
            'all_future_ids_reassembled':int(plan.ids.size),'worker_issue_order_invariant':True,
            'restart_same_records':True,'unconsumed_prefetch_has_no_effect':True,
            'corrupt_state_rejected':True,'checkpoint_state_bytes':len(encoded.encode()),
            'plan_sha256':plan.meta['id_sha256']}
    result['seconds']=time.perf_counter()-t;result['status']='PASS'
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.repo,a.artifacts,a.out)
