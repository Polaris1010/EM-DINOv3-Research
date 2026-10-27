"""Reconcile completed C-1 logs; no torch, images, labels or model updates."""
from pathlib import Path
import argparse,json,csv,statistics,math,hashlib

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode())
def csvout(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for x in rows for k in x)))
  writer.writeheader();writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--private-input-dir',type=Path,required=True);p.add_argument('--historical-audit',type=Path,required=True);p.add_argument('--repository',type=Path,required=True);a=p.parse_args()
 d=a.private_input_dir;w=a.repository;r=w/'experiments/Pilot-C';e=w/'evidence/Pilot-C';audit=read(a.historical_audit)
 receipt=read(d/'collection_receipt.json');status=read(d/'STATUS.json')
 assert status['status']=='complete' and status['C0_updates']==status['C1_updates']==20
 assert not status['formal_SSL_started'] and not status['Primus_started'] and not status['GPU0_used']
 assert receipt['C0_preserved_unchanged'] and 'ActiveState=inactive' in receipt['service'] and 'ExecMainStatus=0' in receipt['service']
 for x in receipt['files']:assert hashlib.sha256((d/x['relative']).read_bytes()).hexdigest()==x['sha256']
 data={};allrows=[];initial=[];complete=[];coordinates=[]
 for branch in ['C0','C1']:
  for rank in range(3):
   folder=d/'smoke'/branch
   rows=[json.loads(x) for x in (folder/f'steps_rank{rank}.jsonl').read_text().splitlines()]
   assert len(rows)==20 and [x['step'] for x in rows]==list(range(30001,30021))
   data[branch,rank]=rows;allrows+=rows
   init=read(folder/f'initial_rank{rank}.json');done=read(folder/f'complete_rank{rank}.json')
   assert init['student_strict_exact'] and init['teacher_strict_exact'] and init['optimizer_steps']==[30000] and init['teacher_is_model_ema']
   assert done['updates']==20 and done['last_step']==30020 and done['optimizer_steps']==[30020] and done['anchor_no_grad']
   assert done['rank_parameter_max_difference']<=1e-6 and not done['checkpoint_written']
   initial.append(dict(init,branch=branch));complete.append(dict(done,branch=branch))
   if branch=='C1':
    c=read(folder/f'coordinate_check_rank{rank}.json');assert c['status']=='PASS' and c['independent_official_layer23_relative_L2']==0 and c['max_abs_difference']==0
    coordinates.append(dict(c,rank=rank))
 comparisons=[]
 for rank in range(3):
  x,y=data['C0',rank],data['C1',rank]
  assert next(z for z in initial if z['branch']=='C0' and z['rank']==rank)['RNG_restored_hash']==next(z for z in initial if z['branch']=='C1' and z['rank']==rank)['RNG_restored_hash']
  for i,(v0,v1) in enumerate(zip(x,y)):
   fields=['batch_first_index','sample_digest','sample_chain','tensor_hashes','before_model_RNG','after_model_RNG','lr','weight_decay','teacher_temp','teacher_momentum','global_batch']
   assert all(v0[k]==v1[k] for k in fields)
   historic=audit['historical_first20'][str(rank)][i]
   assert historic['step']==v0['step'] and historic['sample_chain']==v0['sample_chain'] and historic['loss']==v0['loss'] and historic['lr']==v0['lr']
   comparisons.append({'rank':rank,'step':v0['step'],'all_inputs_RNG_schedule_equal':True,'historical_C0_loss_difference':v0['loss']-historic['loss'],'historical_sample_chain_equal':True})
 c1=[x for x in allrows if x['branch']=='C1']
 for x in c1:
  assert x['diagnostic_student_tokens_max_difference']==0 and not x['anchor_gradients_present'] and x['affinity_dtype']=='torch.float32'
  assert x['barrier_weight']==1 and x['barrier_raw']>0 and x['barrier_backbone_grad_norm_local']>0
  assert all(math.isfinite(x[k]) for k in ['loss','barrier_raw','barrier_backbone_grad_norm_local','vanilla_backbone_grad_norm_local','production_estimate_seconds'])
 steps=[]
 for branch in ['C0','C1']:
  for i in range(20):
   rows=[data[branch,rank][i] for rank in range(3)]
   row={'branch':branch,'step':rows[0]['step'],'observed_bottleneck_seconds':max(x['iteration_seconds'] for x in rows),'production_estimate_bottleneck_seconds':max(x.get('production_estimate_seconds',x['iteration_seconds']) for x in rows),'data_seconds_max':max(x['data_seconds'] for x in rows),'peak_allocated_bytes_max':max(x['peak_allocated_bytes'] for x in rows)}
   if branch=='C1':row.update(active_fraction_global=sum(x['active_pairs'] for x in rows)/sum(x['selected_pairs'] for x in rows),barrier_raw_rank_mean=statistics.mean(x['barrier_raw'] for x in rows),barrier_grad_norm_local_mean=statistics.mean(x['barrier_backbone_grad_norm_local'] for x in rows),gradient_ratio_local_mean=statistics.mean(x['weighted_barrier_to_vanilla_grad_ratio'] for x in rows),anchor_forward_seconds_max=max(x['anchor_forward_seconds'] for x in rows),diagnostic_seconds_max=max(x['diagnostic_VJP_seconds']+x['diagnostic_norm_seconds'] for x in rows))
   steps.append(row)
 throughput={}
 for branch in ['C0','C1']:
  steady=[x for x in steps if x['branch']==branch and x['step']>=30006]
  throughput[branch]={k:statistics.median(x[k] for x in steady) for k in ['observed_bottleneck_seconds','production_estimate_bottleneck_seconds','data_seconds_max']}
  throughput[branch]['production_estimate_range_seconds']=[min(x['production_estimate_bottleneck_seconds'] for x in steady),max(x['production_estimate_bottleneck_seconds'] for x in steady)]
  throughput[branch]['peak_allocated_GiB']=max(x['peak_allocated_bytes_max'] for x in steps if x['branch']==branch)/(1<<30)
 base=throughput['C0']['observed_bottleneck_seconds'];method=throughput['C1']['production_estimate_bottleneck_seconds'];instrumented=throughput['C1']['observed_bottleneck_seconds']
 head=read(w/'experiments/Pilot-B/downstream/runs/Trajectory-Lerp05/seed20261014/training_complete.json')['seconds'];full=read(w/'experiments/Pilot-B/STATUS.json')['seconds']
 ssl_hours=(base+method)*15000/3600;ssl_instrumented=(base+instrumented)*15000/3600
 low=sum(throughput[b]['production_estimate_range_seconds'][0] for b in ['C0','C1'])*15000/3600
 high=sum(throughput[b]['production_estimate_range_seconds'][1] for b in ['C0','C1'])*15000/3600
 budget={'C0_15k_hours':base*15000/3600,'C1_15k_production_estimate_hours':method*15000/3600,'paired_SSL_production_estimate_hours':ssl_hours,'paired_SSL_with_full_diagnostics_hours':ssl_instrumented,'two_Primus40k_hours_historical_reference':2*head/3600,'two_full_downstream_hours_historical_reference':2*full/3600,'SSL_plus_two_heads_hours':ssl_hours+2*head/3600,'SSL_plus_full_downstream_hours':ssl_hours+2*full/3600,'if_full_diagnostics_retained_full_total_hours':ssl_instrumented+2*full/3600,'steady_step_extrema_extrapolation_full_total_hours':[low+2*full/3600,high+2*full/3600],'additional_anchor_forward_median_ms':statistics.median(x['anchor_forward_seconds'] for x in c1)*1000,'C1_production_overhead_fraction':method/base-1,'limits':'20-step extrapolation, not a measured15k run; remove isolated diagnostic student forward/VJP to attain production estimate; I/O/checkpoint/startup and long-run variation not proven by20steps'}
 summary={'status':'VERIFIED_COMPLETED','C0_updates':20,'C1_updates':20,'paired_rank_steps':60,'unit_tests':6,'all_paired_inputs_RNG_schedule_equal':True,'C0_historical_first20_losses_identical':True,'coordinate_checks':coordinates,'throughput':throughput,'budget':budget,'barrier':{'lambda_smoke':1,'lambda_recommended_locked':1,'lambda_search_performed':False,**{k:{'min':min(x[k] for x in c1),'median':statistics.median(x[k] for x in c1),'max':max(x[k] for x in c1)} for k in ['barrier_raw','active_fraction','barrier_backbone_grad_norm_local','weighted_barrier_to_vanilla_grad_ratio']},'weighted_loss_fraction_median':statistics.median(x['barrier_raw']/x['vanilla_loss'] for x in c1),'anchor_gradients_present':False,'NaN_or_Inf':False},'formal_SSL_or_Primus_started':False,'further_experiments_started':False,'actual_source_commits':{'C0':'abf21bb703e01393a89d238e394e8d297d8dcc9a','C1':read(d/'SOURCE.json')['runtime_source_commit']}}
 csvout(r/'PilotC1_per_rank_steps.csv',allrows);csvout(r/'PilotC1_per_step.csv',steps);csvout(r/'PilotC1_pairing_checks.csv',comparisons)
 save(r/'PilotC1_summary.json',summary);save(e/'paired_consistency.json',{'status':'PASS','paired_rank_steps':60,'all_inputs_RNG_schedule_equal':True,'historical_C0_loss_and_sampling_exact':True,'coordinate_max_abs_difference':0,'optimizer_state_30000_to_30020':True,'frozen_anchor_unchanged':True,'unit_tests':6,'C0_files_preserved':True})
 save(e/'restored_state_checks.json',{'initial':initial,'complete':complete,'coordinate_checks':coordinates})
 save(e/'stopped_receipt.json',{'verified_utc':receipt['utc'],'service_active':False,'service_has_main_process':False,'exit_code':0,'website_http':receipt['website_http'],'status':status,'C0_preserved_unchanged':True,'environment':receipt['environment']})
 save(e/'runtime_source.json',read(d/'SOURCE.json'))
 save(e/'input_receipt_hashes.json',{'private_collection_receipt_sha256':hashlib.sha256((d/'collection_receipt.json').read_bytes()).hexdigest(),'files':[{'relative':x['relative'],'sha256':x['sha256'],'bytes':x['bytes']} for x in receipt['files']],'raw_server_paths_and_process_metadata':'retained privately; not published'})
 print(json.dumps({'status':'PASS','throughput':throughput,'budget':budget,'barrier':summary['barrier']}))
if __name__=='__main__':main()
