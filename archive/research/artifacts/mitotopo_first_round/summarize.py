import json,csv,re,statistics
from pathlib import Path
p=Path('artifacts/mitotopo_first_round');r=p/'remote'
resources=list(csv.DictReader((r/'resource_log.csv').open()))
train=[]
for line in (r/'train.log').read_text().splitlines():
    if line.startswith('{'):
        try:train.append(json.loads(line))
        except ValueError:pass
metrics={
 'status':'stopped_resource_guard_not_passed','planned_steps':200,'last_logged_step':max(x['step'] for x in train),'completed_steps_exact':None,
 'stop_reason':'System swap used grew in four consecutive monitor samples; guard sent SIGINT then SIGTERM after 20s because nohup-inherited SIGINT was ignored',
 'training_checkpoint_saved':False,'checkpoint_reload_verified':False,'initial_loss_reproducibility':'in-process assertion passed before training; values were not persisted',
 'peak_gpu_used_mib_sampled':max(float(x['gpu_used_mib']) for x in resources),'peak_rss_gib_sampled':max(float(x['rss_gib']) for x in resources),'peak_process_tree_rss_gib_sampled':max(float(x['rss_tree_gib']) for x in resources),'minimum_mem_available_gib_sampled':min(float(x['mem_available_gib']) for x in resources),
 'first_logged_loss':train[0]['loss'],'last_logged_loss':train[-1]['loss'],'steady_logged_compute_step_median_seconds':statistics.median(x['seconds'] for x in train[1:]),
 'validation_scope':'same five deterministic 192x192 patches across five conrad23 volumes; not full-volume metrics','validation':{},'gpu_physical_ids':[3],
 'note':'Lower train loss alongside worsening held-out Dice is not promotion evidence. No long training was launched.'}
for name in ['before','step_50','step_100','step_150']:
    m=json.loads((r/(name+'_metrics.json')).read_text());metrics['validation'][name]={'mean_patch_dice':m['mean_patch_dice'],**m['pooled_sampled_voxels']}
(p/'metrics.json').write_text(json.dumps(metrics,indent=2))
with (p/'experiments.csv').open('w',newline='') as f:
    fields=['run_id','experiment_id','commit','seed','dataset_split','gpu_physical_ids','max_gpu_mem_mib','max_rss_gib','train_time_h','semantic_dice','instance_ap50','pq','split_error','merge_error','status','conclusion']
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(dict(run_id='mitotopo_20260908_first_round',experiment_id='E0b_smoke',commit='bd6a650dda4e32d22e913cfc28458df24bb39b83+existing_dirty_files',seed=20260907,dataset_split='exploratory: jiang25_vol1 train; conrad23 five volumes val',gpu_physical_ids='3',max_gpu_mem_mib=metrics['peak_gpu_used_mib_sampled'],max_rss_gib=metrics['peak_process_tree_rss_gib_sampled'],semantic_dice=metrics['validation']['step_150']['dice'],status=metrics['status'],conclusion='Last validation is sampled-patch Dice, not completed-run/full-volume performance; swap stop; no trained checkpoint'))
print(json.dumps(metrics,indent=2))
