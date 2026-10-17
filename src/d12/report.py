"""First-seed matched D1/D2 report; existing baselines remain historical."""
import collections,csv,json,math,statistics
from common import R,M3,read,atomic,sha
KEYS=['dice','precision','recall','hd95_nm','nsd_16nm','predicted_foreground_fraction']
def csvout(name,rows):
 if not rows:return
 with (R/name).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)
def mean(values):
 values=[v for v in values if v is not None];return statistics.mean(values) if values else None
def fmt(v):return 'NA' if v is None else f'{v:.6f}'
def main():
 rows=[];hashes=[];source=[];metrics={};seed=20261014
 for model in ['D1','D2']:
  d=R/'downstream/runs'/model/f'seed{seed}';rr=[read(p) for p in d.glob('*_metrics.json')];assert len(rr)==38 and len({v['region_id'] for v in rr})==38
  assert read(d/'training_complete.json')['completed_steps']==40000 and read(d/'inference_complete.json')['regions']==38
  rows+=rr
  for s in sorted({v['source_id'] for v in rr}):
   vv=[v for v in rr if v['source_id']==s];item={'model':model,'source_id':s,'regions':len(vv),**{k:mean([v[k] for v in vv]) for k in KEYS}};source.append(item)
  metrics[model]={k:mean([v[k] for v in source if v['model']==model]) for k in KEYS}
  for step in [10000,30000]:
   p=R/'runs'/model/'formal/checkpoints'/f'step{step:09d}/teacher_backbone.pth';j=read(p.with_suffix('.json'));assert j['step']==step and j['sha256']==sha(p);hashes.append({'model':model,'kind':'SSL_EMA_encoder','step':step,'path':str(p),'sha256':j['sha256']})
  p=d/'head_step040000.pth';j=read(p.with_suffix('.json'));assert j['sha256']==sha(p);hashes.append({'model':model,'kind':'frozen_Primus_head','step':40000,'path':str(p),'sha256':j['sha256']})
 csvout('D12_per_region_metrics.csv',rows);csvout('D12_source_metrics.csv',source);csvout('D12_checkpoint_hashes.csv',hashes)
 paired=[]
 for a in [v for v in rows if v['model']=='D1']:
  b=next(v for v in rows if v['model']=='D2' and v['region_id']==a['region_id']);assert a['tp']+a['fn']==b['tp']+b['fn'] and a['valid_voxels']==b['valid_voxels']
  paired.append({'region_id':a['region_id'],'source_id':a['source_id'],'D1_dice':a['dice'],'D2_dice':b['dice'],'D2_minus_D1_dice':b['dice']-a['dice']})
 csvout('D12_paired_deltas.csv',paired)
 benchmark=[];per_image=[]
 for model in ['D1','D2']:
  j=read(R/'benchmark'/model/'metrics.json');assert j['status']=='complete';benchmark+=j['datasets'];per_image+=j['images']
 csvout('D12_benchmark_per_dataset.csv',benchmark);csvout('D12_benchmark_per_image.csv',per_image)
 benchmark_history=list(csv.DictReader((R.parent/'mb1_official_benchmark_20261015/MB1_per_dataset_metrics.csv').open()));benchmark_history=[v for v in benchmark_history if v['run_id'] in ['P0','B0']];assert len(benchmark_history)==14;csvout('D12_benchmark_historical_first_seed.csv',benchmark_history)
 curves=[];sampling={}
 for model in ['D1','D2']:
  totals=collections.Counter()
  for rank in range(3):
   folder=R/'runs'/model/'formal';totals.update(read(folder/f'sampling_totals_rank{rank}.json'))
   with (folder/f'steps_rank{rank}.jsonl').open() as f:
    for line in f:
     v=json.loads(line)
     if v['step']%100==0 or v['step']==1:curves.append({'model':model,'rank':rank,**{k:v[k] for k in ['step','loss','grad_norm','lr','data_seconds','iteration_seconds','elapsed_seconds','peak_gpu_bytes']},**v['loss_components']})
  assert sum(totals.values())==30000*48;sampling[model]=dict(totals)
 csvout('D12_training_curves_stride100.csv',curves);atomic(R/'D12_sampling_observed.json',sampling)
 original_history=list(csv.DictReader((M3/'MV3_per_region_metrics.csv').open()));original_history=[v for v in original_history if str(v.get('seed'))==str(seed)]
 assert len(original_history)==76
 history=[]
 for model in ['G-P0','G-B100']:
  for source_id in sorted({v['source_id'] for v in original_history}):
   vv=[v for v in original_history if v['model']==model and v['source_id']==source_id]
   history.append({'model':model,'source_id':source_id,'seed':seed,**{k:mean([float(v[k]) if v[k] not in ['',None] else None for v in vv]) for k in KEYS}})
 csvout('D12_historical_MV3_first_seed_reference.csv',history)
 for v in history:
  for k in KEYS:
   if k in v:v[k]=float(v[k]) if v[k] not in ['',None] else None
 historical={m:mean([v['dice'] for v in history if v['model']==m]) for m in ['G-P0','G-B100']}
 delta=metrics['D2']['dice']-metrics['D1']['dice'];by={m:{v['source_id']:v for v in source if v['model']==m} for m in ['D1','D2']}
 changed=[{'source_id':s,'D2_minus_D1_dice':by['D2'][s]['dice']-by['D1'][s]['dice']} for s in sorted(by['D1'])];csvout('D12_source_deltas.csv',changed)
 benchmark_macro={m:mean([v['dice'] for v in benchmark if v['model']==m]) for m in ['D1','D2']}
 result={'status':'complete','decoder_seed':seed,'ssl_steps_each':30000,'decoder_steps_each':40000,'development_regions_each':38,'development_sources':12,'metrics':metrics,'D2_minus_D1_dev_dice':delta,'historical_MV3_first_seed_dice':historical,'historical_B100_not_pure_sampling_control':True,'benchmark_diagnostic_macro_dice':benchmark_macro,'source_deltas':changed,'further_experiments_started':False,'auto100k':False,'autoD3':False}
 atomic(R/'D12_summary.json',result)
 lines=['# D1/D2：多来源 SSL 数据扩展第一轮','',
 '两组均从官方自然DINOv3 ViT-L/16开始，DINO+iBOT+0.1KoLeo、相同seed0/global batch48、相同优化器和30k cosine预算，保存10k/30k。没有新loss、Gram、HRA、TopoCL或新decoder；没有B100续训。固定选择30k EMA encoder，重新训练同seed20261014原版Primus40k，全程冻结encoder、同156train/38dev、TTA OFF。','',
 '## 固定开发结果','',
 '| 系统 | 来源等权Dice | Precision | Recall | HD95 nm | NSD@16nm | 前景比例 |','|---|---:|---:|---:|---:|---:|---:|']
 for m in ['D1','D2']:lines.append('| '+m+' | '+' | '.join(fmt(metrics[m][k]) for k in KEYS)+' |')
 lines+=['','| 来源 | D1 Dice | D2 Dice | D2−D1 |','|---|---:|---:|---:|']
 for item in changed:
  s=item['source_id'];lines.append('| '+s+' | '+fmt(by['D1'][s]['dice'])+' | '+fmt(by['D2'][s]['dice'])+' | '+fmt(item['D2_minus_D1_dice'])+' |')
 lines+=['','## 本轮问题与决策','',
 '1. **来源均衡采样本身是否有效：本轮无法单独因果识别。** D1同时采用新SSL留出池和30k预算，历史B100为全池100k且曾无标签接触评测图像；D1与B100差异不能只归于sampling。旧结果原样保留，首seed历史参考Dice：'+str(historical)+'。没有擅自增加第三组训练。',
 '2. **增加CEM是否进一步改善：** 同协议D2−D1来源等权Dice = '+f'{delta:+.6f}'+'。这是本轮匹配对照，包含已披露的CEM原生尺寸格式适配；仅一个decoder seed，不宣称统计显著性。',
 '3. **是否保留原有正向来源：** 逐来源开发变化见上表，Fly/HeLa/Lucchi等另见固定MB1诊断表；不是只报告总均值。MB1已参与研究决策，不能称盲测，也不用于选择配比、loss或checkpoint。',
 '4. **是否进入局部结构一致性：** '+('CEM扩展具有正向初步信号，可据逐来源代价讨论后续D2/D3消融；尚不能由一个seed认定收益稳定。' if delta>0 else '本轮没有正向平均收益，不支持自动延长数据扩展；下一步应由研究者结合逐来源变化决定转向尺度或局部结构假设。')+' 本轮均到此停止，不自动启动D3、100k或更多seed。','',
 '## 已暴露MB1诊断：二维逐图/逐层Dice','',
 '| Benchmark | 历史P0首seed | 历史B100首seed | D1 | D2 | D2−D1 |','|---|---:|---:|---:|---:|---:|']
 for bid in range(1,8):
  a=next(v for v in benchmark if v['model']=='D1' and v['dataset_id']==bid);b=next(v for v in benchmark if v['model']=='D2' and v['dataset_id']==bid);p=next(v for v in benchmark_history if int(v['dataset_id'])==bid and v['run_id']=='P0');h=next(v for v in benchmark_history if int(v['dataset_id'])==bid and v['run_id']=='B0');lines.append('| '+a['dataset']+' | '+fmt(float(p['dice']))+' | '+fmt(float(h['dice']))+' | '+fmt(a['dice'])+' | '+fmt(b['dice'])+' | '+fmt(b['dice']-a['dice'])+' |')
 lines+=['','MB1七组等权Dice：'+str(benchmark_macro)+'。六volume全体素IoU单列CSV，TEM无可靠标定不填nm；逐层二维推理不是作者完整3D实例流程。本轮不计算DINO实例AP/PQ。','',
 '## 数据与证据边界','',
 '[MitoNet原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9883049/)明确将benchmark及关联图像从CEM1.5M/CEM-MitoLab排除。因此D1/D2共同移除MitoVerse正式test、已映射validation/dev及已知MB1关联资产；D2从作者原始CEM再移除可识别的MV3留出关联集合。CEM未读取CEM-MitoLab标签。','',
 '实际池：153个MitoVerse体积、12来源、34143合法Z；CEM原包1592753图/651目录记录（640非空），排除13个可识别关联集合后1425433图/627非空集合。索引SQLite及ZIP偏移/逐成员CRC提供无损随机访问；数据与排除理由见mitoverse_pool.json、cem_pool.json。集合/volume代理不虚构独立acquisition。','',
 '未知哈希CEM集合与MV3开发图像的精确交叉关系仍UNVERIFIED。固定MV3监督协议也不因本轮改变，同来源/关联采集暴露限制继续存在；仅采用SSL留出原则并不能把完整系统结果升级成严格独立acquisition外测。自然权重的上游暴露亦未穷举。','',
 'MitoVerse保留既有512裁剪与小图reflect padding；CEM保留原生PIL RGB（通常224），不预先放大或补成512，采用同一官方RandomResizedCrop输出256/112，存在不同native FOV/插值因素，不把它们解释成纯数据数量效应。','',
 '10k仅导出留档；所有下游使用固定30k，并且两组共同40k head预算。D12_checkpoint_hashes.csv提供10k/30k EMA及head路径/SHA，训练曲线、吞吐/显存和来源抽样计数保留各rank JSONL。原B100/MV1至MV5/MB1未改写。','',
 '输出：D12_per_region_metrics.csv、D12_source_metrics.csv、D12_paired_deltas.csv、D12_benchmark_per_image.csv、D12_benchmark_per_dataset.csv、D12_summary.json及本报告。服务终止证据和历史输入前后快照由有界控制器保存；代码/指标/Markdown由本轮收取器提交推送。']
 (R/'D12_final_report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
