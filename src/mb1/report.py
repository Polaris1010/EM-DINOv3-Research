"""Durable seven-benchmark results; paper reference and measured metrics stay separate."""
import csv,math,statistics
from pathlib import Path
from common import R,read,sha,atomic,config,manifest,rows,Planes
from metrics import METRICS,average,distribution,equal_model_dataset
MODELS=['G-P0','G-B100','Official-MitoNet-v1','MitoNet-MV']
def fmt(v):return 'NA' if v is None else 'inf' if math.isinf(v) else f'{v:.4f}'
def csvout(path,items,fields=None):
 fields=fields or list(dict.fromkeys(k for x in items for k in x))
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader()
  for item in items:w.writerow({k:format(v,'.7g') if isinstance(v,float) else v for k,v in item.items() if k in fields})
def make_figures(images):
 import numpy as np,matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=R/'figures';out.mkdir(exist_ok=True);selected=[(r,r['planes']//2) for r in rows() if r['type']=='volume'];tem=[r for r in rows() if r['type']=='image'];selected += [(tem[i],0) for i in [0,49,99]];results=[]
 lookup={(r['run_id'],r['entry_id'],r['z']):r for r in images}
 for row,z in selected:
  image=Planes(row['image_path'],row['shape_zyx']);gt=Planes(row['label_path'],row['shape_zyx']);im=np.asarray(image[z]);truth=np.asarray(gt[z])>0
  panels=[im,truth];titles=['Original native image','GT foreground (instance>0)']
  for rid,label in zip(['P0','B0','O','M'],['G-P0 seed20261014','G-B100 seed20261014','Official MitoNet','MitoNet-MV']):
   mask=np.load(R/'runs'/rid/(row['id']+'_prediction.npy'),mmap_mode='r')[z];panels.append(mask);titles.append(label+'\nDice '+fmt(lookup[rid,row['id'],z]['dice']))
  fig,axes=plt.subplots(1,6,figsize=(24,4.5))
  for i,(ax,p,title) in enumerate(zip(axes,panels,titles)):
   ax.imshow(p,cmap='gray',vmin=0,vmax=255 if i==0 else 1,interpolation='nearest');ax.set_title(title,fontsize=11);ax.axis('off')
  fig.suptitle(f'{row["dataset"]} | {row["id"]} | native Z={z} | fixed input-only representative selection',fontsize=11,y=.98);fig.tight_layout(rect=(0,0,1,.87));p=out/(row['id']+f'_z{z:04d}.png');fig.savefig(p,dpi=140);plt.close(fig);image.close();gt.close()
  results.append({'dataset':row['dataset'],'entry_id':row['id'],'z':z,'path':str(p),'filename':p.name,'sha256':sha(p),'selection':'middle native Z of each volume; sorted TEM images0,49,99; chosen before metrics','dinov3_display_seed':20261014,'all_three_seeds_in_numerical_results':True})
 atomic(R/'MB1_figure_manifest.json',{'figures':results,'panoptic_files':'runs/{O,M}/*_panoptic.npy retained; no3D instance identity claim'});return results
def overlap_report():
 data=read(R/'evidence/overlap_source_check.json');lines=['# MB1来源与训练暴露状态','',
 '所有七组照常评测。以下是有限来源/资产核对，不做全面数据审计，不据此筛掉数据或改变推理。','',
 '| Benchmark | B100k无标签池与资产线索 | MV3/MV5监督暴露 |','|---|---|---|']
 for r in data['rows']:
  supervised='候选volume不在156 train中；同来源/采集关系仍须谨慎解释' if 'candidate_volume_id' in r else '未发现TEM benchmark名称匹配；精确图像暴露仍UNVERIFIED'
  lines.append('| '+r['dataset']+' | '+r['status']+' | '+supervised+' |')
 lines+=['','五个Conrad23来源已列入B100的212-volume SSL清单，形状相同，首/中/末三平面与官方原图逐像素相等：确认来源与被抽样图像重叠；没有声称全体积逐像素/哈希比较。Lucchi++与casser20_lucchi++有来源关联，但原生形状[165,768,1024]对[82,768,512]不同，精确坐标暴露未确认；未做重采样、尺度或空间搜索。','',
 'TEM-100在现有212-volume来源/文件名清单中未发现匹配，属于NO_OVERLAP_FOUND而非已证实无图像暴露。没有穷举比对CEM1.5M、CEM-MitoLab或自然图像上游训练池，官方MitoNet的精确上游图像交叉关系仍UNVERIFIED。','',
 'G-P0没有本项目EM持续预训练，但其decoder使用MV3监督数据。G-B100具有上述无标签暴露。MitoNet-MV继承官方MitoNet再用同156区域监督适配。无标签暴露不等同监督标签泄漏，但本轮不能称严格独立、完全未见图像的外部泛化测试；三个decoder seeds不增加独立生物采集数量。','',
 '检查证据：evidence/overlap_source_check.json、已锁定SSL manifest与MV-Generalist-v2训练条目。所有官方GT仅用于固定评分/展示，不进入训练或参数选择。']
 (R/'MB1_overlap_status.md').write_text('\n'.join(lines)+'\n')
def main():
 c=config();images=[];datasets=[];distributions=[]
 for spec in c['runs']:
  data=read(R/'runs'/spec['run_id']/'metrics.json');assert data['status']=='complete' and data['protocol_sha256']==sha(R/'MB1_protocol.json') and len(data['images'])==2595 and len(data['datasets'])==7
  images+=data['images'];datasets+=data['datasets']
  for metric,d in data['tem_distributions'].items():distributions.append({'run_id':spec['run_id'],'model':spec['model'],'seed':spec['seed'],'metric':metric,**d})
 assert len(images)==20760 and len(datasets)==56
 csvout(R/'MB1_per_image_metrics.csv',images,['run_id','dataset_id','z','image_id',*METRICS]);csvout(R/'MB1_per_image_confusion.csv',images,['run_id','dataset_id','z','image_id','tp','fp','fn','tn'])
 csvout(R/'MB1_per_dataset_metrics.csv',datasets);csvout(R/'MB1_TEM_image_distributions.csv',distributions)
 means=[];extra=['pooled_dice','pooled_iou','volumetric_semantic_iou']
 for bid in range(1,8):
  for model in MODELS:
   chosen=[r for r in datasets if r['dataset_id']==bid and r['model']==model];assert len(chosen)==(3 if model in MODELS[:2] else 1)
   item={'dataset_id':bid,'dataset':chosen[0]['dataset'],'model':model,'seeds':len(chosen),**{k:average([r[k] for r in chosen]) for k in METRICS+extra},'dice_seed_sample_sd':statistics.stdev([r['dice'] for r in chosen]) if len(chosen)>1 else None,'physical_calibrated_planes':chosen[0]['calibrated_planes'],'total_planes':chosen[0]['planes']};means.append(item)
 csvout(R/'MB1_per_dataset_model_means.csv',means);macro=equal_model_dataset(datasets);seed_summary=[]
 for spec in c['runs']:
  selected=[r for r in datasets if r['run_id']==spec['run_id']];seed_summary.append({'run_id':spec['run_id'],'model':spec['model'],'seed':spec['seed'],**{k:average([r[k] for r in selected]) for k in METRICS}})
 csvout(R/'MB1_model_seed_summary.csv',seed_summary);idx={(r['model'],r['dataset_id']):r for r in means};delta=[]
 for bid in range(1,8):
  item={'dataset_id':bid,'dataset':idx['G-P0',bid]['dataset'],'B100_minus_P0_dice':idx['G-B100',bid]['dice']-idx['G-P0',bid]['dice'],'B100_minus_official_dice':idx['G-B100',bid]['dice']-idx['Official-MitoNet-v1',bid]['dice'],'adapted_minus_official_dice':idx['MitoNet-MV',bid]['dice']-idx['Official-MitoNet-v1',bid]['dice']};delta.append(item)
 csvout(R/'MB1_dataset_deltas.csv',delta);paired=[]
 for bid in range(1,8):
  for seed in [20261014,20261015,20261016]:
   p=next(r for r in datasets if r['model']=='G-P0' and r['seed']==seed and r['dataset_id']==bid);b=next(r for r in datasets if r['model']=='G-B100' and r['seed']==seed and r['dataset_id']==bid);paired.append({'dataset_id':bid,'dataset':p['dataset'],'seed':seed,'P0_dice':p['dice'],'B100_dice':b['dice'],'B100_minus_P0_dice':b['dice']-p['dice'],'B100_minus_P0_iou':b['iou']-p['iou']})
 csvout(R/'MB1_paired_seed_deltas.csv',paired)
 wins=[r['dataset'] for r in delta if r['B100_minus_official_dice']>0];sslwins=[r['dataset'] for r in delta if r['B100_minus_P0_dice']>0];regress=[r['dataset'] for r in delta if r['adapted_minus_official_dice']<0]
 figures=make_figures(images);overlap_report();physical_counts={m:{k:sum(idx[m,b][k] is not None for b in range(1,8)) for k in ['hd95_nm','nsd_16nm']} for m in MODELS}
 result={'status':'complete','datasets':7,'models':4,'runs':8,'per_image_rows':20760,'per_dataset_run_rows':56,'model_dataset_rows':28,'source_macro':macro,'B100_minus_P0_macro_dice':macro['G-B100']['dice']-macro['G-P0']['dice'],'B100_dice_wins_over_official':wins,'B100_dice_wins_over_P0':sslwins,'MitoNet_MV_dice_regressions_vs_official':regress,'seed_summary':seed_summary,'dataset_deltas':delta,'physical_metric_dataset_counts':physical_counts,'TEM_calibrated_images':manifest()['datasets'][-1]['calibrated_images'],'manifest_sha256':sha(R/'MB1_benchmark_manifest.json'),'protocol_sha256':sha(R/'MB1_protocol.json'),'representative_figures':len(figures),'strict_unseen_image_external_claim':False}
 atomic(R/'MB1_summary.json',result)
 lines=['# MB-1：Official MitoNet Benchmark Evaluation','',
 'EMPIAR-10982全部七组、八个固定run完成：G-P0/G-B100各三个MV3 decoder seeds，官方MitoNet与MV5 MitoNet-MV各一个固定模型。无监督训练、SSL、checkpoint创建/覆盖、阈值或尺度搜索；TTA OFF。','',
 '## 七组等权二维语义结果','',
 '每个体数据按全部原生Z平面计算二维指标后等层平均；TEM按全部100图平均；随后同模型等seed平均、七benchmark等权。不是全体2595层池化，也不是三seed集成预测。Dice/IoU空GT且空预测为1，undefined P/R留空并单列计数；HD95有漏检可为inf，不隐藏。',
 '','| 系统 | Dice | IoU | Precision | Recall | HD95 nm | NSD@16nm | 前景比例 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 lines+=['| '+m+' | '+' | '.join(fmt(macro[m][k]) for k in METRICS)+' |' for m in MODELS]
 lines+=['','**物理指标覆盖限制：** TEM100图无可靠独立物理标定，HD95 nm和NSD@16nm全部留空；因此上述物理宏平均只覆盖六个有可靠XY标定的benchmark，不冒充七组物理指标。未虚构Z spacing或3D nm距离。','',
 '## 每benchmark二维Dice（seed均值）','','| Benchmark | G-P0 | G-B100 | Official MitoNet | MitoNet-MV | B100−P0 |','|---|---:|---:|---:|---:|---:|']
 for d in delta:lines.append('| '+d['dataset']+' | '+' | '.join(fmt(idx[m,d['dataset_id']]['dice']) for m in MODELS)+' | '+fmt(d['B100_minus_P0_dice'])+' |')
 lines+=['','## 全体素volumetric semantic IoU：本轮实测','','以下是逐层二维推理后，对完整体数据全体素混淆计数计算的IoU，不是二维IoU的平均，也不等同作者完整3D实例流程。TEM此表另标为全图像像素池化，不用它代替TEM逐图均值。','',
 '| Benchmark | G-P0 | G-B100 | Official MitoNet | MitoNet-MV |','|---|---:|---:|---:|---:|']
 for bid in range(1,7):lines.append('| '+idx['G-P0',bid]['dataset']+' | '+' | '.join(fmt(idx[m,bid]['volumetric_semantic_iou']) for m in MODELS)+' |')
 lines.append('| TEM-100 (pooled pixels, not volume) | '+' | '.join(fmt(idx[m,7]['pooled_iou']) for m in MODELS)+' |')
 lines+=['','## 原论文公开参考，单独报告','',
 '[MitoNet原论文Table1](https://pmc.ncbi.nlm.nih.gov/articles/PMC9883049/)的volume语义IoU在3D统计，TEM是100图二维IoU均值。以下是文献数字，**不是MB1实测结果，也不是DINOv3实例指标**：','',
 '| 文献Benchmark | 文献语义IoU |','|---|---:|','| C.elegans | 0.604 |','| Fly brain | 0.902 |','| Glycolytic muscle | 0.840 |','| HeLa | 0.791 |','| Lucchi++ | 0.871 |','| Salivary gland | 0.103 |','| TEM-100图均值 | 0.717 |','',
 '本轮官方MitoNet采用固定MV4二维Engine2d而非作者的完整体数据推理/3D后处理；即便同用全体素IoU，复现差异不能仅归为模型权重或数据缺陷。未为贴近文献成绩改变参数。','',
 '## 最终问题的回答','',
 '1. G-B100二维Dice超过官方MitoNet的组：'+(', '.join(wins) or '无')+'。各组完整IoU/P/R/距离/前景比例见MB1_per_dataset_metrics.csv与model_means表。',
 f'2. B100相对P0七组等权Dice变化 {result["B100_minus_P0_macro_dice"]:+.6f}；改善组为 '+(', '.join(sslwins) or '无')+'。三个匹配seed的逐组差值保存在MB1_paired_seed_deltas.csv；不把seed数当生物学独立采集数。',
 '3. MitoNet-MV相对官方MitoNet下降组：'+(', '.join(regress) or '无')+'。这是当前MV5单seed适配模型的固定benchmark表现，不代表MitoNet微调能力上限。',
 '4. 五个Conrad23来源确认B100池来源和抽样图像重叠；Lucchi++source关联、精确区域未确认；TEM未发现名称重叠但不证明无上游暴露。六个候选volume均未直接进入MV3/MV5监督train，细节见MB1_overlap_status.md。',
 '5. B100相对P0的正向宏平均与逐组变化需要分别解释，不能把平均收益称为所有来源一致改善。'+('本轮没有任何benchmark的B100二维Dice超过官方MitoNet，且多个来源仍严重失败，因此不支持已获得广泛通用优势的主张。' if not wins else '仅部分benchmark超过官方MitoNet，尚不能把局部优势外推为广泛通用优势。')+'来源/抽样原图SSL重叠进一步限制了严格未见图像泛化结论。当前结果是固定完整系统的表现，尚未分离自然DINOv3、Primus、多来源监督适配及EM持续预训练的因果贡献。',
 '', '本轮到此停止。没有自动启动新训练、更多seed、独立诊断或参数搜索。','',
 '## 可复现配置和产物','',
 'DINOv3直接复用MV3 mv_probe.sliding/build与原版Primus[5,11,17,23]，native-grid、crop256/stride128、tile_batch16、既有归一化和mean-softmax>0.5。两个MitoNet原样复用MV4 Engine2d/Preprocessor，scale1/confidence0.5、NMS3、整图、TTA OFF；逐层Panoptic保留，不读取GT调输入或参数。','',
 'GT仅使用原官方实例TIFF>0形成语义真值，完整原始XY/Z，不用GT box，不删空GT图像。212个原始TIFF文件长度与官方HEAD一致，逐文件计算SHA留档；官方未提供用于比对的SHA，不宣称原始文件SHA与官方摘要校验通过。10项既有checkpoint/encoder资产SHA已与实验记录复核。GTsigned实例ID格式被正常读取，未重采样、反相或改变有效区域。','',
 f'Manifest SHA `{result["manifest_sha256"]}`；protocol SHA `{result["protocol_sha256"]}`。模型/run/seed/path/SHA对照在MB1_protocol.json；逐平面结果用短run_id与dataset_id避免重复，均可回查完整manifest。CSV浮点保留7位有效数字，完整未舍入JSON和预测在服务器。','',
 'MB1_per_image_metrics.csv：20760行二维指标；MB1_per_image_confusion.csv保留精确整数计数。MB1_per_dataset_metrics.csv为56行run×benchmark；MB1_per_dataset_model_means.csv为28行等seed均值。MB1_TEM_image_distributions.csv保留100图均值/SD/分位/缺失/inf计数，分位数用经验逆CDF、不插值。所有seeds和所有图像都进入统计。','',
 '代表图固定选每volume中间Z和TEM按文件名排序0/49/99图，共9张，四模型均在同图同GT上展示；DINOv3显示首seed，其余seed完整计入数字。MB1_figure_manifest.json保存文件名、SHA和选择规则。没有计算DINOv3实例AP/PQ/F1@75。','',
 '官方来源：[EMPIAR-10982](https://www.ebi.ac.uk/empiar/EMPIAR-10982/)、[torch-em CEM benchmark接口](https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/electron_microscopy/cem.py)。来源/无标签暴露边界见独立overlap报告。']
 (R/'MB1_summary.md').write_text('\n'.join(lines)+'\n')
 atomic(R/'MB1_results_verification.json',{'status':'PASS','metric_rows':20760,'dataset_run_rows':56,'model_dataset_rows':28,'runs':8,'datasets':7,'manifest_sha256':result['manifest_sha256'],'protocol_sha256':result['protocol_sha256'],'calibration_missing_not_fabricated':True,'tta':False,'checkpoint_writes':False,'training':False})
if __name__=='__main__':main()
