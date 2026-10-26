"""Make a source-backed pilot readout and scientific figures; no new experiment selection."""
import collections
import csv
import json
import math
from pathlib import Path
import numpy as np
from common import R,read,atomic,sha,npy


def rows(name):
    with (R/'tables'/name).open() as handle:return list(csv.DictReader(handle))


def num(value):
    return float(value) if value not in (None,'') else float('nan')


def fmt(value):
    value=num(value)
    return f'{value:.4f}' if math.isfinite(value) else 'NA'


def main():
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.fonttype']='none'
    import matplotlib.pyplot as plt
    protocol=read(R/'protocol.json');items=read(R/'samples.json');layers=protocol['layers'];models=[r['id'] for r in protocol['models']]
    sep=rows('separability.csv');paired=rows('D2_paired_AUROC.csv');spec=rows('spectrum.csv');metrics=rows('source_readout_metrics.csv');corr=rows('periodicity_correlations.csv')
    (R/'figures').mkdir(exist_ok=True)
    sources=protocol['diagnostic_sources'];color=dict(zip(models,['#555555','#32965b','#e9a42b','#d24c45','#4b75bc']))
    fig,axes=plt.subplots(2,5,figsize=(17,7),sharex=True)
    for column,source in enumerate(sources):
        for ri,key in enumerate(['linear_AUROC','prototype_AUROC']):
            ax=axes[ri,column]
            for model in models:
                values=[num(next(r[key] for r in sep if r['source']==source and r['model']==model and int(r['layer'])==layer and r['background_set']=='all_far_background')) for layer in layers]
                ax.plot(layers,values,'o-',label=model,color=color[model],lw=1.3)
            ax.set_title(source if ri==0 else key);ax.set_xticks(layers);ax.set_ylim(0,1);ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5);fig.tight_layout(rect=(0,.06,1,1));fig.savefig(R/'figures/AUROC_all_sources.png',dpi=170);fig.savefig(R/'figures/AUROC_all_sources.svg');plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(17,3.8),sharey=True)
    for ax,source in zip(axes,sources):
        for model in models:
            values=[[num(r['effective_rank']) for r in spec if r['source']==source and r['model']==model and int(r['layer'])==layer] for layer in layers]
            ax.errorbar(layers,[np.mean(v) for v in values],yerr=[np.std(v,ddof=1) for v in values],label=model,color=color[model],marker='o',lw=1.2)
        ax.set_title(source);ax.set_xticks(layers);ax.grid(alpha=.2)
    axes[0].set_ylabel('Effective rank (512 matched tokens)');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5);fig.tight_layout(rect=(0,.12,1,1));fig.savefig(R/'figures/effective_rank_all_sources.png',dpi=170);fig.savefig(R/'figures/effective_rank_all_sources.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,target in zip(axes,['linear','prototype']):
        a=np.array([[num(next(r['delta'] for r in paired if r['source']==s and int(r['layer'])==layer and r['score']==target)) for layer in layers] for s in sources])
        lim=max(.01,float(np.nanmax(np.abs(a))));image=ax.imshow(a,vmin=-lim,vmax=lim,cmap='RdBu');ax.set_xticks(range(4),layers);ax.set_yticks(range(5),sources);ax.set_title('D2 100k - 30k: '+target+' AUROC')
        for iy,ix in np.ndindex(a.shape):ax.text(ix,iy,f'{a[iy,ix]:+.3f}',ha='center',va='center',fontsize=9)
        fig.colorbar(image,ax=ax,shrink=.8)
    fig.tight_layout();fig.savefig(R/'figures/D2_paired_AUROC_delta.png',dpi=170);fig.savefig(R/'figures/D2_paired_AUROC_delta.svg');plt.close(fig)
    # Raw-image examples stay as review artifacts, never raw data in Git.
    chosen=[next(i for i in items if i['source']=='glycolytic_muscle' and i['subset']=='random'),next(i for i in items if i['subset']=='historical_D2_error_case')]
    labels_np=np.load(R/'evaluation_labels.npz');images=npy(R/'images.npy');fig,axes=plt.subplots(2,6,figsize=(15,5.8))
    for ri,item in enumerate(chosen):
        k=item['id'];axes[ri,0].imshow(images[k],cmap='gray',vmin=0,vmax=255);axes[ri,0].set_title(item['subset']+' z'+str(item['z']))
        axes[ri,1].imshow(labels_np['GT'][k],cmap='gray',vmin=0,vmax=1);axes[ri,1].set_title('GT')
        for ci,(model,readout) in enumerate([('D2-30k','Primus'),('D2-100k','Primus'),('D2-30k','linear23'),('D2-100k','linear23')],2):
            if readout=='Primus':value=npy(R/('primus_'+model+'.npy'))[k]
            else:value=npy(R/'models'/model/'linear_probability.npy')[k,3].reshape(16,16)
            axes[ri,ci].imshow(value,cmap='magma',vmin=0,vmax=1);axes[ri,ci].set_title(model+' '+readout)
        for ax in axes[ri]:ax.axis('off')
    fig.tight_layout();fig.savefig(R/'figures/muscle_examples_REVIEW_ONLY.png',dpi=170);plt.close(fig)
    def metric(model,layer,key):
        return num(next(r[key] for r in metrics if r['model']==model and r['source']=='glycolytic_muscle' and r['layer']==str(layer)))
    muscle=[r for r in paired if r['source']=='glycolytic_muscle']
    negative=[layer for layer in layers if all(num(next(r['delta'] for r in muscle if int(r['layer'])==layer and r['score']==target))<0 for target in ['linear','prototype'])]
    positive=[layer for layer in layers if all(num(next(r['delta'] for r in muscle if int(r['layer'])==layer and r['score']==target))>0 for target in ['linear','prototype'])]
    q1='SUPPORTED' if len(negative)>=3 else ('CONTRADICTED' if len(positive)>=3 else 'INCONCLUSIVE')
    dice_delta=metric('D2-100k',23,'dice')-metric('D2-30k',23,'dice');fp_delta=metric('D2-100k',23,'far_FP_rate')-metric('D2-30k',23,'far_FP_rate')
    q3='SUPPORTED' if dice_delta<0 and fp_delta>0 else ('CONTRADICTED' if dice_delta>0 and fp_delta<0 else 'INCONCLUSIVE')
    primary=[r for r in corr if r['source']=='glycolytic_muscle' and r['model']=='D2-100k' and r['group']=='source_all' and r['target']=='Primus_FP_fraction']
    strata=[r for r in corr if r['source']=='glycolytic_muscle' and r['model']=='D2-100k' and r['group'].startswith('stratum:') and r['target']=='Primus_FP_fraction']
    association=primary and num(primary[0]['spearman'])>0 and num(primary[0]['partial_rank_correlation'])>0 and sum(num(r['spearman'])>0 for r in strata)>=2
    q4='SUPPORTED' if association else 'INCONCLUSIVE'
    verdicts=[{'question':1,'status':q1,'finding':'Glycolytic muscle: both train-only scores decline at layers '+str(negative)+'; both increase at '+str(positive)+'.'},
        {'question':2,'status':'SUPPORTED' if q1=='SUPPORTED' else 'INCONCLUSIVE','finding':'Layerwise deltas and block intervals are reported; no whole-encoder collapse inferred.'},
        {'question':3,'status':q3,'finding':f'Layer23 linear crop Dice delta={dice_delta:+.6f}; remote FP-rate delta={fp_delta:+.6f}.'},
        {'question':4,'status':q4,'finding':'Descriptive within-source association only; contrast/gradient/position strata and partial ranks reported.'},
        {'question':5,'status':'SUPPORTED' if q1==q3==q4=='SUPPORTED' else 'INCONCLUSIVE','finding':'Only a research rationale, never evidence that a proposed SSL constraint works.'}]
    atomic(R/'verdicts.json',verdicts)
    lines=['# Feature Mechanism Pilot A Report','', '## 实验发现','',
        '本轮比较P0、D1/100k、D2/30k、D2/100k与Coverage-NR/100k，四层[5,11,17,23]。所有模型使用同一预先锁定坐标、native256视野和官方post-norm patch tokens；CLS与4个register tokens单独缓存并从patch统计剔除。', '',
        '| 问题 | 裁决 | 直接观察 |','|---|---|---|']
    for v in verdicts:lines.append('| '+str(v['question'])+' | '+v['status']+' | '+v['finding']+' |')
    lines+=['','裁决是固定样本上的探索性诊断。问题1按两种train-only读出的四层变化是否多数同向判断，块bootstrap区间列在原表；问题4表示可观察的描述性关联。它们不提供独立acquisition的因果证据。', '',
        '### D2同轨迹30k→100k：肌肉patch AUROC', '', '| 层 | 读出 | 30k | 100k | 差值 | Z块bootstrap 95%区间 |','|---|---|---:|---:|---:|---|']
    for r in muscle:lines.append('| '+r['layer']+' | '+r['score']+' | '+' | '.join(fmt(r[k]) for k in ['D2_30k_AUROC','D2_100k_AUROC','delta'])+' | ['+fmt(r['delta_bootstrap_low'])+', '+fmt(r['delta_bootstrap_high'])+'] |')
    lines+=['','### 相同随机视野上的肌肉分割', '', '| 模型 | 读出 | Dice | Precision | Recall | 远端FP率 |','|---|---|---:|---:|---:|---:|']
    for m in models:
        for layer in ['23','Primus']:
            r=next(v for v in metrics if v['source']=='glycolytic_muscle' and v['model']==m and v['layer']==layer)
            lines.append('| '+m+' | '+layer+' | '+' | '.join(fmt(r[k]) for k in ['dice','precision','recall','far_FP_rate'])+' |')
    lines+=['','## 抽样、拟合与统计口径','',
        '抽样seed20261027；MV3 train全部156区域各2个随机视野，五个诊断来源各64个随机视野，另1个历史D2误检案例。历史案例不进入随机总体统计。所有坐标在首次特征提取前锁定，未用D2失败筛选随机样本。', '',
        '所有特征以bfloat16 autocast计算、官方post-norm后统一保存为float16缓存；初始receipt称“精确表示BF16输出”不准确，原输出dtype未被捕获，另有cache_precision_note保留更正。相同量化配置用于全部模型，不将缓存视为全精度特征。', '',
        '每层1x1 logistic probe只在MV3 train拟合，固定500 AdamW更新、lr0.03、weight decay0.001、source/class-balanced minibatch4096、阈值0.5。按train计算来源均衡channel mean/std和前景/背景原型；所有probe hash固定后才读取开发与MB1 GT。没有SSL或Primus训练。', '',
        '前景为有效patch中GT占比≥0.5；远端背景为patch无GT且≥95%面积在同一标注XY平面距GT>180nm。所有远端背景指锁定随机视野中的全部合格patch，不是原始体积全域。随机远端背景每来源固定最多1024个；D2特定FP集合为这些远端背景中旧D2/100k Primus预测占比≥0.5，单列为事后有偏错误集合。', '',
        '谱统计每模型/来源/层使用相同512个有效patch位置，3个固定子抽样重复；有效秩上限511。以中心化特征协方差的非零特征值计算有效秩、top1方差占比和归一化谱分布，另报类内/类间余弦、train中心化余弦、Fisher ratio、各向异性及总方差。三个子抽样不是三次训练或独立生物重复。没有对权重做SVD或修改。', '',
        '周期性为固定64×64 Hann窗在4–32像素周期频带内的top4非DC功率比例。记录对比度、梯度能量、图像位置、周围GT密度和native spacing/FOV；对比度分位和梯度分层阈值仅由train估计。报告来源内/volume内Spearman、固定分层及partial rank；同来源spacing常量不能估计尺度效应，不跨来源合并制造相关。', '',
        'AUROC与部分主对照相关区间按volume_id和Z//8块重采样200次，仅作为这些已暴露数据中的不确定性描述。patch不是独立acquisition；OpenOrganelle多个crop共享acquisition。图像块重叠、单volume肌肉和已暴露MB1限制结论范围。', '',
        '## 仍无法解释的现象','',
        '有效秩下降本身不能区分有害语义混淆与正常各向异性变化；固定线性读出与原版Primus具有不同容量和空间上下文。频谱关联亦可能反映未观测的结构、染色和物理尺度差异。即使本轮SUPPORTED，也不能认定周期纹理是因果机制，或Topo-Anchor/Gram一定有效。', '',
        '## 唯一后续机制建议','',
        ('优先研究train-only证据定位的局部前景/远端背景关系保持，作为单一预训练约束候选；后续仍需与同初始化、同预算Vanilla及官方Gram Anchoring区分。本轮仅建议，不执行Pilot C。' if q1==q3=='SUPPORTED' else '优先研究冻结表征到分割读出的背景响应与校准失配；本轮证据未同时支持稳定表征退化及简单读出复现，不据此实施Topo-Anchor。先保留这个机制判断，不自动追加实验。'), '',
        '## 交付和停止','',
        '完整CSV在tables目录：全部背景集合的separability、spectrum与归一化谱分布、逐crop/逐来源readout指标、周期性控制变量和分层相关、D2逐层配对变化。缺少前景、远端背景或方差时保留NA，不按结果修改阈值或来源。特征缓存、线性权重及完整window响应仅保留独立服务器目录。', '',
        '图表：AUROC_all_sources.png、effective_rank_all_sources.png、D2_paired_AUROC_delta.png；原图/GT/预测示例muscle_examples_REVIEW_ONLY.png为人工复核派生图，不提交原图到Git。代码commit、五模型SHA、环境、样本坐标、train-only读写证据随本轮归档。', '',
        'Coverage-NR是2026-10-22已完成历史资产。Pilot A使用GPU1/2/3的确认空闲计算资源；GPU0禁用、网站保持。Pilot B/C、权重融合、权重SVD、Gram或新SSL均未启动。完成后停止。','']
    (R/'Feature_Mechanism_Pilot_Report.md').write_text('\n'.join(lines)+'\n')
    atomic(R/'figure_manifest.json',[{'name':p.name,'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'review_only':'REVIEW_ONLY' in p.name} for p in sorted((R/'figures').glob('*.png'))])


if __name__=='__main__':main()
