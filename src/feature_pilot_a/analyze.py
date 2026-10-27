"""All evaluation-label use begins after the frozen train-only probes are complete."""
import collections
import csv
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from common import R,read,atomic,sha,npy
from stats import auc,separation,spectrum,periodicity,spearman,partial_rank,bootstrap_auc_delta,bootstrap_spearman


def csvout(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (R/'tables'/name).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def tile_windows(x):
    return x.reshape(-1,4,64,4,64).permute(0,1,3,2,4).reshape(-1,64,64)


def metrics(pred,gt,valid,far):
    p=pred&valid;g=gt&valid
    tp=int((p&g).sum());fp=int((p&~g&valid).sum());fn=int((~p&g).sum());area=int(valid.sum())
    far_n=int(far.sum());far_fp=int((pred&far).sum())
    return dict(tp=tp,fp=fp,fn=fn,valid_pixels=area,dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
        precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
        far_background_pixels=far_n,far_FP_pixels=far_fp,far_FP_rate=far_fp/far_n if far_n else None,
        predicted_foreground_fraction=int(p.sum())/area if area else None)


def run():
    started=time.time();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.5)
    protocol=read(R/'protocol.json');items=read(R/'samples.json');layers=protocol['layers'];models=[x['id'] for x in protocol['models']]
    (R/'tables').mkdir(exist_ok=True);labels=np.load(R/'evaluation_labels.npz');train_labels=np.load(R/'train_labels.npz')
    fractions=labels['fraction'];valid=labels['valid'];far=labels['far'];fg=(fractions>=.5)&valid
    primus={m:npy(R/('primus_'+m+'.npy')) for m in models}
    d2fp=np.array(primus['D2-100k']).reshape(len(items),16,16,16,16).transpose(0,1,3,2,4).mean((3,4)).reshape(len(items),256)>=.5
    error=far&d2fp
    random_bg=np.zeros_like(far)
    rng=np.random.default_rng(protocol['seed'])
    for source in protocol['diagnostic_sources']:
        ids=[i['id'] for i in items if i['source']==source and i['subset']=='random']
        local=far[ids].reshape(-1);selected=np.flatnonzero(local)
        chosen=rng.choice(selected,size=min(1024,len(selected)),replace=False)
        local_mask=np.zeros_like(local);local_mask[chosen]=True;random_bg[ids]=local_mask.reshape(len(ids),256)
    np.savez(R/'diagnostic_groups.npz',foreground=fg,all_far_background=far,random_far_background=random_bg,D2_posthoc_FP=error)
    separation_rows=[];spectral_rows=[];eigen_rows=[];pixel_rows=[];window_rows=[];correlations=[]
    probs={m:npy(R/'models'/m/'linear_probability.npy') for m in models}
    protos={m:npy(R/'models'/m/'prototype_score.npy') for m in models}
    # Image-only fixed Fourier features; train-only cutoffs for control strata.
    images=npy(R/'images.npy');fourier=[]
    for first in range(0,len(items),16):
        x=torch.tensor(np.array(images[first:first+16],copy=True),device='cuda',dtype=torch.float32)
        fourier.append(periodicity(tile_windows(x)).cpu().numpy())
    fourier=np.concatenate(fourier).reshape(len(items),16,5)
    train_ids=[i['id'] for i in items if i['role']=='train']
    train_windows=train_labels['valid'][train_ids].reshape(-1,16,16).reshape(-1,4,4,4,4).transpose(0,1,3,2,4).all((3,4)).reshape(-1)
    train_values=fourier[train_ids].reshape(-1,5)[train_windows]
    train_sources=np.repeat([items[k]['source'] for k in train_ids],16)[train_windows]
    weights=np.array([1/np.count_nonzero(train_sources==s) for s in train_sources]);weights/=weights.sum()
    def weighted_quantile(x,q):
        order=np.argsort(x);cdf=weights[order].cumsum()
        return x[order[np.minimum(np.searchsorted(cdf,np.atleast_1d(q)),len(order)-1)]]
    cut_c=weighted_quantile(train_values[:,2],[1/3,2/3]);cut_g=float(weighted_quantile(train_values[:,3],[.5])[0])
    atomic(R/'train_only_periodicity_strata.json',{'contrast_tertiles':cut_c.tolist(),'gradient_median':float(cut_g),'fit_role':'MV3_train_only','source_balanced_weights':True})
    window_meta=[];window_select=[]
    for item in items:
        if item['role']!='eval':continue
        k=item['id'];G=labels['GT'][k];V=labels['valid_pixels'][k];FAR=labels['far_pixels'][k]
        for wy in range(4):
            for wx in range(4):
                sl=(slice(wy*64,(wy+1)*64),slice(wx*64,(wx+1)*64));w=wy*4+wx;f=fourier[k,w]
                eligible=bool(V[sl].all() and FAR[sl].mean()>=.95 and not G[sl].any())
                row={'sample_id':k,'source':item['source'],'volume_id':item['volume_id'],'acquisition':item['acquisition'],'subset':item['subset'],
                     'z':item['z'],'window_y':item['y']+wy*64,'window_x':item['x']+wx*64,
                     'normalized_y':(item['y']+wy*64+32)/item['shape_zyx'][1],'normalized_x':(item['x']+wx*64+32)/item['shape_zyx'][2],
                     'spacing_y_nm':item['spacing_yx_nm'][0],'spacing_x_nm':item['spacing_yx_nm'][1],
                     'periodicity':float(f[0]),'spectral_entropy':float(f[1]),'contrast':float(f[2]),'gradient_energy':float(f[3]),
                     'dominant_period_px':1/float(f[4]),'context_GT_density':float(G.sum()/max(1,V.sum())),
                     'far_background_eligible':eligible,'contrast_stratum':int(np.searchsorted(cut_c,f[2])),
                     'gradient_stratum':int(f[3]>cut_g),'Y_half':int((item['y']+wy*64+32)/item['shape_zyx'][1]>=.5)}
                row['window_FOV_y_nm']=64*row['spacing_y_nm'];row['window_FOV_x_nm']=64*row['spacing_x_nm']
                window_meta.append(row);window_select.append((k,wy,wx))
    for source in protocol['diagnostic_sources']:
        csvout('image_periodicity_windows_'+source+'.csv',[r for r in window_meta if r['source']==source])
    for model in models:
        out=R/'models'/model;state=torch.load(out/'linear_probes.pt',map_location='cpu',weights_only=True)
        for li,layer in enumerate(layers):
            raw=npy(out/f'layer{layer}.npy');mean=state[layer]['mean'].cuda()
            for si,source in enumerate(protocol['diagnostic_sources']):
                ids=[i['id'] for i in items if i['source']==source and i['subset']=='random'];x=torch.tensor(np.array(raw[ids],copy=True).reshape(-1,1024),device='cuda',dtype=torch.float32)
                positive=torch.as_tensor(fg[ids].reshape(-1),device='cuda');score=torch.tensor(np.array(probs[model][ids,li],copy=True).reshape(-1),device='cuda')
                prototype=torch.tensor(np.array(protos[model][ids,li],copy=True).reshape(-1),device='cuda')
                for label,mask in [('all_far_background',far),('random_far_background',random_bg),('D2_posthoc_FP',error)]:
                    background=torch.as_tensor(mask[ids].reshape(-1),device='cuda')
                    separation_rows.append({'model':model,'layer':layer,'source':source,'background_set':label,
                        'foreground_patches':int(positive.sum()),'background_patches':int(background.sum()),
                        'linear_AUROC':auc(score[positive],score[background]),'prototype_AUROC':auc(prototype[positive],prototype[background]),
                        **separation(x[positive],x[background],mean)})
                candidates=np.flatnonzero(valid[ids].reshape(-1));n=protocol['spectrum_tokens_per_source'];assert len(candidates)>=n
                for repeat in range(protocol['spectrum_repeats']):
                    generator=np.random.default_rng(np.random.SeedSequence([protocol['seed'],si,repeat]))
                    chosen=generator.choice(candidates,n,replace=False);values=spectrum(x[torch.tensor(chosen,device='cuda')]);eigen=values.pop('normalized_eigenvalues')
                    spectral_rows.append({'model':model,'layer':layer,'source':source,'repeat':repeat,'patches':n,**values})
                    eigen_rows.append({'model':model,'layer':layer,'source':source,'repeat':repeat,'variance_fractions_descending':json.dumps(eigen.tolist(),separators=(',',':'))})
                del x;torch.cuda.empty_cache()
            # Evaluate the 1x1 readout on the same native-grid crop pixels as existing Primus masks.
            linear_pixels=np.zeros((len(items),256,256),bool);linear_probability=np.zeros((len(items),256,256),np.float32)
            for item in items:
                if item['role']!='eval':continue
                k=item['id'];z=torch.tensor(np.array(probs[model][k,li],copy=True),device='cuda').reshape(1,1,16,16)
                probability=F.interpolate(z,size=(256,256),mode='bilinear',align_corners=False)[0,0]
                linear_pixels[k]=(probability>.5).cpu().numpy();linear_probability[k]=probability.cpu().numpy()
                pixel_rows.append({'model':model,'layer':layer,'readout':'linear','sample_id':k,'source':item['source'],'volume_id':item['volume_id'],
                    'acquisition':item['acquisition'],'subset':item['subset'],'z':item['z'],
                    **metrics(linear_pixels[k],labels['GT'][k],labels['valid_pixels'][k],labels['far_pixels'][k])})
                if li==0:
                    pixel_rows.append({'model':model,'layer':'Primus','readout':'Primus','sample_id':k,'source':item['source'],'volume_id':item['volume_id'],
                        'acquisition':item['acquisition'],'subset':item['subset'],'z':item['z'],
                        **metrics(primus[model][k],labels['GT'][k],labels['valid_pixels'][k],labels['far_pixels'][k])})
            activation=[]
            for row,(k,wy,wx) in zip(window_meta,window_select):
                sl=(slice(wy*64,(wy+1)*64),slice(wx*64,(wx+1)*64))
                activation.append({**row,'model':model,'layer':layer,'linear_activation':float(linear_probability[k][sl].mean()),
                    'linear_FP_fraction':float(linear_pixels[k][sl].mean()),'Primus_FP_fraction':float(primus[model][k][sl].mean())})
            # Save detailed windows in server artifacts, not an oversized Git table.
            np.savez_compressed(out/f'window_activation_{layer}.npz',linear_activation=np.array([r['linear_activation'] for r in activation]),
                linear_FP_fraction=np.array([r['linear_FP_fraction'] for r in activation]),Primus_FP_fraction=np.array([r['Primus_FP_fraction'] for r in activation]))
            for source in protocol['diagnostic_sources']:
                rows=[r for r in activation if r['source']==source and r['subset']=='random' and r['far_background_eligible']]
                groups=[('source_all',rows)]
                for volume in sorted({r['volume_id'] for r in rows}):groups.append(('volume:'+volume,[r for r in rows if r['volume_id']==volume]))
                for key in sorted({(r['contrast_stratum'],r['gradient_stratum'],r['Y_half'],r['spacing_y_nm'],r['spacing_x_nm']) for r in rows}):
                    groups.append(('stratum:'+str(key),[r for r in rows if (r['contrast_stratum'],r['gradient_stratum'],r['Y_half'],r['spacing_y_nm'],r['spacing_x_nm'])==key]))
                for group,members in groups:
                    if len(members)<protocol['periodicity']['minimum_correlation_windows']:
                        if group=='source_all':correlations.append({'model':model,'layer':layer,'source':source,'group':group,'target':'linear_activation','windows':len(members),'spearman':None,'partial_rank_correlation':None,'status':'insufficient_far_background_windows'})
                        continue
                    xx=torch.tensor([r['periodicity'] for r in members],device='cuda')
                    cov=torch.tensor([[r[k] for k in ['contrast','gradient_energy','normalized_y','normalized_x','context_GT_density','spacing_y_nm','spacing_x_nm']] for r in members],device='cuda')
                    for target in ['linear_activation','linear_FP_fraction','Primus_FP_fraction']:
                        if target=='Primus_FP_fraction' and li!=0:continue
                        yy=torch.tensor([r[target] for r in members],device='cuda')
                        row={'model':model,'layer':'Primus' if target=='Primus_FP_fraction' else layer,'source':source,'group':group,'target':target,'windows':len(members),
                            'volumes':len({r['volume_id'] for r in members}),'acquisitions':len({r['acquisition'] for r in members}),
                            'spearman':spearman(xx,yy),'partial_rank_correlation':partial_rank(xx,yy,cov),
                            'scale_control':'within-source native spacing strata; constant terms excluded, not estimated'}
                        if group=='source_all' and model in ('D2-30k','D2-100k') and (layer==23 or target=='Primus_FP_fraction'):
                            blocks=[r['volume_id']+':'+str(r['z']//8) for r in members]
                            row.update(bootstrap_spearman(xx,yy,blocks,protocol['periodicity']['bootstrap_repeats'],protocol['seed']))
                        correlations.append(row)
            atomic(R/'analysis_progress.json',{'model':model,'layer':layer,'seconds':time.time()-started})
    paired=[]
    for si,source in enumerate(protocol['diagnostic_sources']):
        ids=[i['id'] for i in items if i['source']==source and i['subset']=='random']
        blocks=np.repeat([items[k]['volume_id']+':'+str(items[k]['z']//8) for k in ids],256).tolist()
        positive=torch.tensor(fg[ids].reshape(-1),device='cuda');background=torch.tensor(far[ids].reshape(-1),device='cuda')
        for li,layer in enumerate(layers):
            for target,mapping in [('linear',probs),('prototype',protos)]:
                a=torch.tensor(np.array(mapping['D2-30k'][ids,li],copy=True).reshape(-1),device='cuda')
                b=torch.tensor(np.array(mapping['D2-100k'][ids,li],copy=True).reshape(-1),device='cuda')
                aa,bb=auc(a[positive],a[background]),auc(b[positive],b[background])
                paired.append({'source':source,'layer':layer,'score':target,'D2_30k_AUROC':aa,'D2_100k_AUROC':bb,
                    'delta':bb-aa if aa is not None and bb is not None else None,
                    **bootstrap_auc_delta(a,b,positive,background,blocks,protocol['periodicity']['bootstrap_repeats'],protocol['seed']+li+si*100)})
    grouped=[]
    for model in models:
        csvout('pixel_metrics_'+model+'.csv',[r for r in pixel_rows if r['model']==model])
        for source in protocol['diagnostic_sources']:
            for layer in [*layers,'Primus']:
                rows=[r for r in pixel_rows if r['model']==model and r['source']==source and r['layer']==layer and r['subset']=='random']
                counts={k:sum(r[k] for r in rows) for k in ['tp','fp','fn','valid_pixels','far_background_pixels','far_FP_pixels']}
                tp,fp,fn=counts['tp'],counts['fp'],counts['fn'];den=counts['far_background_pixels']
                grouped.append({'model':model,'source':source,'layer':layer,'crops':len(rows),**counts,
                    'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,'precision':tp/(tp+fp) if tp+fp else None,
                    'recall':tp/(tp+fn) if tp+fn else None,'far_FP_rate':counts['far_FP_pixels']/den if den else None})
    csvout('separability.csv',separation_rows);csvout('spectrum.csv',spectral_rows)
    # Separate per-model spectrum distributions keep individual Git tables below2MB.
    for model in models:csvout('spectrum_distribution_'+model+'.csv',[r for r in eigen_rows if r['model']==model])
    csvout('source_readout_metrics.csv',grouped);csvout('periodicity_correlations.csv',correlations);csvout('D2_paired_AUROC.csv',paired)
    atomic(R/'analysis_complete.json',{'status':'complete','models':models,'layers':layers,'seconds':time.time()-started,
        'all_far_background_scope':'eligible tokens in locked random FOVs','error_cases_excluded_from_random_estimates':True,
        'probe_fit_inputs':'MV3_train_only','patches_are_not_independent_acquisitions':True,'new_training_in_analysis':False})


if __name__=='__main__':run()
