"""Pixel-coincident scoring; evaluation GT never selects parameters."""
import argparse,time
import numpy as np
import torch
from common import R,P,MODELS,SOURCES,read,sha,atomic,csvout,samples,native,setup,counts,summarize,Ranked

def aggregate(rows):
    keys=['tp','fp','fn','valid_pixels','foreground_pixels','predicted_pixels','far_background_pixels','far_FP_pixels','brier_sum']
    return {k:sum(r[k] for r in rows) for k in keys}

def stage(final):
    setup();start=time.time();items=samples();labels=np.load(P/'evaluation_labels.npz')
    if final:
        pinned=read(R/'train_only_bias.json');assert pinned['status']=='PINNED';pinhash=sha(R/'train_only_bias.json')
    pixels=[];pooled=[];ranks=[];curves=[];quantiles=[];paired=[]
    for source in SOURCES:
        selected=[i for i in items if i['source']==source and i['subset']=='random'];ids=[i['id'] for i in selected];assert len(ids)==64
        g=torch.tensor(labels['GT'][ids],device='cuda');v=torch.tensor(labels['valid_pixels'][ids],device='cuda');far=torch.tensor(labels['far_pixels'][ids],device='cuda')
        blocks=[i['volume_id']+':'+str(i['z']//8) for i in selected];unique=sorted(set(blocks));nb=len(unique)
        cropblock=torch.tensor([unique.index(x) for x in blocks],device='cuda');pixelblock=cropblock[:,None,None].expand_as(g)
        rng=np.random.default_rng(20261027+SOURCES.index(source));draws=np.stack([np.bincount(rng.integers(nb,size=nb),minlength=nb) for _ in range(200)])
        draw=torch.tensor(draws,device='cuda',dtype=torch.int64);boot_scores={};boot_counts={}
        for model in MODELS:
            p=native(model,ids);offsets={'raw':0.}
            if final:offsets['calibrated']=pinned['models'][model]['bias']
            for name,b in offsets.items():
                these=[]
                for k,item in enumerate(selected):
                    c=counts(p[k],g[k],v[k],far[k],b)
                    row={'model':model,'source':source,'operation':name,'sample_id':item['id'],'volume_id':item['volume_id'],'z':item['z'],'block':blocks[k],'bias':b,**c,**summarize(c)}
                    pixels.append(row);these.append(row)
                c=aggregate(these);pooled.append({'model':model,'source':source,'operation':name,'crops':64,'bias':b,**c,**summarize(c)})
                if final:
                    keys=['tp','fp','fn','far_background_pixels','far_FP_pixels']
                    perblock=torch.tensor([[sum(r[k] for r in these if r['block']==key) for k in keys] for key in unique],device='cuda',dtype=torch.float64)
                    bc=draw.double()@perblock;tp,fp,fn,nfar,ffp=bc.T
                    boot_counts[model,name]={'dice':2*tp/(2*tp+fp+fn),'precision':tp/(tp+fp),'recall':tp/(tp+fn),'far_FP_rate':ffp/nfar}
            for group,mask in [('all_valid',v),('FG_vs_far_background',v&(g|far))]:
                ranked=Ranked(p[mask],g[mask],pixelblock[mask]);auroc,ap,curve=ranked.calculate(curve=True)
                ranks.append({'model':model,'source':source,'scope':group,'pixels':int(mask.sum()),'positives':int((g&mask).sum()),'AUROC':auroc,'AP':ap,'ranking':'original native probability; exact tie-aware'})
                if not final:curves.extend({'model':model,'source':source,'scope':group,'recall':r,'precision':pr,'FPR':f} for r,pr,f in curve)
                if final:
                    result=[]
                    for k in range(200):
                        a,b,_=ranked.calculate(draw[k]);result.append([a if a is not None else np.nan,b if b is not None else np.nan])
                        if k%25==0:atomic(R/'progress.json',{'phase':'final_bootstrap','source':source,'model':model,'scope':group,'repeat':k,'seconds':time.time()-start})
                    boot_scores[model,group]=torch.tensor(result,device='cuda',dtype=torch.float64)
                del ranked;torch.cuda.empty_cache()
            if not final:
                for name,mask in [('GT_foreground',g&v),('all_valid_background',~g&v),('far_background',far)]:
                    q=torch.quantile(p[mask].double(),torch.tensor([.5,.9,.99],device='cuda',dtype=torch.float64)).cpu().tolist()
                    quantiles.append({'model':model,'source':source,'class':name,'pixels':int(mask.sum()),'Q50':q[0],'Q90':q[1],'Q99':q[2]})
            del p
        if final:
            arrays={}
            for group in ['all_valid','FG_vs_far_background']:
                d=boot_scores['D2-100k',group]-boot_scores['D2-30k',group];arrays[group+'_AUROC_delta']=d[:,0];arrays[group+'_AP_delta']=d[:,1]
            for op in ['raw','calibrated']:
                for key in ['dice','precision','recall','far_FP_rate']:
                    arrays['100k_minus_30k_'+op+'_'+key]=boot_counts['D2-100k',op][key]-boot_counts['D2-30k',op][key]
            for key in ['dice','precision','recall','far_FP_rate']:
                arrays['100k_cal_minus_raw_'+key]=boot_counts['D2-100k','calibrated'][key]-boot_counts['D2-100k','raw'][key]
            arrays['Dice_gap_recovery_R']=arrays['100k_minus_30k_calibrated_dice']-arrays['100k_minus_30k_raw_dice']
            source_pool=[x for x in pooled if x['source']==source]
            def m(model,op,key):return next(r[key] for r in source_pool if r['model']==model and r['operation']==op)
            actual={}
            for group in ['all_valid','FG_vs_far_background']:
                for key in ['AUROC','AP']:
                    a=[r for r in ranks if r['source']==source and r['scope']==group]
                    actual[group+'_'+key+'_delta']=next(r[key] for r in a if r['model']=='D2-100k')-next(r[key] for r in a if r['model']=='D2-30k')
            for op in ['raw','calibrated']:
                for key in ['dice','precision','recall','far_FP_rate']:actual['100k_minus_30k_'+op+'_'+key]=m('D2-100k',op,key)-m('D2-30k',op,key)
            for key in ['dice','precision','recall','far_FP_rate']:actual['100k_cal_minus_raw_'+key]=m('D2-100k','calibrated',key)-m('D2-100k','raw',key)
            actual['Dice_gap_recovery_R']=actual['100k_minus_30k_calibrated_dice']-actual['100k_minus_30k_raw_dice']
            for key,t in arrays.items():
                vals=t[torch.isfinite(t)];q=torch.quantile(vals,torch.tensor([.025,.975],device='cuda',dtype=torch.float64)).cpu().tolist()
                paired.append({'source':source,'contrast':key,'estimate':actual[key],'bootstrap_low':q[0],'bootstrap_high':q[1],'Z_blocks':nb,'bootstrap_valid_repeats':len(vals),'bias_refit_in_bootstrap':False})
            cpu={k:x.cpu().tolist() for k,x in arrays.items()};csvout('bootstrap_draws_'+source+'.csv',[{'repeat':j,**{k:x[j] for k,x in cpu.items()}} for j in range(200)])
        atomic(R/'progress.json',{'phase':'final_score' if final else 'raw_baseline','source':source,'seconds':time.time()-start})
        del g,v,far;torch.cuda.empty_cache()
    prefix='final' if final else 'baseline';csvout(prefix+'_pixel_metrics.csv',pixels);csvout(prefix+'_source_metrics.csv',pooled);csvout(prefix+'_pixel_discrimination.csv',ranks)
    if not final:csvout('PR_curves.csv',curves);csvout('class_score_quantiles.csv',quantiles)
    else:
        csvout('paired_changes.csv',paired);assert pinhash==sha(R/'train_only_bias.json')
        import csv
        with (R/'tables/baseline_source_metrics.csv').open() as f:baseline=list(csv.DictReader(f))
        for row in pooled:
            if row['operation']!='raw':continue
            old=next(x for x in baseline if x['source']==row['source'] and x['model']==row['model'])
            for k in ['tp','fp','fn','valid_pixels','far_background_pixels','far_FP_pixels']:assert int(old[k])==row[k]
    atomic(R/(prefix+'_complete.json'),{'status':'complete','seconds':time.time()-start,'evaluation_GT_use':'scoring_only','bias_sha256':pinhash if final else None,
      'models':MODELS,'sources':SOURCES,'random_crops':320,'historical_error_case_excluded':True,'network_updates':0,'encoder_forward_calls':0})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['baseline','final']);a=p.parse_args();stage(a.mode=='final')
