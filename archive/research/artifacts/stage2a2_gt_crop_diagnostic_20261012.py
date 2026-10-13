"""Development-only GT diagnostic for mitochondrial crop content and truncation."""
import os,sys,csv,json,hashlib,time,collections
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from PIL import Image,ImageDraw
sys.path.insert(0,'/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps')
import zarr
RUN=Path(__file__).resolve().parents[1]
OUT=RUN/'branch_S/gt_mito_diagnostic_20261012'
REF=json.loads((RUN/'references/paths.json').read_text());STAGE2=Path(REF['stage2_root'])

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(p,rows):
    if not rows:return
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x
def save(p,x):p.write_text(json.dumps(clean(x),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def cross_edges(lab,y0,x0,y1,x1):
    sides=[]
    if y0>0:sides.append((lab[y0-1,x0:x1],lab[y0,x0:x1]))
    if y1<lab.shape[0]:sides.append((lab[y1-1,x0:x1],lab[y1,x0:x1]))
    if x0>0:sides.append((lab[y0:y1,x0-1],lab[y0:y1,x0]))
    if x1<lab.shape[1]:sides.append((lab[y0:y1,x1-1],lab[y0:y1,x1]))
    ids=[];fgcount=0;count=0;cross=0
    for a,b in sides:
        count+=a.size;fgcount+=int(np.count_nonzero((a>0)|(b>0)))
        m=(a>0)&(a==b);cross+=int(m.sum());ids.extend(a[m].tolist())
    return cross,len(set(ids)),count,fgcount,len(sides)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'gt_panels_first20').mkdir(exist_ok=True)
    samples=json.loads((RUN/'references/sample_manifest_424.json').read_text())['samples']
    with (STAGE2/'topology_calibration/crop_score/crop_score_pilot_424x4.csv').open() as f:
        geo={(int(r['sample_index']),int(r['candidate'])):{k:int(r[k]) for k in ['x','y','w','h']} for r in csv.DictReader(f)}
    with (RUN/'branch_S/graph_cut/same_shape_scores_424.csv').open() as f:scores=list(csv.DictReader(f))
    bysample=collections.defaultdict(list)
    for row in scores:bysample[int(row['sample_index'])].append(row)
    with (RUN/'branch_S/blind_review/answer_key_CONCEALED.csv').open() as f:
        # Keep the existing A/B geometry. No unreviewed algorithm winner is exported.
        keys=[{k:r[k] for k in ['pair_id','sample_index','source','A_candidate','B_candidate']} for r in csv.DictReader(f) if int(r['pair_id'][1:])<=20]
    human={}
    with (RUN/'branch_S/blind_review_raw_20261011/human_review_P001_P020_overlay_20261011.csv').open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):human[row['pair_id']]=row.get('human_choice_A_B_unclear',row.get('human_choice',''))
    rows=[];sampleaudit=[];pairrows=[];cache={};start=time.time()
    for i,s in enumerate(samples):
        g=zarr.open_group(s['local_path'],mode='r');assert g['img'].shape==g['mito'].shape
        h,w=map(int,s['image_shape']);side=min(h,w,512);oy,ox=(h-side)//2,(w-side)//2;z=int(s['slice_index'])
        sl=(z,slice(oy,oy+side),slice(ox,ox+side));lab=np.asarray(g['mito'][sl]);assert lab.shape==(side,side)
        label_type=dict(g.attrs).get('mitoverse',{}).get('label_type','unknown')
        has_mask='mask' in g
        # Mask-bearing slices are inventoried but excluded from ranking until the annotation policy is applied.
        valid_status='annotation_mask_present_excluded' if has_mask else 'no_region_mask_available'
        totalids,totalcounts=np.unique(lab,return_counts=True);totals=dict(zip(map(int,totalids),map(int,totalcounts)))
        frame_ids=set(map(int,np.unique(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]]))))-{0}
        foreground=int(np.count_nonzero(lab));positive_ids=set(totals)-{0}
        sampleaudit.append({'sample_index':i,'volume_id':s['volume_id'],'source':s['source'],'slice_index':z,'context_y':oy,'context_x':ox,'context_side':side,'label_type':label_type,'has_annotation_mask':has_mask,'GT_context_foreground_pixels':foreground,'GT_context_positive_label_ids':len(positive_ids),'GT_context_edge_label_ids':len(frame_ids),'label_context_sha256':hashlib.sha256(lab.tobytes()).hexdigest()})
        for scored in bysample[i]:
            c=int(scored['candidate']);box=geo[i,c];x0=int(np.floor(box['x']*side/512));y0=int(np.floor(box['y']*side/512));x1=int(np.ceil((box['x']+box['w'])*side/512));y1=int(np.ceil((box['y']+box['h'])*side/512))
            assert 0<=x0<x1<=side and 0<=y0<y1<=side
            a=lab[y0:y1,x0:x1];ids,ns=np.unique(a,return_counts=True);inside={int(k):int(v) for k,v in zip(ids,ns) if k>0}
            cut_ids=[k for k,v in inside.items() if v<totals[k]]
            complete=[k for k,v in inside.items() if v==totals[k] and k not in frame_ids]
            cross,cutinstances,perim,fgperim,n_sides=cross_edges(lab,y0,x0,y1,x1)
            r={'sample_index':i,'volume_id':s['volume_id'],'source':s['source'],'candidate':c,'slice_index':z,'label_type':label_type,'annotation_status':valid_status,'x_display':box['x'],'y_display':box['y'],'w_display':box['w'],'h_display':box['h'],'native_x0':x0,'native_y0':y0,'native_x1':x1,'native_y1':y1,'has_labelled_mito':bool(inside),'mito_pixels':int(np.count_nonzero(a)),'mito_fraction':float(np.count_nonzero(a)/a.size),'intersecting_label_ids':len(inside),'known_complete_2D_section_ids':len(complete),'partially_retained_context_ids':len(cut_ids),'truncated_fraction':len(cut_ids)/len(inside) if inside else None,'mito_cross_boundary_same_id_edges':cross,'mito_cut_ids_on_boundary':cutinstances,'boundary_edges':perim,'mito_cross_edge_fraction':cross/perim if perim else None,'observable_boundary_sides':n_sides,'ranking_eligible':not has_mask and bool(inside) and n_sides==4,'cut_mean':float(scored['cut_mean']),'persist_exclude_largest':float(scored['persist_exclude_largest'])}
            rows.append(r)
        if any(int(k['sample_index'])==i for k in keys):
            im=np.asarray(g['img'][sl],dtype=np.float32);lo,hi=np.percentile(im,[1,99]);gray=(np.clip((im-lo)/max(hi-lo,1e-6),0,1)*255).astype(np.uint8)
            cache[i]=(gray,(lab>0).astype(np.uint8)*255)
        if (i+1)%50==0:print('GT_SLICES',i+1,424,'seconds',round(time.time()-start,1),flush=True)
    index={(r['sample_index'],r['candidate']):r for r in rows};correlations=[]
    for i in range(len(samples)):
        candidates=[index[i,c] for c in range(4) if index[i,c]['ranking_eligible']]
        if len(candidates)<3:continue
        for metric in ['mito_cross_boundary_same_id_edges','mito_cut_ids_on_boundary','truncated_fraction']:
            gt=[r[metric] for r in candidates];pred=[r['cut_mean'] for r in candidates]
            if len(set(gt))>1 and len(set(pred))>1:
                correlations.append({'sample_index':i,'source':samples[i]['source'],'metric':metric,'spearman':float(spearmanr(pred,gt).statistic),'candidates':len(candidates)})
    for k in keys:
        i=int(k['sample_index']);a=index[i,int(k['A_candidate'])];b=index[i,int(k['B_candidate'])]
        # This new mitochondrial diagnostic is independent of the old generic-structure human question.
        if not a['ranking_eligible'] or not b['ranking_eligible']:gtchoice='not_eligible'
        elif a['mito_cross_boundary_same_id_edges']==b['mito_cross_boundary_same_id_edges']:gtchoice='tie'
        else:gtchoice='A' if a['mito_cross_boundary_same_id_edges']<b['mito_cross_boundary_same_id_edges'] else 'B'
        pairrows.append({'pair_id':k['pair_id'],'source':k['source'],'sample_index':i,'old_human_generic_structure_choice':human.get(k['pair_id'],''),'GT_mito_cross_edges_choice_development_only':gtchoice,'A_has_labelled_mito':a['has_labelled_mito'],'B_has_labelled_mito':b['has_labelled_mito'],'A_mito_pixels':a['mito_pixels'],'B_mito_pixels':b['mito_pixels'],'A_known_complete_2D_section_ids':a['known_complete_2D_section_ids'],'B_known_complete_2D_section_ids':b['known_complete_2D_section_ids'],'A_cross_mito_edges':a['mito_cross_boundary_same_id_edges'],'B_cross_mito_edges':b['mito_cross_boundary_same_id_edges'],'A_cut_mito_ids':a['mito_cut_ids_on_boundary'],'B_cut_mito_ids':b['mito_cut_ids_on_boundary'],'annotation_status':a['annotation_status']})
        gray,mask=cache[i];canvas=Image.new('RGB',(1048,1100),(24,24,24));draw=ImageDraw.Draw(canvas)
        for letter,offset in [('A',0),('B',524)]:
            r=a if letter=='A' else b;box=(r['x_display'],r['y_display'],r['x_display']+r['w_display'],r['y_display']+r['h_display'])
            for rownum,arr,label in [(0,gray,'EM'),(1,mask,'GT mito')]:
                img=Image.fromarray(arr).resize((512,512),Image.Resampling.NEAREST if rownum else Image.Resampling.BICUBIC).convert('RGB')
                ImageDraw.Draw(img).rectangle(box,outline=(255,220,0),width=2)
                y=rownum*550;draw.text((offset+10,y+10),k['pair_id']+' '+letter+' '+label,fill='white');canvas.paste(img,(offset+10,y+30))
        canvas.save(OUT/'gt_panels_first20'/(k['pair_id']+'.png'),optimize=True)
    table(OUT/'GT_crop_metrics_424x4.csv',rows);table(OUT/'GT_pair_diagnostic_first20.csv',pairrows);table(OUT/'GT_score_correlations.csv',correlations);save(OUT/'GT_sample_audit.json',sampleaudit)
    summary={'status':'complete_development_diagnostic','objective':'generalist EM mitochondrial segmentation; eventual head-to-head comparison with official MitoNet','samples':424,'candidates':len(rows),'sources':13,'samples_with_annotation_mask':sum(a['has_annotation_mask'] for a in sampleaudit),'samples_with_GT_mito_in_context':sum(a['GT_context_foreground_pixels']>0 for a in sampleaudit),'candidates_with_GT_mito':sum(r['has_labelled_mito'] for r in rows),'eligible_candidates':sum(r['ranking_eligible'] for r in rows),'eligible_metric_correlations':{},'first20_GT_cross_edge_choices':dict(collections.Counter(r['GT_mito_cross_edges_choice_development_only'] for r in pairrows)),'GT_used_for':'offline diagnosis only; no SSL training or crop sampler changed','old_human_audit':'preserved as generic visible-structure audit; not reinterpreted as mitochondrial GT accuracy','scope_limits':['label presence is evidence of labelled mitochondria; no-mask absence does not prove exhaustive annotation','known completeness means 2D section within observed context, not complete 3D organelle','objects touching context border are excluded from known-complete count','annotation-mask samples excluded from ranking until valid-region policy is applied','current samples are development data; no formal benchmark PASS or held-out test claim'],'source_sha256':sha(__file__),'sample_manifest_sha256':sha(RUN/'references/sample_manifest_424.json'),'seconds':time.time()-start,'long_training':'prohibited','P':'paused'}
    for metric in ['mito_cross_boundary_same_id_edges','mito_cut_ids_on_boundary','truncated_fraction']:
        rr=[r for r in correlations if r['metric']==metric]
        summary['eligible_metric_correlations'][metric]={'images':len(rr),'median_within_image_spearman':float(np.median([r['spearman'] for r in rr])) if rr else None,'mean_within_image_spearman':float(np.mean([r['spearman'] for r in rr])) if rr else None}
    save(OUT/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
