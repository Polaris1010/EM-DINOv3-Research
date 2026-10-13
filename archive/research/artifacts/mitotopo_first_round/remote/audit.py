import os,sys,json,csv,hashlib,time,gc,subprocess,collections,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
ROOT=Path('/opt/data/3_117/tanghr')
OUT=Path(__file__).resolve().parent

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str))

def manifest():
    import numpy as np,zarr
    from PIL import Image,ImageDraw
    cat=json.loads((ROOT/'MitoVerse_data/catalog.json').read_text())
    overlay=json.loads((ROOT/'MitoVerse_data/splits/mitoem2.0.json').read_text())
    mapping={v['id']:v for v in overlay['volumes']}
    rows=[]; issues=[]
    for vid,m in sorted(cat.items()):
        p=ROOT/'MitoVerse_data'/m['zarr']
        row=dict(dataset=m['dataset_id'],volume_id=vid,image_path=str(p/'img'),label_path=str(p/'mito'),shape_z_y_x=json.dumps(m.get('shape_zyx')),voxel_nm_z_y_x='',modality=m.get('modality'),image_dtype='',label_dtype='',num_instances=m.get('n_instances',''),foreground_ratio='',split=mapping.get(vid,{}).get('split','unassigned'),notes='num_instances: catalog, not recounted; voxel order unverified; foreground ratio not scanned',voxel_nm_raw=json.dumps(m.get('voxel_nm')),chunks='',compressor='',specimen_group=m['dataset_id'],exploratory_split='train' if m['dataset_id']=='jiang25' and mapping.get(vid,{}).get('split')=='train' else 'val' if m['dataset_id']=='conrad23' else 'excluded',exists=p.exists())
        try:
            a=json.loads((p/'img/.zarray').read_text()); b=json.loads((p/'mito/.zarray').read_text())
            row.update(image_dtype=a['dtype'],label_dtype=b['dtype'],chunks=json.dumps(a['chunks']),compressor=json.dumps(a['compressor']))
            if a['shape']!=b['shape'] or a['shape']!=m['shape_zyx']: issues.append([vid,'shape mismatch'])
            if np.prod(a['chunks'])*np.dtype(a['dtype']).itemsize+np.prod(b['chunks'])*np.dtype(b['dtype']).itemsize>256*1024**2: issues.append([vid,'chunk exceeds 256MiB combined'])
        except Exception as e: row['notes']+='; metadata error: '+str(e)
        rows.append(row)
    with (OUT/'dataset_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Conservative domain-level grouping is stronger than speculative specimen IDs.
    split={'name':'exploratory_domain_disjoint_seed20260907','official':False,'seed':20260907,'grouping':'all volumes of each source dataset remain together; cross-domain validation only','train':[r['volume_id'] for r in rows if r['exploratory_split']=='train' and r['exists']],'val':[r['volume_id'] for r in rows if r['exploratory_split']=='val' and r['exists']],'test':[],'test_policy':'no test data used for parameter selection','limitations':['Does not establish absence of overlap with historical EM pretraining','voxel size order unresolved; no physical scale transforms']}
    assert not set(split['train'])&set(split['val'])
    save('exploratory_split.json',split)
    samples=[]; (OUT/'previews').mkdir(exist_ok=True)
    rng=np.random.default_rng(20260907)
    # Seeded domain-stratified sampling: two training-domain volumes and three held-out-domain volumes.
    chosen=list(rng.choice(split['train'],min(2,len(split['train'])),replace=False))+list(rng.choice(split['val'],min(3,len(split['val'])),replace=False))
    for vid in chosen:
        m=cat[vid]; p=ROOT/'MitoVerse_data'/m['zarr']; g=zarr.open_group(str(p),mode='r')
        im,lab=g['img'],g['mito']; panels=[]
        for j in range(3):
            z=int(rng.integers(im.shape[0])); h=min(192,im.shape[1]);w=min(192,im.shape[2]); y=int(rng.integers(im.shape[1]-h+1));x=int(rng.integers(im.shape[2]-w+1))
            t=time.monotonic();a=np.asarray(im[z,y:y+h,x:x+w]);b=np.asarray(lab[z,y:y+h,x:x+w]);elapsed=time.monotonic()-t
            ids=np.unique(b); rgb=np.stack([(b.astype('uint64')*53)%251,(b.astype('uint64')*97)%253,(b.astype('uint64')*193)%255],-1).astype('uint8');rgb[b==0]=0
            gray=Image.fromarray(a).convert('RGB'); mask=Image.fromarray(((b>0)*255).astype('uint8')).convert('RGB'); inst=Image.fromarray(rgb)
            canvas=Image.new('RGB',(w*3,h+24),'white');canvas.paste(gray,(0,24));canvas.paste(mask,(w,24));canvas.paste(inst,(w*2,24));ImageDraw.Draw(canvas).text((3,3),f'{vid} z={z} y={y} x={x}',fill='black');panels.append(canvas)
            samples.append(dict(volume_id=vid,z=z,y=y,x=x,shape=list(a.shape),image_dtype=str(a.dtype),label_dtype=str(b.dtype),image_min=int(a.min()),image_max=int(a.max()),sample_foreground_ratio=float((b>0).mean()),sample_instances=int((ids>0).sum()),contains_background=bool(0 in ids),read_seconds=elapsed))
        sheet=Image.new('RGB',(max(p.width for p in panels),sum(p.height for p in panels)),'white');off=0
        for panel in panels: sheet.paste(panel,(0,off));off+=panel.height
        sheet.save(OUT/'previews'/f'{vid}.png')
    save('sample_audit.json',samples)
    counts=collections.Counter(v['split'] for v in overlay['volumes'])
    save('dataset_audit.json',dict(catalog_volumes=len(rows),present=sum(r['exists'] for r in rows),sources=dict(collections.Counter(r['dataset'] for r in rows)),overlay_counts=dict(counts),issues=issues,split=split,limitations=['No whole-volume ID-continuity certification from sparse samples','No full-volume foreground statistics','Local overlay is not a validated official split','Catalog voxel_nm has no explicit order; anisotropic examples suggest XYZ despite array ZYX']))
    print('DATASET',len(rows),'volumes;',len(samples),'patches',flush=True)

def weights():
    import torch,yaml
    torch.set_num_threads(2)
    selected=[ROOT/'DINOv3/med_checkpoints_continual_v2/eval/training_124999/teacher_checkpoint.pth',ROOT/'DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',ROOT/'Med_DINOv3/nnUNet_results/Dataset_13799_CMCT_upgrade2_v4/meddinov3_base_primus_multiscale_Trainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth']
    paths=[]
    for base in [ROOT/'DINOv3/pretrained_weights',*sorted((ROOT/'DINOv3').glob('med_checkpoints*')),ROOT/'Med_DINOv3/nnUNet_results']:
        for d,dirs,files in os.walk(base):
            for fn in files:
                if fn.endswith(('.pth','.pt','.distcp')): paths.append(Path(d)/fn)
    rows=[]
    for p in sorted(paths):
        r=dict(path=str(p),bytes=p.stat().st_size,sha256='',top_keys=[],state_keys_count=None,classification='uninspected',gram_use_loss=None)
        cfg=next((q/'config.yaml' for q in p.parents if (q/'config.yaml').is_file()),None)
        if cfg:
            c=yaml.safe_load(cfg.read_text());r['config']=str(cfg);r['gram_use_loss']=c.get('gram',{}).get('use_loss');r['architecture']=c.get('student',{}).get('arch')
        try:
            if p.suffix=='.distcp':
                r['classification']='distributed shard; requires metadata/loader, not independently restored'
            else:
                c=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
                r['top_keys']=list(c)[:30]
                if 'teacher' in c:
                    sd=c['teacher'];r['classification']='EM teacher export (config-associated)' if cfg else 'teacher export; provenance unresolved'
                elif 'network_weights' in c:
                    sd=c['network_weights'];r['classification']='downstream segmentation';r['epoch']=c.get('current_epoch');r['trainer']=c.get('trainer_name');r['best_ema']=c.get('_best_ema');r['has_optimizer']=c.get('optimizer_state') is not None
                else: sd=c;r['classification']='encoder state dictionary; natural initialization candidate' if p in selected else 'state dictionary; provenance unresolved'
                r['state_keys_count']=len(sd);r['state_key_examples']=list(sd)[:8]
                r['has_storage_tokens']=any('storage_tokens' in k for k in sd)
                r['has_gates']=any('last_gates' in k for k in sd)
                if p in selected:
                    save('keys_'+('em' if 'teacher' in c else 'downstream' if 'network_weights' in c else 'natural')+'.json',{k:list(v.shape) if hasattr(v,'shape') else str(type(v)) for k,v in sd.items()})
                del c,sd;gc.collect()
            if p in selected:
                h=hashlib.sha256()
                with p.open('rb') as f:
                    for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
                r['sha256']=h.hexdigest()
        except Exception as e: r['error']=repr(e)
        rows.append(r)
        if len(rows)%20==0: print('INSPECTED',len(rows),flush=True)
    save('checkpoint_inventory.json',rows)
    with (OUT/'checkpoint_inventory.csv').open('w',newline='') as f:
        keys=['path','bytes','classification','sha256','gram_use_loss','state_keys_count','has_gates','has_optimizer','epoch','error'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
    print('CHECKPOINTS',len(rows),flush=True)

if __name__=='__main__':
    t=time.monotonic()
    if sys.argv[1]=='data': manifest()
    if sys.argv[1]=='weights': weights()
    save('audit_resource_'+sys.argv[1]+'.json',dict(cpu_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,elapsed_seconds=time.monotonic()-t,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']))

