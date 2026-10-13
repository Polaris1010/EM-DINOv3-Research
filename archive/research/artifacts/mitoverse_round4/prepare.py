from common import *
import csv,collections,time
FIRST=ROOT/'codex/mitotopo_20260908_first_round';R3=ROOT/'codex/mitotopo_20260909_round3'
def main():
 (BASE/'jobs').mkdir(exist_ok=True)
 rows=list(csv.DictReader(open(FIRST/'dataset_manifest.csv')))
 assignments={'han24':'train','muller21':'train','kunduri22':'train','turner22':'train','guay21':'validation','mekuc20':'validation','wei20':'test1','jiang25':'test1','casser20':'test1','xiao18':'test2','haberl18':'test2','conrad23':'test3','openorganelle':'excluded'}
 split={'version':1,'policy':'Entire catalog source kept together; conservative specimen grouping; no random patch splitting','source_assignments':assignments,'volumes':{},'selection_history':'P2 chosen on CMCC/CMCT; all OpenOrganelle excluded from this round to avoid selection-data overlap','strict_unseen_proven':False}
 cachep=ROOT/'DINOv3/pretrain_data/patches_512_format/train/fast_cache.pth'
 cache=torch.load(cachep,map_location='cpu',weights_only=False)
 counts=collections.Counter();examples={}
 for p,t in cache['samples']:
  rel=p.split('/train/',1)[1];parts=rel.split('/');key='/'.join(parts[:2]);counts[key]+=1;examples.setdefault(key,p)
 save(BASE/'pretraining_cache_evidence.json',{'path':cachep,'sha256':sha(cachep),'sample_count':len(cache['samples']),'counts':counts,'examples':examples,'limitation':'Current cache agrees with EM config, but no immutable historical cache hash or per-step sampled-image log; positive source match is evidence of corpus exposure, exact volume exposure may remain uncertain.'})
 exposure=[]
 for r in rows:
  vid=r['volume_id'];source=r['dataset'];sp=assignments[source]
  p1='uncertain';level='unresolved';evidence=str(cachep)
  if source=='wei20':p1='yes';level='source/specimen present in configured EM corpus'
  elif vid=='casser20_lucchi++':p1='yes';level='Lucchi source present; corrected annotation/raw grid identity not established'
  elif source=='openorganelle':level='domain present; exact ROI identity unresolved'
  else:level='not found by source name in current cache; historical lineage not exhaustive'
  extra='yes' if vid=='jiang25_vol1' else 'uncertain' if source=='jiang25' else 'no'
  overall='yes' if p1=='yes' or extra=='yes' else 'uncertain'
  rr=dict(r,split=sp,source=source,ssl_pretrain_seen=overall,p1_em_seen=p1,p2_extra_seen=extra)
  split['volumes'][vid]=rr
  exposure.append(dict(volume_id=vid,source=source,specimen_group=source,split=sp,ssl_pretrain_seen=overall,p1_em_seen=p1,p2_extra_seen=extra,exposure_level=level,evidence=evidence+'; '+str(FIRST/'em_v2_config.yaml')+'; '+str(R3/'jobs/ssl_b2_w0_lr1_s2000/command.txt'),natural_pretrain_overlap='uncertain'))
 with (BASE/'pretraining_exposure.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(exposure[0]));w.writeheader();w.writerows(exposure)
 for s in ['train','validation','test1','test2','test3','excluded']:split[s]=[v for v,r in split['volumes'].items() if r['split']==s]
 for a in assignments:
  assert len({r['split'] for r in split['volumes'].values() if r['source']==a})==1
 save(BASE/'split_mitoverse_probe_v1.json',split)
 (BASE/'split_audit.md').write_text('# MitoVerse probe split v1\n\n'+ '\n'.join(f'- {k}: {len(split[k])} volumes; '+', '.join(s for s,v in assignments.items() if v==k) for k in ['train','validation','test1','test2','test3','excluded'])+'\n\nWhole source grouping is conservative, not a claim of verified individual-animal identities across publications. OpenOrganelle is excluded because P2 selection used CMCC/CMCT. Test1 includes documented SSL-source exposure (EM MitoEM/Lucchi and P2 Jiang vol1), with per-volume uncertainty retained. Test2 is anisotropic/cross-acquisition challenge. Test3 is an unseen-candidate source, NOT proven SSL-unseen. Strict unseen summary must be unavailable if no rows qualify. No test labels or predictions used for protocol selection. Training source sampling is equal, then volumes equal.\n')
 registry=[]
 for pid,p in PATHS.items():registry.append(dict(id=pid,path=str(p),sha256=sha(p),bytes=p.stat().st_size,selection='locked before MitoVerse validation; P2 is b2 lambda0 old LR 2000 steps' if pid=='P2' else 'pre-specified baseline'))
 with (BASE/'checkpoint_registry.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(registry[0]));w.writeheader();w.writerows(registry)
 # Explicit conversion, folding the masked bias into ordinary qkv bias.
 original=encoder_sd(PATHS['P0']);expected=encoder().state_dict();converted={};removed=[];folded=[]
 for k,v in original.items():
  if k.endswith('qkv.bias_mask'):removed.append(k);continue
  if k=='storage_tokens' and k not in expected:removed.append(k);continue
  if k.endswith('qkv.bias') and k+'_mask' in original:v=v*original[k+'_mask'];folded.append(k)
  converted[k]=v
 missing=sorted(set(expected)-set(converted));unexpected=sorted(set(converted)-set(expected));mismatch={k:[list(expected[k].shape),list(converted[k].shape)] for k in set(expected)&set(converted) if expected[k].shape!=converted[k].shape}
 for n,x in [('missing_keys.json',missing),('unexpected_keys.json',unexpected),('shape_mismatch.json',mismatch)]:save(BASE/n,x)
 assert not missing and not unexpected and not mismatch
 torch.save(converted,BASE/'natural_converted.pth')
 seed(17);inp=torch.rand(1,3,64,64);orig=encoder(original,True).eval();conv=encoder(converted).eval()
 with torch.inference_mode():
  a=orig.get_intermediate_layers(inp,n=[5,11,17,23],reshape=True);b=conv.get_intermediate_layers(inp,n=[5,11,17,23],reshape=True)
  changes=[{'relative_l2':float((x-y).norm()/x.norm()),'cosine':float(torch.nn.functional.cosine_similarity(x.flatten(),y.flatten(),dim=0))} for x,y in zip(a,b)]
 save(BASE/'natural_to_em_arch_conversion.json',{'removed':removed,'qkv_bias_folded':folded,'strict_load':True,'source_sha256':registry[0]['sha256'],'converted_sha256':sha(BASE/'natural_converted.pth'),'cpu_fixed_input_64_feature_changes':changes,'interpretation':'Removing storage tokens changes attention context; P0 is adapted natural weights, P1-P0 is not a pure architecture-independent pretraining effect.'})
 del orig,conv,a,b,original,converted;gc.collect()
 for s in range(3):
  seed(s);net=Primus_Multiscale(embed_dim=1024,patch_embed_size=16,num_classes=2,dino_encoder=encoder(),interaction_indices=[5,11,17,23]);torch.save(head(net),BASE/f'shared_head_init_seed{s}.pth');del net
 save(BASE/'protocol_lock.json',{'encoders':registry,'seeds':[0,1,2],'batch':8,'crop':256,'max_steps':10000,'validation_interval':1000,'checkpoint_interval':1000,'head_lr':1e-4,'weight_decay':.05,'betas':[.9,.98],'warmup_steps':50,'lr_min_ratio':.1,'clip':12,'loss':'CE+Dice (valid pixels, batch aggregate)','encoder_frozen':True,'feature_amp':'bf16','head_dtype':'float32','norm_candidates':['N1','N2'],'norm_selection':'500-step pilot each encoder x norm, seed0 shared schedule/init; equal mean over encoders and validation sources; tie choose N1','selection_validation':'full XY sliding on at most 8 evenly spaced slices per val volume; explicitly subsampled for head selection only','final_validation_test':'all original voxels; tile256 step0.5 gaussian; no resampling; probability float16 and binary uint8 zarr','tta':False,'threshold':.5,'tta_threshold_policy':'fixed a priori, no test tuning','decision':'source-equal macro Dice primary. P1 benefit requires all 3 seed deltas positive, >half held-out sources positive, no precision/recall collapse; no unseen claim if strict unseen absent. P2 additionally mean paired gain > paired sample SD, else retain P1. Exploratory descriptive rule, not significance test.','split_sha256':sha(BASE/'split_mitoverse_probe_v1.json')})
 save(BASE/'prepare_complete.json',{'status':'complete','counts':{s:len(split[s]) for s in ['train','validation','test1','test2','test3']}})
 print('PREPARE_COMPLETE',flush=True)
if __name__=='__main__':main()
