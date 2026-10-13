"""Read-only asset audit and bounded real-weight smoke; no SSL entrypoint."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '3', 'Physical GPU3 only'
import sys, json, hashlib, gc, time, importlib.util, subprocess, urllib.request
from pathlib import Path
import numpy as np
import torch
import diagnosis as d
import server_factory as factory

R=Path(__file__).resolve().parents[1]
OLD=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120')
EXT=Path('/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012')
S2=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002')
sys.path[:0]=['/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps']
torch.set_num_threads(4)

def save(name,obj):
    p=R/'smoke'/name
    p.write_text(json.dumps(obj,indent=2),encoding='utf-8')

def guard(initial=False):
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    for line in rows.splitlines():
        parts=[s.strip() for s in line.split(',')]
        if parts[0]=='GPU-5a68cc29-a13a-bf90-858e-079e14de807a':
            if int(parts[1])!=os.getpid(): raise RuntimeError('GPU3 has another process; stop this smoke only')
    with urllib.request.urlopen('http://127.0.0.1:7860/',timeout=8) as response:
        assert response.status==200
    return rows

def historical_head_digest(net):
    h=hashlib.sha256()
    for name,t in sorted(net.state_dict().items()):
        if name.startswith('dino_encoder.'):continue
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    guard(True)
    plan=json.loads((R/'scripts/diagnosis_plan.unlocked.json').read_text())
    evidence=json.loads((R/'evidence/live_evidence.json').read_text())
    originals=[Path(x['path']) for x in evidence['sources'] if 'sha256' in x]
    originals.append(OLD/'data/pretrain_all_mitoverse.json')
    originals.extend(Path(x['path']) for x in evidence['historical_probes'])
    before={str(p):d.sha(p) for p in originals}
    checkpoints={};metadata={}
    for spec in plan['models']:
        p=Path(spec['checkpoint_path']);checkpoints[spec['id']]=d.sha(p)
        if spec['sha256']!='UNVERIFIED': assert checkpoints[spec['id']]==spec['sha256']
        spec['sha256']=checkpoints[spec['id']]
        saved=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        metadata[spec['id']]={'path':str(p),'sha256':checkpoints[spec['id']],
           'top_keys':list(saved),'metadata':{k:v for k,v in saved.items() if k not in ('backbone',) and not isinstance(v,dict) and not torch.is_tensor(v)}} if spec['id']!='P0' else {'path':str(p),'sha256':checkpoints[spec['id']],'state_tensors':len(saved)}
        del saved
    save('checkpoint_metadata.json',metadata)
    plan['live_hashes_verified']=True
    plan['status']='UNVERIFIED_ROIS_NOT_LOCKED'
    (R/'evidence/diagnosis_plan.unlocked.json').write_text(json.dumps(plan,indent=2))
    import zarr
    manifest=json.loads((R/'evidence/formal_pretrain_manifest.json').read_text())
    entry=next(v for v in manifest['volumes'] if v['volume_id']=='kunduri22_train01')
    assert entry['legacy_split']=='train' and min(entry['shape'][1:])>=512
    group=zarr.open(entry['local_path'],mode='r')
    z=entry['shape'][0]//2;y0=(entry['shape'][1]-512)//2;x0=(entry['shape'][2]-512)//2
    raw=np.asarray(group['img'][z,y0:y0+512,x0:x0+512])
    assert raw.dtype==np.uint8 and raw.shape==(512,512)
    np.save(R/'smoke/paired_native_image.npy',raw,allow_pickle=False)
    roi={'source':'kunduri22','volume_id':entry['volume_id'],'split':'train',
         'image_path':entry['local_path']+'/img','native_bounds_zyx':[z,z+1,y0,y0+512,x0,x0+512],
         'saved_roi_sha256':d.sha(R/'smoke/paired_native_image.npy'),'metadata':dict(group.attrs),
         'spacing_yx_nm_declared':[16,16], 'physical_extent_yx_nm_declared':[8192,8192],
         'acquisition_independence':'UNVERIFIED','annotation_validity':'NOT_USED: resolution smoke reads image only',
         'ssl_exposure':{'P0':'no EM continuation; upstream corpus overlap UNVERIFIED','B100k':'included in formal SSL manifest; exact crop exposure unrecorded','B150k':'included in formal SSL manifest; exact crop exposure unrecorded'}}
    save('paired_roi_metadata.json',roi)
    probe=d.load_module(S2/'src/frozen_probe_control.py')
    sampler=probe.Samples('Dataset111_CMCC',20260907)
    sample_x,sample_y,case=sampler.sample(0)
    valid=sample_y>=0
    assert set(torch.unique(sample_y).tolist()).issubset({-1,0,1})
    save('readout_smoke_input.json',{'dataset':'Dataset111_CMCC','case':case,'split':'train','sample_shape':list(sample_x.shape),
        'labels':torch.unique(sample_y).tolist(),'validity_rule':'preprocessed label >= 0; completeness remains UNVERIFIED',
        'purpose':'two optimizer updates for gradient/freeze verification, not downstream performance'})
    all_results=[];parameters=[]
    image=torch.from_numpy(raw.copy()).float()[None,None]/255
    dummy_label=torch.zeros((1,512,512),dtype=torch.long);dummy_valid=torch.ones_like(dummy_label,dtype=torch.bool)
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    for spec in plan['models']:
        guard();t0=time.monotonic();torch.cuda.reset_peak_memory_stats()
        print('START',spec['id'],flush=True)
        enc=factory.build_encoder(spec).cuda().eval().requires_grad_(False)
        encoder_before=d.digest(enc)
        net=factory.build_primus_head(spec,enc,20260907).cuda()
        head_hash=historical_head_digest(net)
        assert head_hash=='4b19c61be6451662481678ca87ffac1162812df24c31422c0d8d3f63acc358cf'
        count=lambda m:sum(p.numel() for p in m.parameters())
        head=d.PrimusHead(net).cuda()
        parameters.append({'model':spec['id'],'encoder_parameters':count(enc),'encoder_trainable_parameters':sum(p.numel() for p in enc.parameters() if p.requires_grad),
            'up_projection_parameters':count(net.up_projection),'frequency_gate_parameters':count(net.last_gates),
            'head_parameters':count(head),'full_parameters':count(net),'embed_dim':enc.embed_dim,'patch_size':enc.patch_size,
            'interaction_indices':net.interaction_indices,'class':type(net).__module__+'.'+type(net).__name__,
            'head_initial_sha256':head_hash,'trainable_named_parameters':[{'name':n,'shape':list(p.shape),'numel':p.numel()} for n,p in net.named_parameters() if p.requires_grad]})
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            features=[f.detach().float() for f in enc.get_intermediate_layers(sample_x.repeat(1,3,1,1),n=[5,11,17,23],reshape=True)]
        result={'model':spec['id'],'head_initial_sha256':head_hash,'resolution_views':[],'readout_training':{}}
        for head_kind in ['historical_primus_frequency_gated','linear_last_layer']:
            torch.manual_seed(20260907)
            h=head if head_kind.startswith('historical') else d.LinearProbe(1024).cuda()
            opt=torch.optim.AdamW(h.parameters(),lr=1e-4)
            initial=d.digest(h);losses=[];grads=[]
            for update in range(2):
                opt.zero_grad(set_to_none=True)
                loss=d.masked_loss(h(features,sample_y.shape[-2:]),sample_y,valid)
                assert torch.isfinite(loss)
                loss.backward()
                grad_names=[n for n,p in h.named_parameters() if p.grad is not None and torch.isfinite(p.grad).all()]
                assert len(grad_names)==len(list(h.named_parameters()))
                assert all(p.grad is None for p in enc.parameters())
                grads=grad_names;opt.step();losses.append(float(loss.detach()))
            assert d.digest(h)!=initial
            result['readout_training'][head_kind]={'optimizer_steps':2,'losses':losses,'parameters':count(h),'gradient_parameters':grads,'encoder_gradients':False}
            del opt
            if h is not head:del h
        result['dense_layers']={str(i):d.dense_diagnostic(f.cpu(),sample_y.cpu(),valid.cpu(),instance_labels=False) for i,f in zip([5,11,17,23],features)}
        del features
        for mode in ['fixed_fov','native_fov']:
            for size in [256,384,512]:
                guard()
                x,_,_=d.resize_view(image,dummy_label,dummy_valid,size,mode)
                torch.cuda.synchronize();start=time.monotonic()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    fs=enc.get_intermediate_layers((x.cuda().repeat(1,3,1,1)-mean)/std,n=[5,11,17,23],reshape=True)
                with torch.no_grad():out=head([f.float() for f in fs],(size,size))
                assert torch.isfinite(out).all() and out.shape==(1,2,size,size)
                assert all(torch.isfinite(f).all() for f in fs)
                torch.cuda.synchronize()
                input_hash=hashlib.sha256(x.numpy().tobytes()).hexdigest()
                result['resolution_views'].append({'mode':mode,'input_size':size,'patch_tokens':(size//16)**2,'input_tensor_sha256':input_hash,
                    'feature_shapes':[list(f.shape) for f in fs],'logits_shape':list(out.shape),'seconds':time.monotonic()-start,
                    'physical_extent_yx_nm_declared':[8192,8192] if mode=='fixed_fov' else [size*16,size*16],
                    'new_native_information_vs256':mode=='native_fov' and size>256,
                    'same_physical_roi_across_models':True,'same_physical_roi_across_sizes':mode=='fixed_fov','segmentation_metric_computed':False})
                del fs,out
        assert d.digest(enc)==encoder_before and all(not p.requires_grad and p.grad is None for p in enc.parameters())
        result.update(status='PASS',encoder_unchanged=True,seconds=time.monotonic()-t0,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        all_results.append(result);save('real_weight_smoke.json',{'status':'RUNNING','models':all_results})
        save('model_parameter_summary.json',parameters)
        del net,head,enc;gc.collect();torch.cuda.empty_cache()
        print('PASS',spec['id'],result['seconds'],flush=True)
    input_groups={}
    for result in all_results:
        for view in result['resolution_views']:
            key=(view['mode'],view['input_size']);input_groups.setdefault(str(key),set()).add(view['input_tensor_sha256'])
    assert all(len(s)==1 for s in input_groups.values())
    after={str(p):d.sha(p) for p in originals}
    assert before==after
    after_checkpoints={spec['id']:d.sha(spec['checkpoint_path']) for spec in plan['models']}
    assert after_checkpoints==checkpoints
    guard()
    save('immutable_inputs_check.json',{'status':'PASS','before':before,'after':after,'checkpoint_hashes_before':checkpoints,'checkpoint_hashes_after':after_checkpoints})
    save('real_weight_smoke.json',{'status':'PASS','purpose':'technical smoke only; no representation-performance claim','physical_gpu':3,
        'paired_inputs_equal_across_models':True,'models':all_results,'website_http_after':200,'long_training_started':False})
    print('ALL SMOKES PASS',flush=True)

if __name__=='__main__':
    main()
