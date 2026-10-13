"""CPU-only provenance, protocol, metadata-join and repository-content checks."""
from __future__ import annotations
import ast
import hashlib
import json
import re
from pathlib import Path
from update_registry import ROOT, REQUIRED, build_registry, read, sha

def verify_metadata(root=ROOT, inventory=None, manifest=None, protocol=None):
    models={m['id']:m for m in read(root,'experiments/MV-1/model_registry.json')['models']}
    inventory=inventory if inventory is not None else read(root,'experiments/MV-1/checkpoint_inventory.json')
    manifest=manifest if manifest is not None else read(root,'configs/segmentation/MV-Dev-v1.json')
    protocol=protocol if protocol is not None else read(root,'configs/evaluation/MV-1_protocol.yaml')
    expected={(m,s) for m in models for s in protocol['seed_sequence']}
    seen=set();by_seed={}
    train={e['id'] for e in manifest['entries'] if e['supervised_role']=='train'}
    dev={e['id'] for e in manifest['entries'] if e['supervised_role']=='dev'}
    assert len(train)==4 and len(dev)==4 and train.isdisjoint(dev),'Invalid MV-1 train/dev split'
    for checkpoint in inventory:
        assert checkpoint['exists'] and checkpoint['head_parameters_in_state']==44576514
        step=checkpoint['metadata']['step'];meta=checkpoint['metadata']['metadata']
        model,seed=meta['model'],meta['seed']
        assert (model,seed) in expected and (model,seed) not in seen,'Wrong or duplicated head identity'
        assert f'/runs/{model}/seed{seed}/head_last.pth' in checkpoint['path'],'Path and stored identity disagree'
        assert step==meta['completed_steps']==protocol['locked_steps']==20000
        assert meta['encoder_frozen'] is True and meta['gates'] is False and meta['pilot'] is False
        assert meta['encoder_sha256']==models[model]['sha256'],'Encoder/checkpoint mismatch'
        assert meta['manifest_sha256']==sha(root,'configs/segmentation/MV-Dev-v1.json')
        assert meta['protocol_sha256']==sha(root,'configs/evaluation/MV-1_protocol.yaml')
        assert meta['probe_source_sha256']==sha(root,'src/segmentation/mv_probe.py')
        assert models[model]['primus_source_sha256']==sha(root,'src/segmentation/official_primus.py')
        key=(meta['decoder_initial_sha256'],meta['samples_sequence_sha256'])
        assert seed not in by_seed or by_seed[seed]==key,'Unmatched initialization or training sequence'
        by_seed[seed]=key;seen.add((model,seed))
    assert seen==expected,'Missing formal MV-1 heads'
    for model in ['B100k','B150k']:
        result=read(root,f'experiments/B_full/{model}/result.json')
        latest=read(root,f'experiments/B_full/{model}/latest_checkpoint.json')
        sidecar=read(root,f'experiments/B_full/{model}/checkpoint_metadata.json')
        assert result['status']=='complete' and result['completed_steps']==latest['step']==sidecar['step']
        assert latest['sha256']==sidecar['sha256']
        assert result['manifest_sha256']==sha(root,'configs/pretrain/pretrain_all_mitoverse.json')
    assert read(root,'experiments/B_full/B150k/result.json')['parent_checkpoint_sha256']==read(root,'experiments/B_full/B100k/latest_checkpoint.json')['sha256']

def verify_mv2(root=ROOT, manifest=None):
    mv2=manifest if manifest is not None else read(root,'configs/evaluation/MV-2_manifest.json')
    mv1=read(root,'configs/segmentation/MV-Dev-v1.json')
    train={e['id'] for e in mv1['entries'] if e['supervised_role']=='train'}
    ids=[e['id'] for e in mv2['entries']]
    volume_ids={e['volume_id'] for e in mv2['entries']}
    assert mv2['tta']=='OFF' and mv2['training_permitted'] is False and mv2['inference_only'] is True,'MV-2 protocol drift'
    assert len(ids)==len(set(ids))==188
    assert train.isdisjoint(volume_ids),'Current supervised training volume included as holdout'
    assert set(mv2['reserved_test_ids']).isdisjoint(volume_ids),'Reserved test included'
    assert mv2['crop_size']==256 and mv2['stride']==128 and mv2['tile_batch_size']==16
    assert 'unchanged' in mv2['inference_function'] and 'mv_probe.py::sliding' in mv2['inference_function']
    assert set(mv2['reuse_existing_MV1_dev_results'])=={e['id'] for e in mv1['entries'] if e['supervised_role']=='dev'}

def validate(root=ROOT):
    index=read(root,'evidence/file_index.json')
    history=read(root,'evidence/history_file_index.json')
    assert len({row['repository_path'] for row in history})==len(history)
    for record in index+history:
        path=(root/record['repository_path']).resolve()
        assert path.is_relative_to(root.resolve()),'Snapshot escapes repository'
        assert path.stat().st_size==record['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],f"Snapshot changed: {record['repository_path']}"
    upstream=(root/'src/segmentation/official_primus.upstream.py').read_text()
    runtime=(root/'src/segmentation/official_primus.py').read_text()
    assert ast.dump(ast.parse(upstream))==ast.dump(ast.parse(runtime)),'Primus differs from locked upstream source'
    assert sha(root,'src/segmentation/official_primus.upstream.py')==read(root,'evidence/primus/github_source_verification.json')['upstream_sha256']
    verify_metadata(root);verify_mv2(root)
    registry=read(root,'experiments/registry.yaml')
    assert registry==build_registry(root),'Registry is stale'
    identifiers=set()
    for row in registry['experiments']:
        assert all(k in row and row[k] is not None for k in REQUIRED)
        assert row['status'] in registry['statuses']
        assert row['experiment_id'] not in identifiers;identifiers.add(row['experiment_id'])
        assert row['evidence'] and all((root/p).is_file() for p in row['evidence'])
        if row['experiment_id'].startswith('MV-1-') or row['experiment_id']=='MV-2-notta':
            assert row['tta'] is False
        for field in ['metrics_path','report_path','train_config','dataset_manifest']:
            value=row[field]
            if value!='UNVERIFIED':assert (root/value).is_file(),f'Missing {field}: {value}'
    receipt=read(root,'evidence/collection_receipt.json')
    assert receipt['server_writes'] is False and receipt['gpu_work_started'] is False
    forbidden_suffixes={'.pth','.pt','.ckpt','.pem','.key','.tif','.tiff','.h5','.npy','.npz','.zip','.rar','.log'}
    from archive_local_research import SECRET_PATTERNS
    patterns=SECRET_PATTERNS
    scanned=0
    for path in root.rglob('*'):
        if not path.is_file() or '.git' in path.relative_to(root).parts or '__pycache__' in path.relative_to(root).parts:continue
        assert path.suffix not in forbidden_suffixes and not path.name.startswith('.env'),f'Forbidden repository content: {path.relative_to(root)}'
        assert path.stat().st_size<2000000,f'Oversized repository file: {path.relative_to(root)}'
        if path.suffix=='.py':compile(path.read_text(encoding='utf-8'),str(path),'exec')
        text=path.read_text(encoding='utf-8')
        assert not any(re.search(pattern,text) for pattern in patterns),f'Possible credential in {path.relative_to(root)}'
        scanned+=1
    print(json.dumps({'status':'PASS','provenance_files':len(index),'historical_files':len(history),'experiment_records':len(registry['experiments']),'formal_heads':9,'development_entries':188,'text_files_scanned':scanned,'gpu_used':False}))

if __name__=='__main__':validate()
