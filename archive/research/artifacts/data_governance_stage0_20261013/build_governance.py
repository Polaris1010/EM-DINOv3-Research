"""Reconcile frozen metadata. No image download, training, or guessed ROI admission."""
import collections
import csv
import gzip
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SRC = OUT / 'sources'
sys.path.insert(0, str(OUT / 'vendor'))
import pyarrow as pa
import pyarrow.parquet as pq
import openpyxl


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(name, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def rows_csv(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def archive_directories():
    """Parse every central record; validate total length/count; retain per-image names/CRC/offsets."""
    tail = (SRC / 'cem_zip_tail.bin').read_bytes()
    z = struct.unpack_from('<4sQ2H2L4Q', tail, tail.rfind(b'PK\x06\x06'))
    expected_entries, expected_bytes, offset = z[7], z[8], z[9]
    central = SRC / 'cem_central_directory_complete.bin'
    if central.stat().st_size != expected_bytes:
        raise RuntimeError(f'CEM central directory incomplete: {central.stat().st_size}/{expected_bytes}')
    stats, count, files = collections.defaultdict(lambda: collections.Counter()), 0, 0
    with central.open('rb') as f, gzip.open(OUT / 'cem_final_image_inventory.csv.gz', 'wt', encoding='utf-8', newline='') as dst:
        w = csv.writer(dst)
        w.writerow(['archive_path', 'crc32', 'compressed_bytes', 'uncompressed_bytes', 'local_header_offset', 'compression_method'])
        while True:
            header = f.read(46)
            if not header:
                break
            if len(header) != 46 or header[:4] != b'PK\x01\x02':
                raise ValueError('Invalid ZIP central record')
            d = struct.unpack('<4s6H3L5H2L', header)
            name = f.read(d[10]).decode('utf-8' if d[3] & 0x800 else 'cp437')
            extra = f.read(d[11])
            f.read(d[12])
            size, compressed, local_offset = d[9], d[8], d[16]
            pos = 0
            while pos + 4 <= len(extra):
                tag, length = struct.unpack_from('<HH', extra, pos)
                field = extra[pos + 4:pos + 4 + length]
                if tag == 1:
                    q = 0
                    if size == 0xffffffff:
                        size = struct.unpack_from('<Q', field, q)[0]; q += 8
                    if compressed == 0xffffffff:
                        compressed = struct.unpack_from('<Q', field, q)[0]; q += 8
                    if local_offset == 0xffffffff:
                        local_offset = struct.unpack_from('<Q', field, q)[0]
                pos += 4 + length
            count += 1
            if name.endswith('/') and len(name.rstrip('/').split('/')) == 2:
                stats[name.rstrip('/')].update(images=0,bytes_uncompressed=0)
            if not name.lower().endswith(('.tif', '.tiff')):
                continue
            parts = name.split('/')
            directory = '/'.join(parts[:-1])
            stats[directory]['images'] += 1
            stats[directory]['bytes_uncompressed'] += size
            files += 1
            w.writerow([name, f'{d[7]:08x}', compressed, size, local_offset, d[4]])
    if count != expected_entries or files != 1592753 or len(stats) != 651:
        raise ValueError(f'CEM directory reconciliation failed: entries={count}, images={files}, directories={len(stats)}')
    write_json('cem_final_archive_audit.json', dict(entries=count, images=files, source_directories=len(stats),
               central_directory_offset=offset, central_directory_bytes=expected_bytes,
               central_directory_sha256=digest(central), image_payload_read=False,
               image_bearing_directories=sum(v['images']>0 for v in stats.values()),
               empty_source_directories=[k for k,v in stats.items() if v['images']==0]))
    result = [dict(directory=k, directory_uid=k.split('/')[-1], **v) for k, v in sorted(stats.items())]
    write_csv('cem_final_source_directories.csv', result)
    return result


def mitolab_pairs():
    central=SRC/'mitolab_central_directory.bin'
    tail=(SRC/'mitolab_zip_tail.bin').read_bytes()
    eocd=struct.unpack_from('<4s4H2LH',tail,tail.rfind(b'PK\x05\x06'))
    if central.stat().st_size != eocd[5]:raise ValueError('MitoLab directory incomplete')
    images,masks,hidden={}, {}, 0
    records=0
    with central.open('rb') as f:
        while header:=f.read(46):
            d=struct.unpack('<4s6H3L5H2L',header)
            if d[0]!=b'PK\x01\x02':raise ValueError('MitoLab bad ZIP record')
            name=f.read(d[10]).decode('utf-8' if d[3]&0x800 else 'cp437')
            f.read(d[11]+d[12]);records+=1
            if not name.lower().endswith(('.tif','.tiff')):continue
            if name.split('/')[-1].startswith('._') or '__MACOSX' in name:
                hidden+=1;continue
            parts=name.split('/')
            if len(parts)<4 or parts[-2] not in {'images','masks'}:
                raise ValueError('Unrecognized MitoLab image path')
            key=('/'.join(parts[1:-2]),parts[-1])
            target=images if parts[-2]=='images' else masks
            if key in target:raise ValueError('Duplicate MitoLab image path')
            target[key]=dict(path=name,crc32=f'{d[7]:08x}',compressed_bytes=d[8],uncompressed_bytes=d[9],offset=d[16])
    if records!=eocd[4] or images.keys()!=masks.keys():
        raise ValueError(f'MitoLab pair reconciliation: {records}, {len(images)}, {len(masks)}')
    pairs=[dict(source_uid=k[0],filename=k[1],image_path=v['path'],label_path=masks[k]['path'],
                image_crc32=v['crc32'],label_crc32=masks[k]['crc32']) for k,v in sorted(images.items())]
    write_csv('mitolab_image_label_pairs.csv',pairs)
    write_json('mitolab_archive_audit.json',dict(images=len(images),labels=len(masks),source_directories=len({k[0] for k in images}),
               apple_resource_fork_tiff_entries_excluded=hidden,zip_entries=records,raw_label_name_pairs_equal=True,
               official_declared_images=21860,archive_minus_declared_images=len(images)-21860,
               count_discrepancy_status='needs_official_explanation' if len(images)!=21860 else 'reconciled',
               payloads_read=False,central_directory_sha256=digest(central)))
    return pairs


def main():
    catalog = read_json(SRC / 'mv_catalog.json')
    historical = read_json(ROOT / 'artifacts/mitoverse_full_restart_control/remote_snapshot/data/pretrain_all_mitoverse.json')
    prior = {r['volume_id']:r for r in rows_csv(ROOT / 'artifacts/mitoverse_round6/data_manifest_v3.csv')}
    if set(catalog) != {r['volume_id'] for r in historical}:
        raise ValueError('Official catalog/historical manifest volume mismatch')
    iso = [s for group in read_json(SRC / 'orgmim_groups.json').values() for s in group]
    if len(iso) != 928 or len(set(iso)) != 928:
        raise ValueError('IsoOrg identifier inventory mismatch')
    candidates = rows_csv(SRC / 'cem_ngff_candidates.csv')
    workbook = openpyxl.load_workbook(SRC / 'cem_mitolab_metadata.xlsx', read_only=True, data_only=True)
    data = list(workbook['MetaData'].values)
    fields = data[0][:51]
    metadata = [dict(zip(fields, r[:51])) for r in data[1:] if any(r[:51])]
    write_csv('cem_official_source_metadata.csv', metadata)
    dirs = archive_directories()
    labpairs=mitolab_pairs()
    metabyuid = {str(r['Sample UID']).strip():r for r in metadata}
    columns = ['asset_id','dataset','record_grain','source_id','acquisition_id','acquisition_key_candidate',
               'identity_status','image_path','label_path','valid_mask_path','path_status',
               'axes','shape_zyx','voxel_size_zyx','units','parent_roi_origin_zyx','roi_status',
               'modality','species','tissue','preparation','source_evidence','evidence_status',
               'candidate_role','admitted_role','annotation_status','metadata_json']
    registry = []

    def add(**values):
        row = dict.fromkeys(columns)
        row.update(identity_status='unresolved', roi_status='unresolved', admitted_role='quarantine',
                   annotation_status='unverified', evidence_status='metadata_only')
        row.update(values)
        registry.append(row)

    overlays = {}
    for n in ['mitoem2', 'cellmap', 'guay21']:
        for r in read_json(SRC / f'mv_{n}_overlay.json')['volumes']:
            overlays.setdefault(r['id'], []).append(dict(benchmark=n, **r))
    mvparents = collections.defaultdict(list)
    splits, labels, overlaps = [], [], []
    for h in historical:
        vid = h['volume_id']; c = catalog[vid]; p = prior[vid]
        scan = c.get('imaging', {}).get('sample')
        if scan:
            mvparents[scan].append(vid)
        roles = overlays.get(vid, [])
        # Preserve official roles as proposals; placeholder train is not an actual admission.
        candidate = 'quarantine'
        if any(r['split']=='test' for r in roles): candidate='test_candidate'
        elif any(r['split']=='val' for r in roles): candidate='dev_candidate'
        elif any(r['benchmark']=='mitoem2' for r in roles): candidate='train_candidate'
        elif any(r['benchmark']=='guay21' for r in roles): candidate='train_candidate'
        mask_present = p.get('mask_present') == 'True'
        add(asset_id='MV:'+vid, dataset='MitoVerse', record_grain='volume', source_id=c['dataset_id'],
            acquisition_key_candidate='OpenOrganelle:'+scan if scan else None,
            identity_status='official_named_scan' if scan else 'unresolved',
            image_path=h['local_path']+'/img', label_path=h['local_path']+'/mito',
            valid_mask_path=h['local_path']+'/mask' if mask_present else None,
            path_status='historical_remote_path_not_read_this_turn', axes='ZYX', shape_zyx=json.dumps(c['shape_zyx']),
            voxel_size_zyx=json.dumps(c['voxel_nm'][::-1]), units='nm', modality=c['modality'],
            species=c['species'], tissue=c['tissue'], candidate_role=candidate,
            source_evidence='sources/mv_catalog.json#'+vid+'; artifacts/mitoverse_round6/data_manifest_v3.csv',
            metadata_json=json.dumps(dict(catalog=c, legacy_split=h['legacy_split'], overlays=roles),ensure_ascii=False))
        splits.append(dict(volume_id=vid, candidate_role=candidate, admitted_role='quarantine',
                           acquisition_key_candidate='OpenOrganelle:'+scan if scan else None,
                           shape_zyx=c['shape_zyx'], region_zyx=None, official_overlay_records=roles,
                           blockers=['parent_ROI_coordinates_unverified','final_test_selection_unlocked']+
                           (['placeholder_cellmap_split'] if any(r['benchmark']=='cellmap' for r in roles) else [])+
                           (['within_volume_Pyra_region_missing'] if vid=='wei20_mitoEM-H' else [])+
                           (['acquisition_identity_unresolved'] if not scan else [])))
        labels.append(dict(dataset='MitoVerse',asset_id='MV:'+vid,label_path=h['local_path']+'/mito',
                           declared_label_type=c['label_type'],instance_id_scope='volume_candidate_not_verified',
                           zero_background_trust='unverified',annotation_completeness='unverified',
                           mask_present=str(mask_present), mask_polarity='unresolved' if mask_present else 'not_applicable',
                           valid_mask_policy='NO_ADMISSION',pixel_audit='not_performed_this_turn',
                           catalog_provenance=c['provenance'],historical_array_provenance=p.get('actual_provenance'),
                           evidence='sources/mv_catalog.json; artifacts/mitoverse_round6/data_manifest_v3.csv'))
    isoparents=collections.defaultdict(list)
    for s in iso:
        scan=s.rsplit('_',1)[0];isoparents[scan].append(s)
        add(asset_id='IsoOrg:'+s,dataset='IsoOrg-1K',record_grain='sample_identifier',source_id='OpenOrganelle',
            acquisition_key_candidate='OpenOrganelle:'+scan,identity_status='identifier_named_scan_candidate',
            path_status='original_raw_image_manifest_unavailable',source_evidence='sources/orgmim_groups.json',
            candidate_role='unlabelled_candidate',metadata_json=json.dumps({'sample_id':s}))
    for d in dirs:
        uid=d['directory_uid'];m=metabyuid.get(uid);scan=None
        if m:
            match=re.search(r'openorganelle\.janelia\.org/datasets/(jrc_[\w-]+)',str(m['Study Description']))
            scan=match[1] if match else None
        add(asset_id='CEM:'+d['directory'],dataset='CEM1.5M',record_grain='archive_source_directory',source_id=uid,
            acquisition_key_candidate='OpenOrganelle:'+scan if scan else None,
            identity_status='official_source_metadata_named_scan' if scan else 'unresolved',
            image_path='EMPIAR-11035/cem1.5m.zip::'+d['directory'],path_status='central_directory_verified_pixels_not_read',
            modality=str(m.get('Imaging Method')) if m else None,species=str(m.get('Organism')) if m else None,
            tissue=str(m.get('Biological entity')) if m else None,preparation=str(m.get('Sample Prep')) if m else None,
            source_evidence='cem_final_image_inventory.csv.gz; sources/cem_mitolab_metadata.xlsx',
            candidate_role='ssl_candidate',metadata_json=json.dumps(dict(directory=d,source_metadata=m),ensure_ascii=False))
    for index,m in enumerate(metadata,2):
        add(asset_id=f'MetaData:row{index}',dataset='CEM_official_metadata',record_grain='source_metadata_row',
            source_id=str(m['Sample UID']),path_status='metadata_only_not_proof_of_archive_membership',
            modality=str(m['Imaging Method']),species=str(m['Organism']),tissue=str(m['Biological entity']),
            preparation=str(m['Sample Prep']),source_evidence=f'sources/cem_mitolab_metadata.xlsx#MetaData!A{index}',
            metadata_json=json.dumps(m,ensure_ascii=False))
    for r in labpairs:
        add(asset_id='CEM-MitoLab:'+r['source_uid']+'/'+r['filename'],dataset='CEM-MitoLab',record_grain='2D_image_label_pair',
            source_id=r['source_uid'],image_path='EMPIAR-11037/cem_mitolab.zip::'+r['image_path'],
            label_path='EMPIAR-11037/cem_mitolab.zip::'+r['label_path'],path_status='central_directory_pair_verified_pixels_not_read',
            candidate_role='supervised_label_asset_no_duplicate_SSL_raw_admission',axes='YX',
            source_evidence='mitolab_image_label_pairs.csv; sources/empiar11037.json',
            metadata_json=json.dumps(r,ensure_ascii=False))
    tem = rows_csv(SRC/'mitonet_tem_metadata.csv')
    for i,m in enumerate(tem,2):
        add(asset_id=f'MitoNet:TEM:row{i}',dataset='MitoNet Benchmark',record_grain='TEM_metadata_row',source_id=m['Sample UID'],
            path_status='image_listing_not_yet_linked',candidate_role='external_candidate',
            modality=m.get('Imaging Method'),species=m.get('Organism'),tissue=m.get('Biological entity'),
            source_evidence=f'sources/mitonet_tem_metadata.csv#line{i}',metadata_json=json.dumps(m,ensure_ascii=False))
    benchmark_dirs=re.findall(r'href="([\w-]+)/"', (SRC/'mitonet_volume_index.html').read_text())
    for name in benchmark_dirs:
        add(asset_id='MitoNet:vEM:'+name,dataset='MitoNet Benchmark',record_grain='benchmark_directory',source_id=name,
            image_path='https://ftp.ebi.ac.uk/empiar/world_availability/10982/data/mito_benchmarks/'+name+'/',
            candidate_role='external_candidate',path_status='official_directory_verified_pixels_not_read',
            source_evidence='sources/mitonet_volume_index.html')
    fieldschema=pa.schema([(c,pa.string()) for c in columns])
    table=pa.Table.from_pylist(registry,schema=fieldschema)
    pq.write_table(table,OUT/'master_acquisition_registry.parquet',compression='zstd')
    write_csv('master_acquisition_registry.csv',registry,columns)

    def relation(a,b,level,status,evidence,action,scan=None,n_a=None,n_b=None):
        overlaps.append(dict(dataset_a=a,dataset_b=b,acquisition_key_candidate=scan,
                             relation_level=level,verification_status=status,pixel_overlap_status='unverified',
                             evidence=evidence,required_action=action,count_a=n_a,count_b=n_b))
    cemparents={r['url'].rsplit('/',1)[-1] for r in candidates if r['source']=='openorganelle'}
    shared=set(mvparents)&set(isoparents)&cemparents
    more=(set(isoparents)&cemparents)-set(mvparents)
    for scan in sorted(set(mvparents)&set(isoparents)):
        relation('MitoVerse','IsoOrg-1K','L2_candidate','named_scan_match',
                 'sources/mv_catalog.json; sources/orgmim_groups.json','holdout_entire_scan_when_reserved',
                 scan,len(mvparents[scan]),len(isoparents[scan]))
    for scan in sorted(set(isoparents)&cemparents):
        relation('CEM_download_candidates','IsoOrg-1K','L2_candidate','candidate_config_only',
                 'sources/cem_ngff_candidates.csv; sources/orgmim_groups.json','require_final_archive_membership_and_ROI',
                 scan,None,len(isoparents[scan]))
    for r in registry:
        if r['dataset']=='CEM1.5M' and r['acquisition_key_candidate']:
            scan=r['acquisition_key_candidate'].removeprefix('OpenOrganelle:')
            for target, parents in [('MitoVerse',mvparents),('IsoOrg-1K',isoparents)]:
                if scan in parents:
                    relation('CEM1.5M',target,'L2_candidate','final_directory_and_metadata_UID_match',
                             r['source_evidence']+'; '+r['asset_id'],'verify_scan_identity_then_parent_ROI',scan,None,len(parents[scan]))
    relation('CEM-MitoLab','CEM1.5M','subset_declared','official_archive_statement',
             'sources/empiar11037.json','keep_labels_separate_do_not_add_raw_images_twice')
    probe=read_json(OUT/'mitolab_duplicate_probe_evidence.json')
    relation('CEM-MitoLab::'+probe['payloads'][0]['archive_path'],
             'CEM-MitoLab::'+probe['payloads'][2]['archive_path'],'L0','file_and_decoded_SHA256_confirmed',
             'mitolab_duplicate_probe_evidence.json','quarantine_conflicting_labels_and_review_annotation_policy')
    overlaps[-1]['pixel_overlap_status']='exact_raw_pixels_confirmed_labels_semantically_differ'
    relation('MitoEM2.0','MitoVerse','overlay','official_split_overlay',
             'sources/mv_mitoem2_overlay.json','never_count_as_additional_acquisition')
    for vid in catalog:
        if vid.startswith('conrad23_'):
            relation('MitoVerse:'+vid,'MitoNet Benchmark','source_correspondence','official_catalog_EMPIAR10982',
                     'sources/mv_catalog.json#'+vid,'B_allMV_not_unseen_on_corresponding_ROI')
    relation('MitoVerse:turner22_pinky_stitched_vol19-vol34','MitoVerse:turner22_pinky_stitched_vol19-vol34_realigned',
             'L1_candidate','name_based_risk','sources/mv_catalog.json','register_realigned_vs_original_before_split')
    for scan in ['jrc_jurkat-1','jrc_macrophage-2','jrc_mus-kidney']:
        relation('MitoEM2_train:'+scan,'MitoEM2_test:'+scan,'L2','official_same_named_scan',
                 'sources/mv_mitoem2_overlay.json; sources/mv_catalog.json','strict_holdout_all_MV_CEM_IsoOrg_assets_for_scan',scan)
    write_csv('cross_dataset_overlap.csv',overlaps)
    labels.append(dict(dataset='CEM-MitoLab',asset_id='EMPIAR-11037',declared_label_type='2D_instances',
                       instance_id_scope='per_image_1_to_255',zero_background_trust='official_background_0_completeness_not_audited',
                       annotation_completeness='not_pixel_audited',valid_mask_policy='NO_ADMISSION',
                       evidence='sources/empiar11037.json',pixel_audit='not_performed_this_turn'))
    for index in [1,3]:
        labels.append(dict(dataset='CEM-MitoLab',asset_id=probe['payloads'][index]['archive_path'],
                           label_path=probe['payloads'][index]['archive_path'],declared_label_type='2D_instances',
                           annotation_completeness='unverified_conflicting_foreground_on_identical_raw_image',
                           valid_mask_policy='NO_ADMISSION',pixel_audit='selected_pair_SHA256_and_semantic_mask_comparison',
                           evidence='mitolab_duplicate_probe_evidence.json'))
    write_csv('label_validity_audit.csv',labels)
    write_json('mitoverse_clean_splits.json',dict(schema_version='1.0',status='draft_NO_GO',
               source_revision=read_json(SRC/'hf_dataset_info.json')['sha'],claim='unseen_acquisition',
               train=[],dev=[],test=[],external=[],quarantine=[r['volume_id'] for r in splits],
               candidate_assignments=splits,rule='No legacy split filtering or restored 31-volume whitelist',
               test_freeze=False,region_coordinate_convention='half_open_ZYX_stored_array_with_verified_parent_transform'))
    write_json('ssl_train_manifest.json',dict(schema_version='1.0',status='NO_GO',training_permitted=False,
               claim='unseen_acquisition',entries=[],partition_entries=[],excluded_assets=len(registry),
               blockers=['ROI_and_acquisition_proof_incomplete','test_selection_not_locked','annotation_and_pixel_audits_incomplete'],
               auto_start_training=False))
    gt=rows_csv(ROOT/'artifacts/mitoverse_gt_mito_diagnostic_20261012/GT_crop_metrics_424x4.csv')
    gtvols={r['volume_id'] for r in gt}
    exposure=[]
    for model in ['P0','B_allMV_transductive_100k','B_allMV_transductive_150k','TopoCrop_development','L_LT_historical','B_clean_future','Our_SSL_future']:
        for h in historical:
            vid=h['volume_id']
            ssl='eligible_in_all_volume_pool_not_every_slice_proven' if model.startswith('B_allMV') else 'none_project_EM_SSL' if model=='P0' else 'not_trained' if model.endswith('future') else 'unknown_historical' if model=='L_LT_historical' else 'not_a_trained_backbone'
            exposure.append(dict(model=model,asset_id='MV:'+vid,ssl_exposure=ssl,
                                  supervised_exposure='not_trained' if model.endswith('future') else 'per_decoder_label_manifest_pending',
                                  method_development_exposed=str(vid in gtvols),
                                  GT_development_exposed=str(vid in gtvols),
                                  unseen_image_claim_allowed='False' if model.startswith('B_allMV') else 'pending',
                                  evidence='historical all-MV manifest; stage1_summary.md; GT_crop_metrics_424x4.csv'))
    for model in ['B_allMV_transductive_100k','B_allMV_transductive_150k','B_clean_future']:
        exposure.append(dict(model=model,asset_id='CMCC/CMCT',ssl_exposure='raw_ROI_linkage_unknown',
                             supervised_exposure='development_decoder_runs_exist' if 'transductive' in model else 'not_trained',
                             method_development_exposed='True',GT_development_exposed='True',unseen_image_claim_allowed='pending',
                             evidence='artifacts/mitoverse_B_full_extend150k_20261012; user V3.0'))
    write_csv('data_exposure_matrix.csv',exposure)
    alias=dict(report_only=True,checkpoint_files_modified=False,
               B_full_100k=dict(report_name='B_allMV_transductive_100k',full_checkpoint_sha256='12bbf09ffcd7f93cbad88c4e55df55d36b3a299990ad976bf29af51a118a265b',hash_evidence='stage1_summary.md; not rehashed this turn'),
               B_full_150k=dict(report_name='B_allMV_transductive_150k',full_checkpoint_sha256=None,hash_status='final_checkpoint_receipt_not_available_locally',
                               teacher_backbone_sha256='9c67d08faf508a14d25418014422c0dbd641a51c122161dbd31f8af0a386d2ee',
                               teacher_hash_evidence='150k_CMCC_protocol.json; not rehashed this turn',
                               continuation_result='artifacts/mitoverse_B_full_extend150k_20261012/training_result_150k.json'))
    write_json('model_report_aliases.json',alias)
    facts=dict(MV_volumes=len(catalog),MV_sources=len({r['dataset_id'] for r in catalog.values()}),
               MV_Z_slices=sum(r['shape_zyx'][0] for r in catalog.values()),
               MV_named_OpenOrganelle_scans=len(mvparents),IsoOrg_samples=len(iso),IsoOrg_named_scan_candidates=len(isoparents),
               shared_CEM_candidate_scans=len(shared),shared_MV_volumes=sum(len(mvparents[s]) for s in shared),
               shared_IsoOrg_samples=sum(len(isoparents[s]) for s in shared),other_CEM_IsoOrg_candidate_scans=len(more),
               other_CEM_IsoOrg_samples=sum(len(isoparents[s]) for s in more),CEM_final_directories=len(dirs),
               CEM_directory_metadata_UID_matches=sum(d['directory_uid'] in metabyuid for d in dirs),
               official_metadata_rows=len(metadata),MitoNet_TEM_metadata_rows=len(tem),MitoNet_vEM_directories=len(benchmark_dirs),
               CEM_MitoLab_image_label_pairs=len(labpairs),
               CEM_MitoLab_confirmed_duplicate_pairs=1,CEM_MitoLab_CRC_duplicate_candidate_groups=20,
               GT_diagnostic_rows=len(gt),GT_diagnostic_volumes=len(gtvols),registry_rows=len(registry),
               admitted_training_images=0,verified_independent_acquisitions=None,status='NO_GO')
    write_json('audit_summary.json',facts)
    local_manifest=ROOT/'artifacts/mitoverse_full_restart_control/remote_snapshot/data/pretrain_all_mitoverse.json'
    write_json('manifest_version_reconciliation.json',dict(local_snapshot_sha256=digest(local_manifest),
               preliminary_recorded_sha256='e25c88f86d79116d9eef21f733328ed0d5f4518c25ac1061f910684bb0c36fa0',
               formal_recorded_sha256='3fd614082bf584c21e212884856875e328104ea09a293a95510c0f5f3117edb9',
               preliminary_protocol_superseded=True,formal_manifest_bytes_available=False,
               common_volume_set_verified=True,exact_formal_manifest_not_reconstructed=True,
               sampling_preliminary='source_uniform_then_volume_uniform',sampling_formal_report='uniform_legal_slices'))
    receipts=[]
    for p in sorted(SRC.iterdir()):
        if p.is_file():receipts.append(dict(file=str(p.relative_to(OUT)),sha256=digest(p),bytes=p.stat().st_size))
    write_json('source_checksums.json',receipts)
    print(json.dumps(facts,indent=2))


if __name__=='__main__':
    main()
