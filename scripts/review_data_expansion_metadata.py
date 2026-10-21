"""Reproduce the bounded metadata review; no images, network, or model imports."""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/Data-Expansion-20261021'
INPUT = OUT / 'inputs'


def js(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csvrows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def writecsv(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def scans(s):
    return set(re.findall(r'(?:jrc|aic)_[a-z0-9]+(?:-[a-z0-9]+)*', s.lower()))


def dois(s):
    return {re.sub(r'\.v[0-9]+$', '', x.rstrip('.,;)')) for x in re.findall(r'10\.\d{4,9}/[-._;()/:a-z0-9]+', s.lower())}


def string(d):
    return json.dumps(d, ensure_ascii=False)


def main():
    for rec in js(OUT/'metadata_receipts.json')['files']:
        assert hashlib.sha256((OUT/rec['path']).read_bytes()).hexdigest() == rec['sha256']
    qreceipt = js(OUT/'quantem_sources.csv.receipt.json')
    assert hashlib.sha256((OUT/'quantem_sources.csv').read_bytes()).hexdigest() == qreceipt['sha256']
    receipts = js(OUT / 'input_receipts.json')['files']
    for rec in receipts:
        path = ROOT / rec['repository_path'] if 'repository_path' in rec else OUT / 'inputs' / rec['original_path'].replace('\\', '/').rsplit('/', 1)[-1]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == rec['sha256'], path
    cp = js(ROOT / 'experiments/D12/cem_pool.json')
    mv = js(ROOT / 'experiments/D12/mitoverse_pool.json')
    catalog = js(INPUT / 'mv_catalog.json')
    iso_groups = js(INPUT / 'orgmim_groups.json')
    ids = [v for group in iso_groups.values() for v in group]
    assert len(ids) == len(set(ids)) == 928
    iso = Counter(re.sub(r'_\d+$', '', s) for s in ids)
    retained = {g['name']: g for g in cp['groups']}
    final_dirs = {r['directory_uid']: r for r in csvrows(INPUT / 'cem_final_source_directories.csv')}
    metadata = defaultdict(list)
    for row in csvrows(INPUT / 'cem_official_source_metadata.csv'):
        metadata[row['Sample UID']].append(row)
    cscan, cdoi = defaultdict(set), defaultdict(set)
    all_cem_scans = set().union(*(scans(name + ' ' + string(metadata[name])) for name,row in final_dirs.items() if int(row['images']) > 0))
    cem_rows = []
    for name, group in retained.items():
        assert int(final_dirs[name]['images']) == group['images']
        records = metadata[name]
        text = name + ' ' + string(records)
        for s in scans(text): cscan[s].add(name)
        for d in dois(text): cdoi[d].add(name)
        row = {'collection': name, 'images': group['images'], 'final_zip_present': True,
               'metadata_rows_exact_uid': len(records), 'scan_ids': ';'.join(sorted(scans(text))),
               'dois': ';'.join(sorted(dois(text))), 'modality': ';'.join(sorted({r['Imaging Method'] for r in records})),
               'tissue': ';'.join(sorted({r['Coarse Anatomical Origin'] for r in records})),
               'voxel_spacing_original_fields': string([{k:r[k] for k in ['Pixel size X','Pixel size Y','Pixel size Z','Units']} for r in records]),
               'roi_status': 'UNVERIFIED', 'pixel_duplicate_status': 'NOT_TESTED'}
        cem_rows.append(row)
    writecsv('current_cem_collections.csv', cem_rows)
    mv_scans = set().union(*(scans(v['acquisition_sampling_key']) for v in mv['volumes']))
    full_scans = set().union(*(scans(k) for k in catalog))
    mv_rows = []
    active = {v['volume_id']: v for v in mv['volumes']}
    for name, v in catalog.items():
        mv_rows.append({'volume_id': name, 'retained_ssl': name in active,
                        'scan_ids': ';'.join(sorted(scans(name))), 'source': v['dataset_id'],
                        'modality': v.get('modality','UNVERIFIED'), 'species': v.get('species','UNVERIFIED'),
                        'tissue': v.get('tissue','UNVERIFIED'), 'spacing_xyz_nm': string(v.get('voxel_nm')),
                        'allowed_z': string(active[name]['allowed_z']) if name in active else 'EXCLUDED',
                        'roi_relation_other_datasets': 'UNVERIFIED'})
    writecsv('mitoverse_membership.csv', mv_rows)
    quantem = csvrows(OUT / 'quantem_sources.csv')
    # Enrich DOI-only QuantEM names only from explicitly retrieved source metadata.
    canonical_by_doi = {}
    for f in (OUT/'metadata').glob('figshare_*.json'):
        d = js(f)
        canonical_by_doi.update({doi: scans(d.get('title','') + ' ' + d.get('description','')) for doi in dois(d.get('doi',''))})
    protected = string(js(ROOT/'experiments/MB-1/MB1_benchmark_manifest.json')) + string(csvrows(INPUT/'mitonet_tem_metadata.csv'))
    protected_doi = dois(protected)
    protected_scans = scans(protected)
    qrows = []
    qscan = defaultdict(set)
    for i, row in enumerate(quantem, 1):
        text = string(row)
        ds = dois(row['Dataset DOI / URL'])
        ss = scans(text)
        for d in ds:
            ss |= canonical_by_doi.get(d, set())
            if d.startswith('10.25378/janelia.'):
                resolved = set().union(*(scans(name + ' ' + string(metadata[name])) for name in cdoi.get(d,set()))) if cdoi.get(d) else set()
                if len(resolved) == 1: ss |= resolved
        for s in ss: qscan[s].add(i)
        cm_scan = set().union(*(cscan.get(s,set()) for s in ss)) if ss else set()
        cm_doi = set().union(*(cdoi.get(d,set()) for d in ds)) if ds else set()
        # Family flags are warnings, never proofs of acquisition or pixel identity.
        family = [x for x in ['mitoem','microns','kasthuri','lucchi','muller','mueller','guay','cdeep3m','glycolytic','salivary'] if x in text.lower()]
        warning = set(ss & full_scans) | set(ss & protected_scans) | set(ds & protected_doi) | set(family)
        qrows.append({'quantem_row_1based': i, **row, 'canonical_scan_ids': ';'.join(sorted(ss)),
                      'current_mv_scan_matches': ';'.join(sorted(ss & mv_scans)),
                      'historical_mv_scan_matches': ';'.join(sorted(ss & full_scans)),
                      'cem_retained_scan_matches': ';'.join(sorted(cm_scan)),
                      'cem_retained_doi_matches': ';'.join(sorted(cm_doi)),
                      'isoorg_scan_matches': ';'.join(sorted(ss & set(iso))),
                      'locked_evaluation_or_family_warning': ';'.join(sorted(warning)),
                      'relation': 'SAME_NAMED_ACQUISITION' if ss & (mv_scans | set(cscan) | set(iso)) else ('SAME_RECORD_DOI_NOT_PROVEN_ACQUISITION' if cm_doi else 'UNVERIFIED_NO_IDENTIFIER_MATCH'),
                      'roi_relation': 'UNVERIFIED', 'pixel_duplicate_status': 'NOT_TESTED'})
    writecsv('quantem_overlap_all_rows.csv', qrows)
    historical_candidates = scans(string(csvrows(INPUT/'cem_ngff_candidates.csv')))
    iso_rows = []
    for s,n in sorted(iso.items()):
        iso_rows.append({'scan_name_candidate': s, 'subvolumes': n,
                         'in_current_mv_ssl': s in mv_scans, 'in_historical_mv_212': s in full_scans,
                         'retained_cem_collections': ';'.join(sorted(cscan.get(s,set()))),
                         'in_historical_cem_download_candidates': s in historical_candidates,
                         'in_final_cem_archive_named_scans': s in all_cem_scans,
                         'quantem_rows_1based': ';'.join(map(str,sorted(qscan.get(s,set())))),
                         'relation': 'SAME_NAMED_ACQUISITION' if s in full_scans or s in cscan else 'UNVERIFIED_NO_IDENTIFIER_MATCH',
                         'roi_relation': 'UNVERIFIED', 'pixel_duplicate_status': 'NOT_TESTED'})
    writecsv('isoorg_overlap_all_scans.csv', iso_rows)
    pair_rows = []
    pairs = [('MitoVerse-current','CEM-current', mv_scans & set(cscan)),
             ('MitoVerse-current','IsoOrg', mv_scans & set(iso)),
             ('CEM-current','IsoOrg', set(cscan) & set(iso)),
             ('MitoVerse-current','QuantEM', mv_scans & set(qscan)),
             ('CEM-current','QuantEM', set(cscan) & set(qscan)),
             ('IsoOrg','QuantEM', set(iso) & set(qscan))]
    for a,b,ss in pairs:
        pair_rows.append({'dataset_a':a,'dataset_b':b,'named_scan_intersection_lower_bound':len(ss),
                          'scan_ids':';'.join(sorted(ss)), 'roi_relation':'UNVERIFIED',
                          'confirmed_pixel_duplicate_count':'NOT_TESTED',
                          'scope':'Exact named scans only; DOI matches and family warnings retained in row table; not exhaustive'})
    writecsv('four_dataset_overlap_summary.csv', pair_rows)
    summary = {'quantem_rows':len(quantem),'quantem_unique_names':len({r['Dataset'] for r in quantem}),
               'quantem_unique_url_strings':len({r['Dataset DOI / URL'] for r in quantem}),
               'quantem_grouping':dict(Counter(r['Grouping'] for r in quantem)),
               'current_mv_volumes':len(mv['volumes']),'current_mv_legal_z':mv['legal_slices'],
               'current_mv_named_scans':len(mv_scans),'historical_mv_named_scans':len(full_scans),
               'cem_retained_collections':len(retained),'cem_retained_images':sum(r['images'] for r in retained.values()),
               'cem_exact_uid_metadata_matches':sum(bool(metadata[n]) for n in retained),
               'cem_unknown_metadata_collections':[n for n in retained if not metadata[n]],
               'cem_modality_collection_counts':dict(Counter(r['modality'] or 'UNVERIFIED' for r in cem_rows)),
               'isoorg_no_full_mv_or_final_cem_match_scans':sorted(set(iso)-full_scans-all_cem_scans),
               'isoorg_no_full_mv_or_final_cem_match_subvolumes':sum(n for s,n in iso.items() if s not in full_scans and s not in all_cem_scans),
               'isoorg_scan_names':len(iso),'isoorg_subvolumes':sum(iso.values()),
               'isoorg_no_current_mv_or_cem_name_match_scans':sorted(set(iso)-mv_scans-set(cscan)),
               'isoorg_no_historical_mv_or_current_cem_name_match_scans':sorted(set(iso)-full_scans-set(cscan)),
               'isoorg_no_historical_mv_or_current_cem_name_match_subvolumes':sum(n for s,n in iso.items() if s not in full_scans and s not in cscan),
               'quantem_rows_cem_doi_match':sum(bool(r['cem_retained_doi_matches']) for r in qrows),
               'quantem_rows_protected_warning':sum(bool(r['locked_evaluation_or_family_warning']) for r in qrows),
               'confirmed_new_independent_acquisitions':'UNVERIFIED: absence of identifier matches is not proof of independence',
               'training_started':False, 'image_reads':0}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    candidates = js(OUT/'candidate_selection.json')['candidate_rows']
    assert len(candidates) == len({r['candidate_id'] for r in candidates}) == 15
    for c in candidates:
        q = qrows[int(c['quantem_row_1based'])-1]
        assert q['Dataset'] == c['dataset_title']
        assert not any(q[k] for k in ['cem_retained_scan_matches','cem_retained_doi_matches','historical_mv_scan_matches','locked_evaluation_or_family_warning']), c['candidate_id']
    writecsv('recommended_candidates.csv', candidates)
    print(json.dumps({'status':'PASS','quantem_rows':len(qrows),'cem_collections':len(cem_rows),'isoorg_names':len(iso_rows),'candidates':len(candidates),'image_reads':0,'training_started':False},indent=2))


if __name__ == '__main__':
    main()
