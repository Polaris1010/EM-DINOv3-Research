"""Fail-closed formal B_full admission after live data, model, and GPU smoke audits."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path


RUN = Path(__file__).resolve().parent.parent
ROOT = Path('/opt/data/3_117/tanghr/MitoVerse_data')
EXPECTED_WEIGHT = '8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035'


def digest(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def check(condition: bool, label: str, gates: dict):
    gates[label] = 'PASS' if condition else 'FAIL'
    if not condition:
        raise RuntimeError(f'Admission gate failed: {label}')


def main():
    gates = {}
    manifest_path = RUN / 'data/pretrain_all_mitoverse.json'
    manifest = read(manifest_path)
    rows = manifest['volumes']
    manifest_hash = digest(manifest_path)
    check(manifest_hash == (RUN / 'data/pretrain_all_mitoverse.sha256').read_text().split()[0],
          'manifest_sha256', gates)
    check(manifest['training_role'] == 'ssl_full_mitoverse' and len(rows) == manifest['volume_count'] == 212,
          'manifest_role_and_count', gates)
    catalog_path = ROOT / 'catalog.json'
    catalog = read(catalog_path)
    check(manifest['catalog_sha256'] == digest(catalog_path), 'catalog_sha256', gates)
    expected = {(ROOT / item['zarr']).resolve() for item in catalog.values()}
    disk = {path.resolve() for path in (ROOT / 'data').rglob('*.zarr') if path.is_dir()}
    actual = {Path(row['local_path']).resolve() for row in rows}
    ids = [row['volume_id'] for row in rows]
    check(expected == disk == actual and len(actual) == len(rows), 'catalog_disk_manifest_path_equality', gates)
    check(set(ids) == set(catalog) and len(set(ids)) == len(rows), 'unique_catalog_volume_ids', gates)
    source_counts = {}
    for row in rows:
        vid = row['volume_id']
        source = catalog[vid]['dataset_id']
        check(row['source_dataset'] == source and row['original_dataset_id'] == source and
              row['training_role'] == 'ssl_full_mitoverse' and row['image_array_path'] == 'img',
              f'row_mapping:{vid}', gates)
        for relative, expected_hash in row['metadata_sha256'].items():
            check(digest(Path(row['local_path']) / relative) == expected_hash,
                  f'metadata:{vid}:{relative}', gates)
        source_counts[source] = source_counts.get(source, 0) + 1
    check(len(source_counts) == manifest['source_count'] == 13, 'all_sources_present', gates)
    check(source_counts.get('openorganelle') == 157, 'openorganelle_included', gates)
    sampler = (RUN / 'scripts/ssl_full_data.py').read_text()
    trainer = (RUN / 'scripts/train_B_full.py').read_text()
    check('legacy_split' not in sampler + trainer and "['mito']" not in sampler + trainer and
          "['mask']" not in sampler + trainer, 'no_legacy_or_label_filter', gates)
    (RUN / 'audit/no_legacy_filter_check.txt').write_text('PASS: no legacy_split, mito, or mask indexing in sampler/trainer\n')
    reader = read(RUN / 'audit/reader_validation.json')
    check(reader['status'] == 'pass' and reader['volume_count'] == 212 and reader['read_count'] == 636,
          'representative_slice_reads', gates)
    label = read(RUN / 'audit/no_label_access_check.json')
    check(label['status'] == 'pass' and label['samples'] == 1000 and label['slice_population'] == 48722 and
          label['label_access'] is False and not label['forbidden_getter_calls'],
          '1000_sample_no_label_access', gates)
    (RUN / 'audit/no_label_access_check.txt').write_text('PASS: 1000 augmented samples; zero non-img Zarr getter calls\n')
    init = read(RUN / 'audit/initialization_check.json')
    check(init['status'] == 'pass' and init['official_weight_sha256'] == EXPECTED_WEIGHT and
          init['coverage_ratio'] == 1 and not init['missing_keys'] and not init['unexpected_keys'] and
          not init['shape_mismatch'], 'strict_official_initialization', gates)
    check(all(item['max_absolute_difference'] == 0 for item in init['feature_comparison'].values()),
          'official_student_feature_equivalence', gates)
    single = read(RUN / 'runs/B_full/smoke20/result.json')
    check(single['status'] == 'complete' and single['completed_steps'] == 20 and
          single['tracked_student_delta_l2'] > 0 and single['tracked_teacher_delta_l2'] > 0,
          'single_gpu_smoke', gates)
    resumed = read(RUN / 'runs/B_full/resume_smoke/resume_validation.json')
    check(resumed['status'] == 'pass' and resumed['sample_chain_equal'] and resumed['resume_step'] == 20,
          'same_run_resume', gates)
    distributed = read(RUN / 'runs/B_full/distributed_final200/result.json')
    check(distributed['status'] == 'complete' and distributed['completed_steps'] == 200 and
          distributed['world_size'] == 3 and distributed['effective_global_batch'] == 48 and
          distributed['max_rank_parameter_difference'] == 0 and
          distributed['sampler_sha256'] == digest(RUN / 'scripts/ssl_full_data.py'),
          'distributed_200_steps', gates)
    heartbeat_smoke = read(RUN / 'runs/B_full/distributed_heartbeat2/result.json')
    check(heartbeat_smoke['status'] == 'complete' and heartbeat_smoke['completed_steps'] == 2 and
          heartbeat_smoke['world_size'] == 3 and heartbeat_smoke['max_rank_parameter_difference'] == 0 and
          heartbeat_smoke['trainer_sha256'] == digest(RUN / 'scripts/train_B_full.py'),
          'final_code_guarded_environment_smoke', gates)
    receipt = read(RUN / 'runs/B_full/distributed_final200/latest_checkpoint.json')
    check(receipt['step'] == 200 and Path(receipt['path']).is_file() and
          receipt['sha256'] == digest(Path(receipt['path'])), 'checkpoint_write_and_checksum', gates)
    with urllib.request.urlopen('http://127.0.0.1:7860/', timeout=3) as response:
        check(response.status == 200, 'website_healthy', gates)
    gpu_used = [int(line.strip()) for line in subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True).stdout.splitlines()]
    check(len(gpu_used) == 4 and gpu_used[1] < 600 and gpu_used[2] < 600 and gpu_used[3] < 100,
          'gpu_ownership_safe', gates)
    check(shutil.disk_usage(RUN).free > 100 * 1024**3, 'enough_disk', gates)
    check(not (RUN / 'runs/B_full/formal_attempt1').exists(), 'formal_output_fresh', gates)
    result = {'status': 'admitted', 'gates': gates, 'manifest_sha256': manifest_hash,
              'official_weight_sha256': EXPECTED_WEIGHT,
              'trainer_sha256': digest(RUN / 'scripts/train_B_full.py'),
              'sampler_sha256': digest(RUN / 'scripts/ssl_full_data.py'),
              'config_sha256': digest(RUN / 'configs/B_full.yaml'),
              'dinov3_commit': (RUN / 'src/dinov3_commit.txt').read_text().strip(),
              'world_size': 3, 'batch_per_gpu': 16, 'effective_global_batch': 48,
              'total_optimizer_steps': 100000, 'topology_feasibility':
              'separate; remote OrgMIM/SAM/GUDHI unavailable, does not block B under Stage 1 section 36'}
    (RUN / 'audit/formal_B_full_admission.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'admitted', 'gate_count': len(gates), 'manifest_sha256': manifest_hash}))


if __name__ == '__main__':
    main()
