import pathlib,tarfile,hashlib,zipfile,json,csv,re
p=pathlib.Path('artifacts/mitotopo_first_round');archive=p/'MitoTopo-DINO_first_round.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='8ec27b9658fe1065e061f66e6a1719b7c80f9fff4bca6f6d35cb6e33f47b52af'
d=p/'delivery';d.mkdir(exist_ok=True)
with tarfile.open(archive) as t:t.extractall(d,filter='data')
for name in ['audit_report.md','dataset_manifest.csv','checkpoint_inventory.csv','metrics.json','resource_log.csv','notes.md','resolved_config.yaml','guard_result.json','signal_regression.json']:
    assert (d/name).is_file(),name
rows=list(csv.DictReader((d/'dataset_manifest.csv').open()));assert len(rows)==212
assert next(r for r in rows if r['volume_id']=='jiang25_vol1')['voxel_nm_z_y_x']=='[40.0, 15.0, 15.0]'
assert json.loads((d/'metrics.json').read_text())['training_checkpoint_saved']==False
for link in re.findall(r'\]\(([^)]+)\)',(d/'audit_report.md').read_text(encoding='utf-8-sig')):
    if not link.startswith(('http','/')):assert (d/link).exists(),link
with zipfile.ZipFile(p/'MitoTopo-DINO_first_round.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in d.rglob('*'):
        if f.is_file():z.write(f,f.relative_to(d))
print('Verified archive checksum, report links, 212 manifest rows, ZYX conversion, stopped-run status and ZIP package.')
