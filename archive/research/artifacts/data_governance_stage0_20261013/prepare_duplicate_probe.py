import collections
import csv
import json
import struct
from pathlib import Path

out=Path(__file__).resolve().parent
records={};groups=collections.defaultdict(list)
with (out/'sources/mitolab_central_directory.bin').open('rb') as f:
    while h:=f.read(46):
        d=struct.unpack('<4s6H3L5H2L',h);name=f.read(d[10]).decode();f.read(d[11]+d[12])
        if not name.endswith(('.tif','.tiff')):continue
        record=dict(name=name,crc32=f'{d[7]:08x}',size=d[9],compressed=d[8],offset=d[16],method=d[4])
        records[name]=record
        if '/images/' in name:groups[(d[7],d[9])].append(record)
risk=[]
for i,group in enumerate([g for g in groups.values() if len(g)>1]):
    for r in group:risk.append(dict(group_id=i,**r,status='CRC_size_candidate_not_exact_hash'))
with (out/'mitolab_duplicate_candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(risk[0]));w.writeheader();w.writerows(risk)
selected=[r for r in risk if r['group_id']==0]
jobs=[]
for i,r in enumerate(selected):
    for kind in ['images','masks']:
        rec=records[r['name'].replace('/images/',f'/{kind}/')]
        # Bound covers max ZIP extra-field length; local header is parsed, never guessed.
        length=30+len(rec['name'].encode())+65535+rec['compressed']
        jobs.append(dict(**rec,output_name=f'dup_probe_{i}_{kind}',end=rec['offset']+length-1))
(out/'sources/duplicate_probe_jobs.json').write_text(json.dumps(jobs,indent=2),encoding='utf-8')
print('CRC-size candidates:',len(risk),'groups:',len({r['group_id'] for r in risk}),'payloads_selected:',len(jobs))
