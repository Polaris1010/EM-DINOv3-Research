from pathlib import Path
import csv
import numpy as np
import tifffile
from model import OUT,BASE,read,MB,sha
from mixture import M,C,ROOT,arrays,instance
from common import Planes
HQ=Path('/opt/data/3_117/zhangyc/Large-Field-Workspace/mitos/HQ-5nm_20261120')

def cem_items():
    cm=read(C/'manifest.json')
    for g in sorted(cm['validation']):
        yield g,[(n,0) for n,_,_ in cm['groups'][g]]

def cem_read(group,name):
    a,sem=np.load(Path('/opt/cem_mitolab_20261118')/group/(name+'.npy'))
    gt=tifffile.imread(ROOT/'MitoNet/dataset/EMPIAR-11037/cem_mitolab'/group/'masks'/name)
    assert np.array_equal(gt>0,sem>0)
    return a,gt

def hq_entries():
    p=OUT.parent/'square_hole_diagnosis_20261122/probe2_hq5nm_holes_per_image.csv'
    with p.open() as f:
        ids={r['fov'] for r in csv.DictReader(f) if r['pack']=='original' and r['model']=='Barrier100'}
    assert len(ids)==20
    with (HQ/'manifest.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['id'] in ids:yield r

def datasets():
    def cem(group,keys):
        for key,z in keys:
            a,gt=cem_read(group,key)
            yield key,z,a,gt,None,True
    for group,keys in cem_items():
        yield 'cem_validation',group,group,cem(group,keys)
    def mb(r):
        a,b=[Planes(r[k],r['shape_zyx']) for k in ('image_path','label_path')]
        for z in range(r['shape_zyx'][0]):
            yield r['id'],z,a[z],b[z],None,True
        a.close();b.close()
    for r in read(MB/'MB1_benchmark_manifest.json')['entries']:
        yield 'mb1',r['dataset'],r['id'],mb(r)
    def mv(r):
        a,b,v,j=arrays(r['id'])
        oz,oy,ox=(t[0] for t in r['region_zyx'])
        reliable=r.get('annotation_contract')=='instance_proofread'
        ids=instance(r['image_path']) if reliable else None
        for z in j['valid_z']:
            h,w=a[z].shape
            gt=b[z] if ids is None else np.asarray(ids[z+oz,oy:oy+h,ox:ox+w])
            valid=v[z]>0 if v is not None else None
            good=(gt>0)==(b[z]>0)
            assert np.all(good if valid is None else good[valid])
            yield r['id'],int(z),a[z],gt,valid,reliable
    for r in read(M/'MV-Generalist.json')['entries']:
        if r['supervised_role']=='dev':yield 'mv3',r['source_id'],r['id'],mv(r)
    def hq(r):
        p=HQ/r['file']
        assert sha(p)==r['file_sha256'] and r['annotations']=='none'
        yield r['id'],0,tifffile.imread(p),None,None,False
    for r in hq_entries():yield 'hq',r['organ'],r['id'],hq(r)
