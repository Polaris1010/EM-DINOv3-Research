"""Read image geometry/prefix only; never fetch test label data."""
from pathlib import Path
import sys,json,gzip,struct,concurrent.futures
sys.path.insert(0,str(Path(__file__).parent))
from fetch_zip_metadata import RemoteFile,R
import zipfile,numpy as np
def work(v):
 name=v['key'].removesuffix('.zip');rows=[]
 try:
  rf=RemoteFile(f"https://zenodo.org/records/20417683/files/{v['key']}?download=1",v['size'])
  with zipfile.ZipFile(rf) as z:
   for p in z.namelist():
    if '/imagesTr/' not in p or not p.endswith('.nii.gz'):continue
    with z.open(p) as f,gzip.GzipFile(fileobj=f) as g:
     h=g.read(352);order='<' if struct.unpack('<i',h[:4])[0]==348 else '>'
     assert struct.unpack(order+'i',h[:4])[0]==348
     dims=struct.unpack(order+'8h',h[40:56]);dt=struct.unpack(order+'h',h[70:72])[0]
     pix=struct.unpack(order+'8f',h[76:108]);off=int(struct.unpack(order+'f',h[108:112])[0]);assert 352<=off<1<<20
     slope,inter=struct.unpack(order+'2f',h[112:120]);aff=[struct.unpack(order+'4f',h[j:j+16]) for j in [280,296,312]]
     if off>352:g.read(off-352)
     dtype={2:'u1',4:'i2',8:'i4',16:'f4',64:'f8',512:'u2',768:'u4'}[dt]
     values=np.frombuffer(g.read(1024*np.dtype(dtype).itemsize),dtype=np.dtype(order+dtype)).copy()
    row={'file':p,'shape_xyz':list(dims[1:4]),'shape_zyx':list(reversed(dims[1:4])),
     'pixdim_xyz':list(pix[1:4]),'datatype':dt,'slope':slope,'intercept':inter,'affine_srows':aff,'prefix_values':values.tolist()}
    rows.append(row);print('IMAGE_HEADER',name,Path(p).name,row['shape_zyx'],row['pixdim_xyz'],'dt',dt,'range',float(values.min()),float(values.max()),flush=True)
  (R/'sources'/(name+'_image_headers.json')).write_text(json.dumps(rows,indent=2))
  return {'name':name,'status':'fetched','images':len(rows),'test_labels_accessed':False}
 except Exception as e:
  print('HEADER_FAILED',name,type(e).__name__,str(e),flush=True);return {'name':name,'status':'UNVERIFIED','error':str(e)}
if __name__=='__main__':
 items=[v for v in json.loads((R/'sources/mitoem_zenodo.txt').read_text())['files'] if v['key'].endswith('.zip') and 'OMEZarr' not in v['key']]
 with concurrent.futures.ThreadPoolExecutor(4) as ex:rows=list(ex.map(work,items))
 (R/'evidence/nifti_header_fetch.json').write_text(json.dumps(rows,indent=2))
