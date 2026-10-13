"""Read only public metadata from official ZIPs via checked HTTP byte ranges."""
from pathlib import Path
import io,json,zipfile,urllib.request,concurrent.futures,hashlib,time
R=Path(__file__).resolve().parents[1]
class RemoteFile(io.RawIOBase):
 def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.cache={};self.requests=0
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  assert 0<=self.pos<=self.size
  return self.pos
 def read(self,n=-1):
  n=min(self.size-self.pos,n if n>=0 else self.size)
  if not n:return b''
  a,b=self.pos,self.pos+n-1;self.pos+=n
  key=(a,b)
  if key in self.cache:return self.cache[key]
  assert n<4<<20,'Refuse bulk data request'
  req=urllib.request.Request(self.url,headers={'User-Agent':'Codex-split-metadata','Range':f'bytes={a}-{b}'})
  with urllib.request.urlopen(req,timeout=45) as f:
   assert f.status==206,(f.status,'Range unsupported')
   assert f.headers['Content-Range'].startswith(f'bytes {a}-{b}/')
   value=f.read(n+1);assert len(value)==n
  self.requests+=1;self.cache[key]=value;return value
def work(v):
 name=v['key'].removesuffix('.zip');result={'name':name,'url':v['links']['self'],'size':v['size']}
 try:
  rf=RemoteFile(f"https://zenodo.org/records/20417683/files/{v['key']}?download=1",v['size'])
  with zipfile.ZipFile(rf) as z:
   info=[{'name':i.filename,'bytes':i.file_size} for i in z.infolist()]
   (R/'sources'/(name+'_listing.json')).write_text(json.dumps(info,indent=2))
   names=[i.filename for i in z.infolist() if Path(i.filename).name in ['split.json','dataset.json','metadata.csv','README.md','splits_final.json']]
   for file in names:
    data=z.read(file);assert len(data)<1<<20
    target=R/'sources'/name/Path(file).name;target.parent.mkdir(exist_ok=True)
    # Do not conflate multiple dataset folders.
    assert not target.exists()
    target.write_bytes(data)
    print('ZIP_METADATA',name,file,len(data),data[:900].decode('utf8',errors='replace').encode('ascii','backslashreplace').decode(),flush=True)
   result.update(status='fetched',files=names,range_requests=rf.requests)
  print('ZIP_DONE',name,len(info),flush=True)
 except Exception as e:
  result.update(status='UNVERIFIED',error=str(e));print('ZIP_FAILED',name,type(e).__name__,str(e),flush=True)
 return result
if __name__=='__main__':
 items=[v for v in json.loads((R/'sources/mitoem_zenodo.txt').read_text())['files'] if v['key'].endswith('.zip') and 'OMEZarr' not in v['key']]
 with concurrent.futures.ThreadPoolExecutor(4) as ex:results=list(ex.map(work,items))
 (R/'evidence/official_zip_metadata.json').write_text(json.dumps(results,indent=2))
