from pathlib import Path
import urllib.request, json, concurrent.futures,hashlib
R=Path(__file__).resolve().parents[1]
URLS={
'mitoem_zenodo':'https://zenodo.org/api/records/20417683',
'mitoverse_tree':'https://api.github.com/repos/PytorchConnectomics/mitoverse/git/trees/df884f6bf353b3cd6b4b6a5ee3620daabd8a79be?recursive=1',
'mitoem_card':'https://huggingface.co/datasets/pytc/MitoEM2.0/raw/main/README.md',
'cellmap_tree':'https://api.github.com/repos/janelia-cellmap/cellmap-segmentation-challenge/git/trees/main?recursive=1',
'mitoem_classic':'https://mitoem.grand-challenge.org/MitoEM/'
}
def fetch(item):
 name,url=item
 try:
  b=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Codex-MV-audit'}),timeout=25).read()
  (R/'sources'/(name+'.txt')).write_bytes(b)
  print('FETCH',name,len(b),hashlib.sha256(b).hexdigest(),flush=True)
  if name=='mitoem_zenodo':print('FILES',[{'key':v['key'],'size':v['size'],'url':v['links']['self']} for v in json.loads(b)['files']],flush=True)
  elif name.endswith('tree'):print('PATHS',name,[v['path'] for v in json.loads(b)['tree'] if any(k in v['path'] for k in ['ingest','split','csv','han24'])][:100],flush=True)
  elif name=='mitoem_card':print('CARD',b.decode().encode('ascii','backslashreplace').decode(),flush=True)
  return {'name':name,'url':url,'status':'fetched','sha256':hashlib.sha256(b).hexdigest()}
 except Exception as e:
  print('FAILED',name,type(e).__name__,str(e),flush=True)
  return {'name':name,'url':url,'status':'UNVERIFIED','error':str(e)}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(5) as ex:rows=list(ex.map(fetch,URLS.items()))
 (R/'evidence/source_fetch.json').write_text(json.dumps(rows,indent=2))
