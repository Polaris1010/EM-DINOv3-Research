"""Technical guarantees for heldout exclusion and indexed CEM image reads."""
import collections,importlib.util,io,sys,tempfile,threading,unittest,zipfile,zlib
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'src/d12'
def load(name):
 spec=importlib.util.spec_from_file_location('d12_test_'+name,P/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
s=load('sampling');c=load('common')
class D12SamplingTests(unittest.TestCase):
 @unittest.skipIf(sys.platform=='win32','Concurrent replacement is tested on the Linux training server; Windows denies open-file rename')
 def test_atomic_heartbeat_never_exposes_empty_content(self):
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder)/'HEARTBEAT';c.heartbeat(p,1.);errors=[]
   def reader():
    for _ in range(1500):
     try:float(p.read_text())
     except Exception as error:errors.append(error)
   thread=threading.Thread(target=reader);thread.start()
   for i in range(300):c.heartbeat(p,float(i+2))
   thread.join();self.assertEqual(errors,[])
 def fixtures(self):
  rows=[{'volume_id':'big','shape':[10000,512,512],'allowed_z':[[0,4],[8,10]],'legal_slices':6},{'volume_id':'small','shape':[1,512,512],'allowed_z':[[0,1]],'legal_slices':1}]
  pool={'volumes':rows,'sampling_groups':{'A':{'unknown_volume_big':['big']},'B':{'unknown_volume_small':['small']}}}
  cem={'groups':[{'name':'one','first_id':0,'last_id':0,'probability':.25},{'name':'nine','first_id':1,'last_id':9,'probability':.75}]}
  return pool,cem
 def test_holdout_and_source_balance_independent_of_raw_volume_size(self):
  pool,cem=self.fixtures();counts=collections.Counter()
  for i in range(6000):
   a=s.choose(pool,cem,0,0,i);counts[a['source']]+=1
   self.assertEqual(a['domain'],'MitoVerse')
   if a['volume_id']=='big':self.assertIn(a['z'],[0,1,2,3,8,9])
  self.assertTrue(2700<counts['A']<3300)
 def test_mixture_and_collection_weighting(self):
  pool,cem=self.fixtures();counts=collections.Counter(s.choose(pool,cem,.5,4,i)['source'] for i in range(12000))
  total=counts['one']+counts['nine'];self.assertTrue(5700<total<6300);self.assertTrue(.22<counts['one']/total<.28)
 def test_zip_read_is_lossless_and_crc_is_enforced(self):
  stream=io.BytesIO();raw=bytes(range(256))*17
  with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:archive.writestr('cem1.5M/group/image.tiff',raw)
  with zipfile.ZipFile(stream) as archive:info=archive.infolist()[0]
  args=(stream,info.header_offset,info.compress_size,info.file_size,info.CRC,info.compress_type)
  self.assertEqual(s.read_member(*args),raw)
  with self.assertRaises(AssertionError):s.read_member(*args[:4],info.CRC^1,info.compress_type)
 def test_overlapping_holdouts_and_known_source_exclusion(self):
  a=c.subtract_z([[0,100]],[20,40]);a=c.subtract_z(a,[30,70]);self.assertEqual(a,[[0,20],[70,100]])
  self.assertIsNotNone(c.cem_exclusion('Muller2020_high_c1_source'));self.assertIsNone(c.cem_exclusion('unknown_hashed_collection'))
