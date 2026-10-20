"""Detect missing readout deployment inputs before expensive SSL can finish."""
import hashlib,importlib.util,json,os,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('readout_input_contract',os.environ.get('D12_READOUT_INPUTS_MODULE',Path(__file__).resolve().parents[1]/'src/d12_100k/readout_inputs.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ReadoutInputs(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);base=Path(self.temp.name)
  self.parent=base/'completed30k';self.root=base/'new100k';self.source=self.parent/'downstream';self.source.mkdir(parents=True)
  (self.source/'src').mkdir();(self.source/'src/data.py').write_text('# locked loader\n')
  self.protocol={'models':{'D1':'D1','D2':'D2'},'optimizer_steps':40000,'tta':False}
  self.write_protocol();entries=[{'id':str(i),'source_id':'han24' if i%12==0 else 'source'+str(i%12),'supervised_role':'train' if i<156 else 'dev'} for i in range(194)]
  path=self.source/'MV-Generalist.json';path.write_text(json.dumps({'entries':entries}));checksum=module.digest(path)
  cache=self.source/'cache';cache.mkdir();image=cache/'image.npy';label=cache/'label.npy';image.touch();label.touch()
  for row in entries:
   d=cache/row['id'];d.mkdir();(d/'CACHE.json').write_text(json.dumps({'manifest_sha256':checksum,'paths':{'image':str(image),'label':str(label)},'valid_path':None}))
 def write_protocol(self): (self.source/'MV3_protocol.json').write_text(json.dumps(self.protocol))
 @unittest.skipIf(os.name=='nt','Linux symlink contract, separately executed on server')
 def test_prepares_exact_inputs_and_is_idempotent(self):
  original={str(p.relative_to(self.source)):p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
  first=module.prepare(self.root,self.parent);second=module.prepare(self.root,self.parent)
  self.assertEqual(first,second);self.assertEqual(first['regions'],{'train':156,'dev':38});self.assertEqual(first['sources'],12)
  for name,value in original.items():self.assertEqual((self.source/name).read_bytes(),value)
 def test_missing_protocol_fails_before_training(self):
  (self.source/'MV3_protocol.json').unlink()
  with self.assertRaises(FileNotFoundError):module.prepare(self.root,self.parent)
 @unittest.skipIf(os.name=='nt','Linux symlink contract, separately executed on server')
 def test_changed_existing_config_is_not_overwritten(self):
  module.prepare(self.root,self.parent);path=self.root/'downstream/MV3_protocol.json';path.write_text('changed')
  with self.assertRaises(ValueError):module.prepare(self.root,self.parent)
  self.assertEqual(path.read_text(),'changed')
 def test_different_budget_is_rejected(self):
  self.protocol['optimizer_steps']=50000;self.write_protocol()
  with self.assertRaises(ValueError):module.prepare(self.root,self.parent)
 def test_target_cannot_be_completed_experiment(self):
  with self.assertRaises(ValueError):module.prepare(self.parent,self.parent)
if __name__=='__main__':unittest.main()
