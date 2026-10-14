"""Split failures and Han-mask regression checks; CPU only."""
import importlib.util,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_mv3_manifest import validate
spec=importlib.util.spec_from_file_location('mv3_data_checks',ROOT/'src/mv3/data.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class MV3Tests(unittest.TestCase):
 def setUp(self):self.m=json.loads((ROOT/'configs/segmentation/MV-Generalist-v1/manifest.json').read_text(encoding='utf-8'))
 def test_locked_sources_and_group_holdout(self):
  validate(self.m)
  train=[r for r in self.m['entries'] if r['supervised_role']=='train'];dev=[r for r in self.m['entries'] if r['supervised_role']=='dev']
  self.assertEqual(len(train),156);self.assertEqual(len(dev),38)
  a={r['acquisition_id'] for r in train if r['source_id']=='openorganelle'};b={r['acquisition_id'] for r in dev if r['source_id']=='openorganelle'}
  self.assertTrue(a.isdisjoint(b));self.assertEqual(len(b),4)
  self.assertEqual(next(r['supervised_role'] for r in self.m['entries'] if r['id']=='muller21_high_c1'),'dev')
 def test_reserved_test_is_rejected(self):
  self.m['entries'][0]['volume_id']=self.m['reserved_test_ids'][0]
  with self.assertRaises(AssertionError):validate(self.m)
 def test_within_volume_train_dev_overlap_is_rejected(self):
  rows=[r for r in self.m['entries'] if r['volume_id']=='xiao18_vol0']
  train=next(r for r in rows if r['supervised_role']=='train');dev=next(r for r in rows if r['supervised_role']=='dev');dev['region_zyx'][0][0]=train['region_zyx'][0][1]-1
  with self.assertRaises(AssertionError):validate(self.m)
 def test_han_terminal_ids_are_valid_and_zero_is_ignored(self):
  self.assertEqual(module.is_valid(np.array([0,1,13,14]),'han24').tolist(),[False,True,True,True])
  self.assertEqual(module.is_valid(np.array([0,1,13]),'other').tolist(),[True,False,False])
  row=next(r for r in self.m['entries'] if r['source_id']=='han24');row['valid_mask_policy']='mask==0'
  with self.assertRaises(AssertionError):validate(self.m)
 def test_v2_ignores_only_han_mask_and_preserves_other_sources(self):
  v2=json.loads((ROOT/'configs/segmentation/MV-Generalist-v2/manifest.json').read_text(encoding='utf-8'));validate(v2)
  for before,after in zip(self.m['entries'],v2['entries']):
   if before['source_id']!='han24':self.assertEqual(before,after)
   else:
    self.assertFalse(module.uses_annotation_mask(after));self.assertTrue(after['mask_present'])
    self.assertEqual(before['region_zyx'],after['region_zyx']);self.assertEqual(before['supervised_role'],after['supervised_role'])
  original=json.loads((ROOT/'configs/segmentation/MV-Generalist-v1/protocol.json').read_text(encoding='utf-8'))
  current=json.loads((ROOT/'configs/segmentation/MV-Generalist-v2/protocol.json').read_text(encoding='utf-8'))
  for key,value in original.items():self.assertEqual(value,current[key])

if __name__=='__main__':unittest.main()
