"""CPU checks for the authorized baseline protocol and headless upstream adapter."""
import ast,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/mv4'))
spec=importlib.util.spec_from_file_location('mv4_official',ROOT/'src/mv4/official.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)

class MV4Tests(unittest.TestCase):
 def test_only_selected_definitions_execute_without_gui_or_module_side_effects(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'upstream.py';p.write_text('raise RuntimeError("GUI import must not execute")\ndef selected(x):\n return x*2\n\ndef other(x):\n return x*9\n')
   ns={};official.definitions(p,{'selected'},ns)
   self.assertEqual(ns['selected'](7),14);self.assertNotIn('other',ns)
 def test_missing_upstream_definition_is_explicit_error(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'upstream.py';p.write_text('def renamed(x): return x\n')
   with self.assertRaises(AssertionError):official.definitions(p,{'Engine2d'},{})
 def test_inference_has_no_gt_value_access(self):
  tree=ast.parse((ROOT/'src/mv4/infer.py').read_text())
  imported=[a.name for n in tree.body if isinstance(n,ast.ImportFrom) for a in n.names]
  self.assertNotIn('evaluation_arrays',imported)
  self.assertIn('image_array',imported)
 def test_locked_protocol_reuses_completed_assets_and_official_defaults(self):
  c=json.loads((ROOT/'configs/evaluation/MV-4_protocol.json').read_text())
  self.assertEqual(c['physical_gpus'],[1,2,3]);self.assertFalse(c['tta']);self.assertFalse(c['gt_threshold_tuning'])
  self.assertEqual(c['models'],['G-P0','G-B100','Official-MitoNet-v1'])
  self.assertEqual(c['engine_parameters']['confidence_thr'],.5)
  self.assertEqual(c['official_weight_md5'],'b56fce28d877a3c35b7ca970cec73bc9')
  self.assertEqual(c['official_weight_bytes'],221241200)
  self.assertIn('whole GT',c['han_mask_use'])

if __name__=='__main__':unittest.main()
