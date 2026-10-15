"""MB1 reductions must not weight datasets by their different plane counts."""
import importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('mb1_metrics',Path(__file__).resolve().parents[1]/'src/mb1/metrics.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class MB1MetricTests(unittest.TestCase):
 def test_empty_cases_and_undefined_precision_are_explicit(self):
  x=m.overlap(0,0,0,16);self.assertEqual((x['dice'],x['iou']),(1,1));self.assertIsNone(x['precision']);self.assertIsNone(x['recall'])
  self.assertEqual(m.overlap(0,0,2,14)['recall'],0);self.assertEqual(m.overlap(2,1,1,12)['iou'],.5)
 def test_equal_seeds_then_equal_seven_datasets(self):
  rows=[]
  for d in range(1,8):
   for seed in range(3):rows.append({'model':'G-B100','dataset_id':d,'planes':1260 if d==6 else 100,**{k:float(d)+seed for k in m.METRICS}})
  self.assertEqual(m.equal_model_dataset(rows)['G-B100']['dice'],5.)
 def test_infinite_distance_distribution_does_not_make_fake_finite_quantiles(self):
  d=m.distribution([1.,2.,float('inf'),None]);self.assertEqual(d['n'],3);self.assertEqual(d['n_missing'],1);self.assertEqual(d['p90'],float('inf'));self.assertIsNone(d['sample_sd'])
 def test_missing_physical_calibration_stays_missing(self):
  d=m.distribution([None]*100);self.assertEqual(d['n_missing'],100);self.assertIsNone(d['mean']);self.assertIsNone(d['median'])
if __name__=='__main__':unittest.main()
