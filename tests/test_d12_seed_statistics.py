"""Report uncertainty must reflect three independent decoder initializations, with paired contrasts."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

CODE = Path(__file__).resolve().parents[1] / 'src/d12_seeds'

def load():
    # Avoid the historical adapters' unrelated modules named common.
    old = sys.modules.pop('common', None)
    sys.path.insert(0, str(CODE))
    try:
        spec = importlib.util.spec_from_file_location('seed_report_test', CODE / 'report.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)
        sys.modules.pop('common', None)
        if old is not None:
            sys.modules['common'] = old

class SeedStatistics(unittest.TestCase):
    def test_sample_not_population_standard_deviation(self):
        v = load().stats([.7, .8, .9])
        self.assertAlmostEqual(v['mean'], .8)
        self.assertAlmostEqual(v['sd'], .1)

    def test_nonfinite_distance_not_silently_removed(self):
        v = load().stats([1., math.inf, 2.])
        self.assertTrue(math.isinf(v['mean']))
        self.assertIsNone(v['sd'])
        self.assertEqual(v['finite_seeds'], 2)

    def test_three_seed_requirement(self):
        with self.assertRaises(AssertionError):
            load().stats([.8, .9])

    def test_paired_difference_removes_shared_seed_variation(self):
        mod = load()
        a, b = [.6, .7, .8], [.62, .72, .82]
        paired = mod.stats([y-x for x,y in zip(a,b)])
        self.assertAlmostEqual(paired['mean'], .02)
        self.assertAlmostEqual(paired['sd'], 0.)

if __name__ == '__main__':
    unittest.main()
