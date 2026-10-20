"""Budget/coverage invariants on toy pools; actual-index distributed tests are separate."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

spec=importlib.util.spec_from_file_location('d2_coverage_core',Path(__file__).resolve().parents[1]/'src/d2_coverage/core.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)

class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.scope=patch.multiple(core,START=1,END=4,BATCH=4);self.scope.start();self.addCleanup(self.scope.stop)
        self.cem={'images':12,'groups':[{'name':'large','images':10,'first_id':0,'last_id':9},
                                     {'name':'small','images':2,'first_id':10,'last_id':11}]}
        self.ids=np.array([0,1,-1,-1,0,1,2,3,10,11,10,11,10,11,10,11],np.int32)
    def test_baseline_unchanged(self):
        np.testing.assert_array_equal(core.compile_plan('A',self.ids,self.cem),self.ids[4:])
    def test_B_removes_repeats_but_cannot_fill_large_collection(self):
        b=core.compile_plan('B',self.ids,self.cem)
        self.assertEqual(int((b<10).sum()),4)
        self.assertEqual(len(set(b[:4])),4)
        self.assertFalse(set(b[:4])&{0,1})
        self.assertEqual(len(set(b)|{0,1}),8)
    def test_C_full_coverage_and_minimum_distribution_shift(self):
        c=core.compile_plan('C',self.ids,self.cem)
        self.assertEqual(set(c)|{0,1},set(range(12)))
        self.assertEqual(len(set(c[:10])),10)
        q=np.array([(c<10).sum(),(c>=10).sum()])
        np.testing.assert_array_equal(q,[8,4])
        self.assertEqual(int(np.abs(q-[4,8]).sum()),8)
    def test_C_rejects_impossible_budget(self):
        ids=self.ids.copy();ids[6:]=-1
        with self.assertRaises(ValueError):core.compile_plan('C',ids,self.cem)
    def test_plan_compilation_repeats_exactly_without_global_rng_side_effects(self):
        state=np.random.get_state()
        for strategy in ['B','C']:
            np.testing.assert_array_equal(core.compile_plan(strategy,self.ids,self.cem),core.compile_plan(strategy,self.ids,self.cem))
        after=np.random.get_state();np.testing.assert_array_equal(state[1],after[1])
    def test_integer_remainder_quotas_and_zero_repeat_budget(self):
        np.testing.assert_array_equal(core.coverage_repeat_quotas(np.array([8,2]),np.array([4,8])),[0,2])
        np.testing.assert_array_equal(core.coverage_repeat_quotas(np.array([3,2]),np.array([2,3])),[0,0])

if __name__=='__main__':unittest.main()
