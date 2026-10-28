"""No optimizer steps. Same tests run on server CUDA and optionally local CPU."""
import unittest,os
import torch
from barrier import spatial_pairs,hinge_from_affinities,separation_barrier

class BarrierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device='cuda' if os.environ.get('PILOT_C1_TEST_CUDA')=='1' else 'cpu'

    def test_inactive_loss_and_derivative_exact_zero(self):
        a=torch.zeros((1,3,3),device=self.device,requires_grad=True)
        s=torch.full_like(a,0.04,requires_grad=True)
        loss=hinge_from_affinities(s,a,torch.ones_like(a,dtype=torch.bool))
        loss.backward();self.assertEqual(float(loss),0);self.assertEqual(int(s.grad.count_nonzero()),0);self.assertIsNone(a.grad)

    def test_excluded_pair_direct_gradient_zero(self):
        a=torch.zeros((1,3,3),device=self.device,requires_grad=True)
        s=torch.full_like(a,0.3,requires_grad=True);mask=torch.zeros_like(a,dtype=torch.bool);mask[0,0,2]=True
        loss=hinge_from_affinities(s,a,mask);loss.backward()
        self.assertGreater(float(s.grad[0,0,2]),0);self.assertEqual(int(s.grad[~mask].count_nonzero()),0);self.assertIsNone(a.grad)

    def test_nonzero_gradient_reaches_backbone_not_anchor(self):
        torch.manual_seed(11);backbone=torch.nn.Linear(7,5,device=self.device)
        x=torch.randn(2,16,7,device=self.device);anchor=torch.randn(2,16,5,device=self.device,requires_grad=True)
        loss,stats=separation_barrier(backbone(x),anchor,grid_hw=(4,4));loss.backward()
        self.assertGreater(float(loss),0);self.assertGreater(float(backbone.weight.grad.norm()),0)
        self.assertIsNone(anchor.grad);self.assertGreater(stats['active_pairs'],0)

    def test_same_features_loss_exact_zero_and_spatial_grid(self):
        torch.manual_seed(3);x=torch.randn(2,256,9,device=self.device,requires_grad=True)
        loss,stats=separation_barrier(x,x);loss.backward()
        self.assertEqual(float(loss),0);self.assertEqual(int(x.grad.count_nonzero()),0)
        mask=spatial_pairs(16,16,2,self.device)
        self.assertFalse(bool(mask[0,2]));self.assertTrue(bool(mask[0,3]));self.assertFalse(bool(mask.diagonal().any()))
        self.assertEqual(stats['patch_tokens'],256)

    def test_fp32_reduction_inside_bf16_context(self):
        torch.manual_seed(5);s=torch.randn(2,256,9,device=self.device,requires_grad=True);a=torch.randn_like(s)
        with torch.autocast(device_type=self.device,dtype=torch.bfloat16):loss,stats=separation_barrier(s,a)
        self.assertEqual(loss.dtype,torch.float32);self.assertEqual(stats['affinity_dtype'],'torch.float32')
        loss.backward();self.assertTrue(bool(torch.isfinite(s.grad).all()))

    def test_empty_selection_safe_zero(self):
        x=torch.randn(1,4,3,device=self.device,requires_grad=True)
        loss,stats=separation_barrier(x,x,grid_hw=(2,2));loss.backward()
        self.assertEqual(float(loss),0);self.assertEqual(int(x.grad.count_nonzero()),0)

if __name__=='__main__':unittest.main(verbosity=2)
