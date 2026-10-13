import copy
import sys
from pathlib import Path
import unittest
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from diagnosis import (torch,LinearProbe,FrozenReadout,masked_loss,dense_diagnostic,
                       resize_view,validate_plan,read_roi,sha,digest,fit_readout)
from source_smoke import SyntheticEncoder,load_primus_source,HISTORICAL


class DiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_encoder_frozen_linear_gradients_and_repeatable_initialization(self):
        encoder=SyntheticEncoder(16)
        hashes=[]
        for _ in range(3):
            torch.manual_seed(123)
            hashes.append(digest(LinearProbe(16)))
        self.assertEqual(len(set(hashes)),1)
        net=FrozenReadout(encoder,LinearProbe(16)); before=digest(encoder)
        x=torch.rand(1,1,32,32);y=torch.zeros(1,32,32,dtype=torch.long)
        net.train();loss=masked_loss(net(x),y,torch.ones_like(y,dtype=torch.bool));loss.backward()
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        self.assertFalse(encoder.training);self.assertEqual(before,digest(encoder))
        self.assertTrue(all(p.grad is not None for p in net.head.parameters()))

    def test_ignored_pixels_do_not_contribute_gradient(self):
        logits=torch.randn(1,2,4,4,requires_grad=True)
        y=torch.zeros(1,4,4,dtype=torch.long);valid=torch.ones_like(y,dtype=torch.bool);valid[:,:2]=False
        masked_loss(logits,y,valid).backward()
        self.assertEqual(float(logits.grad[:,:,:2].abs().sum()),0)
        with self.assertRaises(ValueError): masked_loss(logits,y,torch.zeros_like(valid))

    def test_matched_readout_fitter_three_models(self):
        torch.manual_seed(99)
        f=[torch.randn(1,16,2,2) for _ in range(4)]
        y=torch.zeros(1,32,32,dtype=torch.long);y[:,8:24,8:24]=1
        valid=torch.ones_like(y,dtype=torch.bool)
        hashes=[];curves=[]
        for model in ["P0","B100k","B150k"]:
            torch.manual_seed(123);head=LinearProbe(16)
            result=fit_readout(head,[(f,y,valid)],[(f,y,valid)],steps=3,seed=123)
            hashes.append(result["initial_head_sha256"]);curves.append(result["curve"])
            self.assertFalse(result["encoder_updated"])
        self.assertEqual(len(set(hashes)),1);self.assertTrue(all(c==curves[0] for c in curves))
        with self.assertRaises(ValueError):fit_readout(head,[(f,y,valid)],[(f,y,valid)],steps=100000)
        f[0].requires_grad_(True)
        with self.assertRaises(ValueError):fit_readout(head,[(f,y,valid)],[(f,y,valid)],steps=3)

    def test_target_aware_boundary_and_rank(self):
        y=torch.zeros(1,64,64,dtype=torch.long);y[:,:,32:]=3
        valid=torch.ones_like(y,dtype=torch.bool)
        f=torch.zeros(1,2,4,4);f[:,0,:,2:]=1;f[:,1,:,:2]=1
        out=dense_diagnostic(f,y,valid)
        self.assertEqual(out["valid_pure_patches"],16)
        delta=[v["delta_boundary"] for v in out["adjacent_pair_strata"].values() if v["delta_boundary"] is not None]
        self.assertTrue(delta);self.assertTrue(all(abs(v-1)<1e-6 for v in delta))
        zero=dense_diagnostic(torch.zeros_like(f),y,valid)
        self.assertEqual(zero["effective_rank_singular_entropy"],0)
        binary=dense_diagnostic(f,y,valid,instance_labels=False)
        self.assertIn("identity unavailable",binary["within_interpretation"])
        self.assertTrue(all("instance_size_UNVERIFIED" in k for k in binary["adjacent_pair_strata"]))

    def test_ignored_and_mixed_patches_are_excluded(self):
        y=torch.zeros(1,32,32,dtype=torch.long);valid=torch.ones_like(y,dtype=torch.bool)
        y[:,0,0]=1;valid[:,17,17]=False
        out=dense_diagnostic(torch.randn(1,4,2,2),y,valid)
        self.assertEqual(out["valid_pure_patches"],2)

    def test_resolution_modes_and_ignore_mask(self):
        image=torch.arange(512*512,dtype=torch.float32).reshape(1,1,512,512)
        y=torch.zeros(1,512,512,dtype=torch.long);valid=torch.ones_like(y,dtype=torch.bool);valid[:,0,0]=False
        for size in (256,384,512):
            a,b,c=resize_view(image,y,valid,size,"fixed_fov")
            self.assertEqual(tuple(a.shape[-2:]),(size,size));self.assertFalse(bool(c[0,0,0]))
            d,_,_=resize_view(image,y,valid,size,"native_fov")
            origin=(512-size)//2
            self.assertEqual(float(d[0,0,0,0]),float(image[0,0,origin,origin]))
        with self.assertRaises(ValueError):resize_view(image[:,:,:128,:128],y[:,:128,:128],valid[:,:128,:128],512,"native_fov")

    def plan(self):
        return {"status":"locked_development_diagnosis","automatic_ssl_training":False,
                "models":[{"id":m,"sha256":"a"*64} for m in ["P0","B100k","B150k"]],
                "decoder_seeds":[1,2,3],"rois":[{"id":"roi","split":"dev","source_id":"s","acquisition_id":"a",
                    "image_path":"image","label_path":"label","valid_mask_path":"mask","annotation_evidence":"audit",
                    "label_kind":"binary_mito","spacing_yx_nm":[8,8],"roi_yx_nm":[0,4096,0,4096],
                    "image_sha256":"a"*64,"label_sha256":"b"*64,"valid_mask_sha256":"c"*64}]}

    def test_final_test_unknown_and_missing_provenance_rejected(self):
        validate_plan(self.plan())
        for split in ["test","external","quarantine","unknown"]:
            p=self.plan();p["rois"][0]["split"]=split
            with self.assertRaises(ValueError):validate_plan(p)
        p=self.plan();p["rois"][0]["acquisition_id"]=""
        with self.assertRaises(ValueError):validate_plan(p)
        p=self.plan();p["rois"][0]["spacing_yx_nm"]=[-1,8]
        with self.assertRaises(ValueError):validate_plan(p)

    def test_input_digest_and_physical_bounds_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            roi=self.plan()["rois"][0]
            for key,a in [("image",np.ones((512,512))), ("label",np.zeros((512,512),dtype=np.int64)),("valid_mask",np.ones((512,512),dtype=np.uint8))]:
                path=Path(temp)/(key+".npy");np.save(path,a)
                roi[key+"_path"]=str(path);roi[key+"_sha256"]=sha(path)
            read_roi(roi)
            roi["roi_yx_nm"]=[0,2048,0,4096]
            with self.assertRaises(ValueError):read_roi(roi)


if __name__=="__main__": unittest.main()
