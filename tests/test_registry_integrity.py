"""Reject scientifically material provenance/protocol mistakes without GPU access."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from validate_repository import verify_metadata,verify_mv2
from update_registry import ROOT,read

class IntegrityTests(unittest.TestCase):
    def test_real_head_and_protocol_joins(self):
        verify_metadata();verify_mv2()

    def test_wrong_encoder_identity_is_rejected(self):
        inventory=copy.deepcopy(read(ROOT,'experiments/MV-1/checkpoint_inventory.json'))
        inventory[0]['metadata']['metadata']['encoder_sha256']='0'*64
        with self.assertRaises(AssertionError):verify_metadata(inventory=inventory)

    def test_swapped_head_path_is_rejected(self):
        inventory=copy.deepcopy(read(ROOT,'experiments/MV-1/checkpoint_inventory.json'))
        inventory[0]['path']=inventory[0]['path'].replace('/P0/','/B100k/')
        with self.assertRaises(AssertionError):verify_metadata(inventory=inventory)

    def test_tta_drift_is_rejected(self):
        manifest=copy.deepcopy(read(ROOT,'configs/evaluation/MV-2_manifest.json'))
        manifest['tta']='four_flip'
        with self.assertRaises(AssertionError):verify_mv2(manifest=manifest)

    def test_reserved_test_is_rejected(self):
        manifest=copy.deepcopy(read(ROOT,'configs/evaluation/MV-2_manifest.json'))
        manifest['reserved_test_ids'].append(manifest['entries'][0]['volume_id'])
        with self.assertRaises(AssertionError):verify_mv2(manifest=manifest)

if __name__=='__main__':unittest.main()
