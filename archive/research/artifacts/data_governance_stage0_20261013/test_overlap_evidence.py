import unittest
import numpy as np
from overlap_image_evidence import decoded_hash, registration, phash


class OverlapEvidenceTests(unittest.TestCase):
    def test_same_decoded_pixels_hash_same_and_shape_is_part_of_hash(self):
        a=np.arange(1024,dtype=np.uint8).reshape(32,32)
        self.assertEqual(decoded_hash(a),decoded_hash(a.copy()))
        self.assertNotEqual(decoded_hash(a),decoded_hash(a.reshape(16,64)))

    def test_crop_rotation_flip_and_inversion_registration(self):
        a=np.random.default_rng(4).integers(0,256,(80,80),dtype=np.uint8)
        b=255-np.rot90(np.fliplr(a[12:52,23:63]))
        result=registration(a,b)
        self.assertGreater(abs(result['ncc']),0.9999)
        self.assertTrue(result['contrast_inverted'])
        self.assertEqual(result['reduced_grid_yx'],[12,23])

    def test_constant_images_do_not_receive_false_registration_certainty(self):
        self.assertIsNone(registration(np.zeros((32,32),dtype=np.uint8),np.zeros((32,32),dtype=np.uint8))['ncc'])
        self.assertEqual(len(phash(np.zeros((32,32),dtype=np.uint8))),63)


if __name__=='__main__':unittest.main()
