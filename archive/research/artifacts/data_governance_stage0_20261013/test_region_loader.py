import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from region_loader import ProtocolError, RegionLoader, validate_partition


class RegionSafetyTests(unittest.TestCase):
    def row(self, role='train', region=None, acquisition='a'):
        return dict(role=role, region_zyx=region or [[0, 2], [0, 4], [0, 4]],
                    shape_zyx=[4, 4, 4], axes='ZYX', coordinate_frame='stored_array',
                    acquisition_id=acquisition, acquisition_evidence='fixture',
                    identity_status='verified', roi_status='verified', image_path='fixture')

    def test_shared_acquisition_is_rejected_even_when_local_regions_disjoint(self):
        with self.assertRaises(ProtocolError):
            validate_partition([self.row(), self.row('test', [[2, 4], [0, 4], [0, 4]])])

    def test_region_protocol_accepts_adjacent_boxes_but_rejects_overlap(self):
        validate_partition([self.row(), self.row('test', [[2, 4], [0, 4], [0, 4]])], False)
        with self.assertRaises(ProtocolError):
            validate_partition([self.row(), self.row('test', [[1, 4], [0, 4], [0, 4]])], False)

    def test_different_crop_stores_require_parent_mapping(self):
        b = self.row('test', [[2, 4], [0, 4], [0, 4]])
        b['image_path'] = 'other-crop'
        with self.assertRaises(ProtocolError):
            validate_partition([self.row(), b], False)

    def loader(self, row, opener, purpose='ssl', status='locked'):
        manifest = dict(status=status, training_permitted=status == 'locked',
                        claim='unseen_acquisition', entries=[row], partition_entries=[row])
        raw = json.dumps(manifest).encode()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'manifest.json'
            p.write_bytes(raw)
            return RegionLoader(p, hashlib.sha256(raw).hexdigest(), opener, purpose)

    def test_ssl_reads_only_requested_image_region_and_never_labels(self):
        class ImageOnly:
            def __getitem__(self, key):
                if key != 'img':
                    raise AssertionError('SSL attempted to read labels')
                return np.arange(64).reshape(4, 4, 4)
        row = self.row(region=[[1, 3], [0, 4], [0, 4]])
        loader = self.loader(row, lambda _: ImageOnly())
        np.testing.assert_array_equal(loader.read(0, [[0, 1], [1, 3], [1, 3]]),
                                      np.arange(64).reshape(4, 4, 4)[1:2, 1:3, 1:3])
        with self.assertRaises(ProtocolError):
            loader.read(0, [[0, 3], [0, 4], [0, 4]])

    def test_no_go_cannot_be_opened(self):
        with self.assertRaises(ProtocolError):
            self.loader(self.row(), lambda _: None, status='NO_GO')

    def test_mask_polarity_controls_valid_pixels(self):
        row = self.row()
        row.update(annotation_status='verified_complete', valid_mask_policy='mask_zero_is_valid')
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        mask[0] = 3
        group = {'img':mask, 'mito':mask, 'mask':mask}
        _, _, valid = self.loader(row, lambda _:group, 'supervised').read(0, [[0, 2], [0, 4], [0, 4]])
        self.assertFalse(valid[0].any())
        self.assertTrue(valid[1].all())


if __name__ == '__main__':
    unittest.main()
