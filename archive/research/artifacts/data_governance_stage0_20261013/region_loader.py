"""Read only approved, explicit half-open ZYX regions; never infer a full-volume split."""
import hashlib
import json
from pathlib import Path


class ProtocolError(ValueError):
    pass


def validate_region(region, shape):
    if len(region) != 3 or len(shape) != 3:
        raise ProtocolError('Region and shape must have Z,Y,X axes')
    for bounds, size in zip(region, shape):
        if len(bounds) != 2 or any(type(v) is not int for v in bounds):
            raise ProtocolError('Integer [start, stop) coordinates required')
        if not 0 <= bounds[0] < bounds[1] <= size:
            raise ProtocolError('Region is empty or outside stored array')
    return tuple(slice(a, b) for a, b in region)


def intersect(a, b):
    return all(max(x[0], y[0]) < min(x[1], y[1]) for x, y in zip(a, b))


def validate_partition(rows, strict_acquisition=True):
    for row in rows:
        validate_region(row['region_zyx'], row['shape_zyx'])
        if row.get('axes') != 'ZYX' or row.get('coordinate_frame') != 'stored_array':
            raise ProtocolError('Explicit ZYX stored-array coordinate frame required')
        if not row.get('acquisition_id') or not row.get('acquisition_evidence'):
            raise ProtocolError('Acquisition identity and evidence required')
        if row.get('identity_status') != 'verified' or row.get('roi_status') != 'verified':
            raise ProtocolError('Unverified acquisition or ROI')
        if row.get('role') not in {'train', 'dev', 'test', 'external'}:
            raise ProtocolError('Unknown partition role')
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a['role'] == b['role']:
                continue
            if a['acquisition_id'] == b['acquisition_id']:
                if strict_acquisition:
                    raise ProtocolError('Acquisition occurs in multiple partition roles')
                if a['image_path'] != b['image_path']:
                    # Different crop stores require parent-space registration, not local boxes.
                    raise ProtocolError('Cross-store same-acquisition ROI separation needs parent-space proof')
                if intersect(a['region_zyx'], b['region_zyx']):
                    raise ProtocolError('Partition regions overlap')


class RegionLoader:
    def __init__(self, manifest_path, expected_sha256, opener, purpose='ssl'):
        raw = Path(manifest_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ProtocolError('Manifest checksum mismatch')
        manifest = json.loads(raw)
        if manifest.get('status') != 'locked' or manifest.get('training_permitted') is not True:
            raise ProtocolError('Manifest is not admitted for training')
        if purpose not in {'ssl', 'supervised'}:
            raise ProtocolError('Unknown loader purpose')
        self.rows = manifest['entries']
        if not self.rows:
            raise ProtocolError('Empty training manifest')
        validate_partition(manifest['partition_entries'], manifest['claim'] == 'unseen_acquisition')
        approved = {json.dumps(r, sort_keys=True) for r in manifest['partition_entries'] if r['role'] == 'train'}
        if any(json.dumps(r, sort_keys=True) not in approved for r in self.rows):
            raise ProtocolError('SSL entries are not exact approved train records')
        self.opener, self.purpose = opener, purpose

    def read(self, index, relative_region_zyx):
        row = self.rows[index]
        allowed = row['region_zyx']
        extent = [b - a for a, b in allowed]
        validate_region(relative_region_zyx, extent)
        translated = [[a + c, a + d] for (a, _), (c, d) in zip(allowed, relative_region_zyx)]
        key = validate_region(translated, row['shape_zyx'])
        group = self.opener(row['image_path'])
        if list(group['img'].shape) != row['shape_zyx']:
            raise ProtocolError('Runtime shape differs from locked manifest')
        image = group['img'][key]
        if self.purpose == 'ssl':
            return image
        if row.get('annotation_status') not in {'verified_complete', 'verified_valid_regions'} or not row.get('valid_mask_policy'):
            raise ProtocolError('Annotation validity has not been certified')
        if list(group['mito'].shape) != row['shape_zyx']:
            raise ProtocolError('Label shape differs from image/manifest')
        label = group['mito'][key]
        policy = row['valid_mask_policy']
        if policy == 'complete_without_ignore':
            if row['annotation_status'] != 'verified_complete':
                raise ProtocolError('Partial annotation requires explicit validity mask')
            import numpy as np
            valid = np.ones(label.shape, dtype=bool)
        elif policy in {'mask_nonzero_is_valid', 'mask_zero_is_valid'}:
            if list(group['mask'].shape) != row['shape_zyx']:
                raise ProtocolError('Mask shape differs from image/manifest')
            mask = group['mask'][key]
            valid = mask != 0 if policy == 'mask_nonzero_is_valid' else mask == 0
        else:
            raise ProtocolError('Unknown mask polarity')
        return image, label, valid
