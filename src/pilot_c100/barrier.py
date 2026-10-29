"""Selective One-Sided Gram Barrier; no semantic labels or trainable parameters."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def spatial_pairs(height: int, width: int, radius: int, device):
    """Ordered pairs outside the fixed Chebyshev neighborhood; no CLS/registers."""
    yy, xx = torch.meshgrid(torch.arange(height, device=device),
                            torch.arange(width, device=device), indexing='ij')
    xy = torch.stack((yy.flatten(), xx.flatten()), -1)
    return (xy[:, None] - xy[None, :]).abs().amax(-1) > radius


def hinge_from_affinities(student, anchor, selection, margin=0.05):
    """Excluded affinities have exactly zero direct derivative, including above-margin."""
    anchor = anchor.detach()
    selection = selection.detach().bool()
    if student.shape != anchor.shape or student.shape != selection.shape:
        raise ValueError('Affinity and selection shapes differ')
    gap = student - anchor - margin
    positive = gap.clamp_min(0)
    loss = (positive.square() * selection).sum() / selection.sum().clamp_min(1)
    return loss


def separation_barrier(student_tokens, anchor_tokens, *, grid_hw=(16,16),
                       radius=2, quantile=0.5, margin=0.05):
    """FP32 similarities/reduction even inside the real BF16 SSL autocast context.

    Median selection is per view, over ordered spatially eligible pairs. Ties at
    the median are included. Normalization is by the total selected pair count.
    Coordinates are the real SSL global raster, not Pilot A offline features.
    """
    if student_tokens.shape != anchor_tokens.shape or student_tokens.ndim != 3:
        raise ValueError('Expected matching [global_views*batch,patches,channels]')
    b, n, _ = student_tokens.shape
    if n != grid_hw[0]*grid_hw[1] or margin < 0 or not 0 < quantile < 1:
        raise ValueError('Invalid spatial grid or barrier constants')
    with torch.autocast(device_type=student_tokens.device.type, enabled=False):
        student = F.normalize(student_tokens.float(), dim=-1)
        anchor = F.normalize(anchor_tokens.detach().float(), dim=-1)
        affinity_s = student @ student.transpose(-1,-2)
        affinity_0 = anchor @ anchor.transpose(-1,-2)
        eligible = spatial_pairs(*grid_hw,radius,student.device)
        if not eligible.any():
            return affinity_s.sum()*0, {'eligible_pairs':0,'selected_pairs':0,'active_pairs':0,'active_fraction':0.0}
        threshold = torch.quantile(affinity_0[:,eligible],quantile,dim=1)
        selected = eligible[None] & (affinity_0 <= threshold[:,None,None])
        loss = hinge_from_affinities(affinity_s,affinity_0,selected,margin)
        active = selected & (affinity_s-affinity_0 > margin)
        stats = {'eligible_pairs':int(eligible.sum())*b,'selected_pairs':int(selected.sum()),
                 'active_pairs':int(active.sum()),'active_fraction':float(active.sum()/selected.sum().clamp_min(1)),
                 'max_selected_gap':float((affinity_s-affinity_0)[selected].max()),
                 'min_reference_threshold':float(threshold.min()),'max_reference_threshold':float(threshold.max()),
                 'affinity_dtype':str(affinity_s.dtype),'patch_tokens':n,'views_times_batch':b}
        return loss, stats
