"""
Contact position and coverage losses for dexterous grasping.

These losses supervise the 5-fingertip contact prediction against ground-truth
contact sets (which may have variable count and are stored padded).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def contact_position_loss(
    pred_contacts: torch.Tensor,   # (B, 5, 3) predicted fingertip contacts
    gt_contacts: torch.Tensor,     # (B, M, 3) ground-truth contacts (padded)
    gt_finger_ids: torch.Tensor,   # (B, M) int 0..4, -1 for padding
    valid_mask: torch.Tensor,      # (B, M) bool
) -> torch.Tensor:                 # scalar loss
    """Fingertip-wise contact regression loss.

    For each finger f in 0..4:
        gt_f = GT contacts where finger_ids == f (variable per sample)
        loss_f = min over gt_f of ||pred_contacts[:, f, :] - gt_c||_2^2
               = min-distance from predicted contact to nearest GT contact of that finger

    Averaged over fingers and batch.
    Falls back to a global nearest-GT-contact loss if no GT contacts exist for a finger.
    """
    B, n_fingers, _ = pred_contacts.shape

    _INF = 1e9
    total_loss = pred_contacts.new_zeros(())
    n_terms = 0

    # (B,) — which samples have at least one valid GT contact
    has_any_gt = valid_mask.any(dim=1)

    for f in range(n_fingers):
        pred_f = pred_contacts[:, f, :]          # (B, 3)
        finger_mask = (gt_finger_ids == f) & valid_mask  # (B, M)
        has_finger_gt = finger_mask.any(dim=1)   # (B,)

        # Pairwise squared distances from pred_f to every GT contact
        diff = pred_f.unsqueeze(1) - gt_contacts  # (B, M, 3)
        dist_sq = (diff ** 2).sum(dim=-1)          # (B, M)

        # Min distance to a finger-f GT contact (INF where no such contact exists)
        dist_finger = dist_sq.masked_fill(~finger_mask, _INF)
        min_dist_f = dist_finger.min(dim=1).values  # (B,)

        # Fallback: min distance to any valid GT contact
        dist_all = dist_sq.masked_fill(~valid_mask, _INF)
        min_dist_all = dist_all.min(dim=1).values   # (B,)

        # Prefer finger-specific when available; fallback otherwise
        min_dist_sq = torch.where(has_finger_gt, min_dist_f, min_dist_all)  # (B,)

        # Only average over samples that have at least one GT contact
        if has_any_gt.any():
            total_loss = total_loss + min_dist_sq[has_any_gt].mean()
            n_terms += 1

    if n_terms > 0:
        return total_loss / n_terms
    return total_loss


def contact_coverage_loss(
    pred_contact_logits: torch.Tensor,  # (B, 5) confidence per finger
    gt_finger_ids: torch.Tensor,        # (B, M) int 0..4, -1 for padding
    valid_mask: torch.Tensor,           # (B, M) bool
) -> torch.Tensor:                      # scalar
    """Binary cross-entropy: predict whether each finger has a contact in GT.

    For each finger f in 0..4, the GT label is 1 if any valid contact has
    finger_id == f, else 0.  pred_contact_logits[:, f] is the raw logit.
    """
    B, n_fingers = pred_contact_logits.shape

    # Build GT binary labels: (B, 5)
    gt_active = pred_contact_logits.new_zeros(B, n_fingers)
    for f in range(n_fingers):
        has_finger = ((gt_finger_ids == f) & valid_mask).any(dim=1).float()  # (B,)
        gt_active[:, f] = has_finger

    loss = F.binary_cross_entropy_with_logits(pred_contact_logits, gt_active)
    return loss
