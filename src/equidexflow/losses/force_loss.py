"""
Force regression and direction losses for dexterous grasping.

These losses supervise the 5-fingertip force prediction against ground-truth
force sets (variable count, padded) and geometric priors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def force_regression_loss(
    pred_forces: torch.Tensor,   # (B, 5, 3)
    gt_forces: torch.Tensor,     # (B, M, 3) padded GT forces
    gt_finger_ids: torch.Tensor, # (B, M) int 0..4, -1 for padding
    valid_mask: torch.Tensor,    # (B, M) bool
) -> torch.Tensor:               # scalar
    """L2 force regression loss, matched by finger index.

    For each finger f, all GT forces assigned to that finger are averaged to
    produce a target force vector.  The predicted force is compared via L2.
    Falls back to the mean of all valid GT forces if no GT forces exist for
    a particular finger.
    """
    B, n_fingers, _ = pred_forces.shape

    total_loss = pred_forces.new_zeros(())
    n_terms = 0

    has_any_gt = valid_mask.any(dim=1)  # (B,)

    for f in range(n_fingers):
        pred_f = pred_forces[:, f, :]               # (B, 3)
        finger_mask = (gt_finger_ids == f) & valid_mask  # (B, M)
        has_finger_gt = finger_mask.any(dim=1)      # (B,)

        # Aggregate GT forces for this finger by mean over matched contacts
        # Sum: (B, 3)
        mask_f = finger_mask.float().unsqueeze(-1)           # (B, M, 1)
        gt_sum = (gt_forces * mask_f).sum(dim=1)             # (B, 3)
        n_contacts = finger_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        gt_mean_f = gt_sum / n_contacts                      # (B, 3)

        # Fallback: mean of all valid GT forces
        mask_all = valid_mask.float().unsqueeze(-1)
        gt_sum_all = (gt_forces * mask_all).sum(dim=1)       # (B, 3)
        n_all = valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        gt_mean_all = gt_sum_all / n_all                     # (B, 3)

        # Choose target per sample
        gt_target = torch.where(
            has_finger_gt.unsqueeze(-1).expand_as(gt_mean_f),
            gt_mean_f,
            gt_mean_all,
        )  # (B, 3)

        diff_sq = ((pred_f - gt_target) ** 2).sum(dim=-1)   # (B,)

        if has_any_gt.any():
            total_loss = total_loss + diff_sq[has_any_gt].mean()
            n_terms += 1

    if n_terms > 0:
        return total_loss / n_terms
    return total_loss


def force_direction_loss(
    pred_forces: torch.Tensor,   # (B, 5, 3)
    pred_contacts: torch.Tensor, # (B, 5, 3) contact positions
    object_points: torch.Tensor, # (B, 3, N) or (B, N, 3)
) -> torch.Tensor:               # scalar
    """Cosine loss: force direction should point approximately toward object center.

    Estimated inward normal: direction from each contact position toward the
    object centroid (mean of the point cloud).

    Loss = 1 - cosine_similarity(force, inward_direction) for each contact,
    averaged over fingers and batch (only for fingers with nonzero force magnitude).
    """
    # Normalise object_points to (B, N, 3)
    if object_points.shape[-1] == 3:
        pts = object_points           # already (B, N, 3)
    else:
        pts = object_points.permute(0, 2, 1)  # (B, 3, N)  ->  (B, N, 3)

    # Object centroid: (B, 3)
    centroid = pts.mean(dim=1)                      # (B, 3)

    # Inward direction: from contact toward centroid
    inward = centroid.unsqueeze(1) - pred_contacts  # (B, 5, 3)
    inward_norm = F.normalize(inward, dim=-1)       # (B, 5, 3)

    # Only penalise fingers with a non-trivial force
    force_mag = pred_forces.norm(dim=-1)            # (B, 5)
    nonzero = force_mag > 1e-8                      # (B, 5) bool

    # Cosine similarity between (normalised) force and inward direction
    force_norm = F.normalize(pred_forces, dim=-1)   # (B, 5, 3)
    cos_sim = (force_norm * inward_norm).sum(dim=-1)  # (B, 5)

    # Loss = 1 - cos_sim for active fingers
    loss_per_finger = (1.0 - cos_sim) * nonzero.float()  # (B, 5)

    n_valid = nonzero.float().sum()
    if n_valid > 0:
        return loss_per_finger.sum() / n_valid
    return pred_forces.new_zeros(())
