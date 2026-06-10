"""
Soft collision penalties for dexterous grasping.

Two penalties are provided:

1. collision_penalty - penalises fingertip penetration of the object using a
   signed nearest-point projection: signed distance is computed as
   ``(fingertip - nearest_point) · outward_normal``. The penalty grows
   linearly with penetration depth and is zero outside the surface. This is
   asymmetric (only penetration costs), in contrast to the old unsigned-
   distance proximity heuristic.

2. self_collision_penalty - penalises fingertip pairs that are too close,
   acting as a proxy for finger self-collision.

Note
----
The signed projection assumes outward-pointing per-point normals. For
fingertips that are inside the object but whose nearest sample point is
on the wrong side (rare for dense point clouds), the proxy can under-
estimate penetration depth - true mesh SDF remains the gold standard.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def collision_penalty(
    fingertip_positions: torch.Tensor,        # (B, F, 3)
    object_points: torch.Tensor,              # (B, N, 3) or (B, 3, N)
    object_point_normals: torch.Tensor | None = None,  # (B, N, 3) or (B, 3, N)
    margin: float | torch.Tensor = 0.0,       # scalar or (F,) per-sphere margins
) -> torch.Tensor:                            # (B,)
    """Signed-distance penetration penalty using nearest-point projection.

    For each sphere ``f`` we find its nearest object sample point ``p_j``
    and the corresponding outward normal ``n_j``. The signed distance is

        s = (f - p_j) · n_j

    Positive ``s`` means outside the surface, negative means penetrating.
    The penalty is::

        penalty_i = ReLU(margin_i - s_i)

    ``margin`` may be a scalar (same for all spheres) or a 1-D tensor of
    shape ``(F,)`` for per-sphere margins (e.g. when collision spheres have
    different radii).

    Backward compatibility
    ----------------------
    When ``object_point_normals`` is ``None``, falls back to the legacy
    unsigned proximity heuristic ``ReLU(margin - min_dist)`` to keep older
    callers working. New training code must pass normals.

    Args:
        fingertip_positions:  (B, F, 3) sphere centre XYZ positions
        object_points:        (B, N, 3) or (B, 3, N) object point cloud
        object_point_normals: (B, N, 3) or (B, 3, N) outward unit normals.
                              Zero rows are treated as "no surface signal"
                              and contribute nothing to the penalty.
        margin:               float or (F,) per-sphere margin in metres

    Returns:
        penalty: (B,) total penalty per sample (≥ 0)
    """
    if fingertip_positions.dim() == 2:
        fingertip_positions = fingertip_positions.unsqueeze(0)
    if object_points.dim() == 2:
        object_points = object_points.unsqueeze(0)

    # Normalise to (B, N, 3) if given as (B, 3, N)
    if object_points.shape[-1] == 3:
        pts = object_points
    else:
        pts = object_points.permute(0, 2, 1)

    ft = fingertip_positions  # (B, F, 3)

    # Squared distance from each fingertip to every object point: (B, F, N)
    ft_sq = (ft ** 2).sum(-1, keepdim=True)              # (B, F, 1)
    pts_sq = (pts ** 2).sum(-1).unsqueeze(1)              # (B, 1, N)
    cross = torch.bmm(ft, pts.permute(0, 2, 1))           # (B, F, N)
    dist_sq = (ft_sq + pts_sq - 2.0 * cross).clamp(min=0.0)

    # Nearest object-point index per fingertip: (B, F)
    nearest_idx = dist_sq.argmin(dim=-1)

    # Gather nearest points: (B, F, 3)
    idx_expand = nearest_idx.unsqueeze(-1).expand(-1, -1, 3)
    nearest_pts = torch.gather(pts, dim=1, index=idx_expand)

    # Legacy fallback path (no normals available).
    if object_point_normals is None:
        min_dist = dist_sq.gather(-1, nearest_idx.unsqueeze(-1)).squeeze(-1).clamp(min=0.0).sqrt()
        penalty = F.relu(margin - min_dist)               # (B, F)
        return penalty.sum(dim=-1)                        # (B,)

    # Signed-distance path: project (ft - nearest_pt) onto nearest outward normal.
    nrm = object_point_normals
    if nrm.dim() == 2:
        nrm = nrm.unsqueeze(0)
    if nrm.shape[-1] != 3:
        nrm = nrm.permute(0, 2, 1)                        # (B, N, 3)

    nearest_normals = torch.gather(nrm, dim=1, index=idx_expand)  # (B, F, 3)
    diff = ft - nearest_pts                                # (B, F, 3)
    signed = (diff * nearest_normals).sum(dim=-1)          # (B, F)

    # Mask out spheres whose nearest normal is zero (proxy fallback path):
    # the dataset signals "no surface info" by emitting zero normals.
    normal_mag = nearest_normals.norm(dim=-1)              # (B, F)
    valid = (normal_mag > 1e-6).to(signed.dtype)           # (B, F) ∈ {0, 1}

    # margin may be scalar or (F,); broadcast to (1, F) for subtraction.
    if isinstance(margin, torch.Tensor):
        m = margin.to(signed.device).to(signed.dtype).unsqueeze(0)  # (1, F)
    else:
        m = margin

    penalty = F.relu(m - signed) * valid                   # (B, F)
    return penalty.sum(dim=-1)                             # (B,)


def self_collision_penalty(
    fingertip_positions: torch.Tensor,  # (B, F, 3)
    min_distance: float = 0.01,         # 1 cm minimum finger separation
) -> torch.Tensor:                      # (B,)
    """Penalises fingertip pairs that are closer than ``min_distance``.

    For each pair (i, j) with i < j::

        penalty_ij = ReLU(min_distance - ‖p_i - p_j‖)

    Summed over all C(F, 2) pairs.
    """
    if fingertip_positions.dim() == 2:
        fingertip_positions = fingertip_positions.unsqueeze(0)

    B, F_n, _ = fingertip_positions.shape

    ft = fingertip_positions
    diff = ft.unsqueeze(2) - ft.unsqueeze(1)              # (B, F, F, 3)
    dist = torch.norm(diff, dim=-1)                        # (B, F, F)

    penalty = torch.zeros(B, dtype=ft.dtype, device=ft.device)
    for i in range(F_n):
        for j in range(i + 1, F_n):
            penalty = penalty + F.relu(min_distance - dist[:, i, j])

    return penalty  # (B,)
