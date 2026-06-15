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


def _segment_segment_distance(
    a1: torch.Tensor,  # (..., 3) segment-1 start
    b1: torch.Tensor,  # (..., 3) segment-1 end
    a2: torch.Tensor,  # (..., 3) segment-2 start
    b2: torch.Tensor,  # (..., 3) segment-2 end
    eps: float = 1e-9,
) -> torch.Tensor:     # (...,)
    """Differentiable closest distance between two 3-D line segments.

    Vectorized clamped-``(s, t)`` solution (Ericson, *Real-Time Collision
    Detection*, ``ClosestPtSegmentSegment``). Broadcasts over arbitrary leading
    dims (only the last, size-3, axis is the point coordinate).

    Pitfall guards (the well-known capsule failure modes):
      * near-parallel segments make the line-line denominator vanish -> we clamp
        the denominator and fall back to ``s = 0`` (any point on segment 1);
      * degenerate zero-length segments -> ``A``/``E`` clamped away from 0;
      * exactly-touching segments make ``d/d(dist) sqrt`` singular -> we add
        ``eps`` inside the sqrt so the gradient stays finite at distance 0.
    """
    d1 = b1 - a1
    d2 = b2 - a2
    r = a1 - a2
    A = (d1 * d1).sum(-1)        # |seg1|^2
    E = (d2 * d2).sum(-1)        # |seg2|^2
    F = (d2 * r).sum(-1)
    B = (d1 * d2).sum(-1)
    C = (d1 * r).sum(-1)

    A_safe = A.clamp_min(eps)
    E_safe = E.clamp_min(eps)
    denom = A * E - B * B

    # s from the unconstrained line-line closest approach (clamped to [0,1]);
    # fall back to 0 when the segments are (near-)parallel.
    s = torch.where(
        denom > eps,
        ((B * F - C * E) / denom.clamp_min(eps)).clamp(0.0, 1.0),
        torch.zeros_like(denom),
    )
    t = (B * s + F) / E_safe
    # If the corresponding t fell outside [0,1], pin t to the nearer endpoint and
    # recompute s for that endpoint (decided on the *unclamped* t).
    s = torch.where(t < 0.0, (-C / A_safe).clamp(0.0, 1.0), s)
    s = torch.where(t > 1.0, ((B - C) / A_safe).clamp(0.0, 1.0), s)
    t = t.clamp(0.0, 1.0)

    c1 = a1 + s.unsqueeze(-1) * d1
    c2 = a2 + t.unsqueeze(-1) * d2
    diff = c1 - c2
    return torch.sqrt((diff * diff).sum(-1) + eps)


def capsule_capsule_distance(
    a1: torch.Tensor, b1: torch.Tensor, r1: torch.Tensor | float,
    a2: torch.Tensor, b2: torch.Tensor, r2: torch.Tensor | float,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Surface distance between two capsules (segment distance minus radii).

    Negative values mean interpenetration. Differentiable, broadcasts like
    :func:`_segment_segment_distance`.
    """
    return _segment_segment_distance(a1, b1, a2, b2, eps=eps) - (r1 + r2)


def link_self_collision_penalty(
    seg_a: torch.Tensor,        # (B, L, 3) capsule starts (world)
    seg_b: torch.Tensor,        # (B, L, 3) capsule ends (world)
    radii: torch.Tensor,        # (L,) capsule radii
    pair_mask: torch.Tensor,    # (L, L) bool, upper-tri pairs to evaluate
    clearance: float = 0.002,   # extra margin so contact is penalized early
) -> torch.Tensor:              # (B,)
    """One-sided hinge penalty on overlapping finger-link capsules.

    Over every ``pair_mask`` pair ``(i, j)``::

        penalty_ij = ReLU(r_i + r_j + clearance - dist(cap_i, cap_j))

    summed over pairs. This is the DexGraspNet ``E_spen`` one-sided hinge, but on
    capsule (segment) geometry rather than point spheres, and with adjacent
    same-finger links already removed from ``pair_mask`` (the standard
    false-positive exclusion). The ``clearance`` band yields a non-zero gradient
    just before contact, so the optimizer separates links *before* they overlap.
    """
    ii, jj = pair_mask.nonzero(as_tuple=True)          # (P,), (P,)
    if ii.numel() == 0:
        return seg_a.new_zeros(seg_a.shape[0])
    a1 = seg_a[:, ii]; b1 = seg_b[:, ii]               # (B, P, 3)
    a2 = seg_a[:, jj]; b2 = seg_b[:, jj]               # (B, P, 3)
    dist = _segment_segment_distance(a1, b1, a2, b2)   # (B, P)
    rsum = (radii[ii] + radii[jj]).unsqueeze(0)        # (1, P)
    pen = F.relu(rsum + clearance - dist)              # (B, P)
    return pen.sum(dim=-1)                             # (B,)


def inter_finger_clustering_penalty(
    seg_a: torch.Tensor,            # (B, L, 3) capsule starts (world)
    seg_b: torch.Tensor,            # (B, L, 3) capsule ends (world)
    radii: torch.Tensor,            # (L,) capsule radii
    finger_ids: torch.Tensor,       # (L,) long, link -> finger id
    personal_space: float = 0.015,  # min surface clearance between fingers
) -> torch.Tensor:                  # (B,)
    """Anti-clustering hinge on the *realized* (post-seat) finger geometry.

    For each unordered finger pair ``(f, g)`` we take the minimum capsule
    surface distance over all of their link pairs and penalize::

        penalty_fg = ReLU(personal_space - min_dist_surface(f, g))

    summed over the C(n_fingers, 2) pairs. Unlike the scorer's predicted-contact
    variance (``Q_risk``), this sees the actual seated links, so it catches
    fingers that bunch together only after contact-IK + seating. The larger
    ``personal_space`` margin keeps fingers spread even when they are not yet
    colliding (which is what :func:`link_self_collision_penalty` handles).
    """
    fids = finger_ids.to(seg_a.device)
    uniq = torch.unique(fids).tolist()
    B = seg_a.shape[0]
    pen = seg_a.new_zeros(B)
    for ai in range(len(uniq)):
        for bi in range(ai + 1, len(uniq)):
            idx_f = (fids == uniq[ai]).nonzero(as_tuple=True)[0]
            idx_g = (fids == uniq[bi]).nonzero(as_tuple=True)[0]
            # All link pairs between the two fingers: (B, nf, ng).
            a1 = seg_a[:, idx_f][:, :, None, :]
            b1 = seg_b[:, idx_f][:, :, None, :]
            a2 = seg_a[:, idx_g][:, None, :, :]
            b2 = seg_b[:, idx_g][:, None, :, :]
            a1, b1, a2, b2 = torch.broadcast_tensors(a1, b1, a2, b2)
            dist = _segment_segment_distance(a1, b1, a2, b2)        # (B, nf, ng)
            rsum = (radii[idx_f][:, None] + radii[idx_g][None, :])  # (nf, ng)
            surf = dist - rsum.unsqueeze(0)                          # (B, nf, ng)
            d_min = surf.reshape(B, -1).min(dim=1).values            # (B,)
            pen = pen + F.relu(personal_space - d_min)
    return pen


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
