"""
Combined physics loss for dexterous grasp generation.

Wraps the kinematic penalty functions from the kinematics module into a single
differentiable loss dictionary suitable for use in training loops.
"""

from __future__ import annotations

import torch

from equidexflow.kinematics.grasp_map import wrench_balance_residual
from equidexflow.kinematics.friction_cone import friction_cone_penalty, friction_cone_penalty_local
from equidexflow.kinematics.collision import collision_penalty, self_collision_penalty


def physics_loss(
    pred_contacts: torch.Tensor,   # (B, 5, 3) predicted contact positions
    pred_normals: torch.Tensor,    # (B, 5, 3) estimated normals
    pred_forces: torch.Tensor,     # (B, 5, 3) predicted forces
    object_points: torch.Tensor,   # (B, 3, N) or (B, N, 3)
    valid_mask: torch.Tensor,      # (B, 5) bool per finger
    object_point_normals: torch.Tensor | None = None,  # (B, 3, N) or (B, N, 3)
    pred_fingertips: torch.Tensor | None = None,    # (B, n_f, 3) FK-derived fingertip positions
    fingertip_radius: torch.Tensor | float = 0.012, # sphere radius (m); 12mm for Allegro
    pred_collision_spheres: torch.Tensor | None = None,  # (B, S, 3) ALL collision spheres
    collision_sphere_radii: torch.Tensor | None = None,  # (S,) per-sphere radii
    pred_force_coords: torch.Tensor | None = None,  # (B, 5, 3), (f_t1, f_t2, f_n)
    object_mass: float = 0.2,
    mu: float = 0.5,
    alpha_w: float = 1.0,
    alpha_mu: float = 0.5,
    alpha_coll: float = 0.1,
    alpha_self: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Combined physics loss dictionary.

    Returns:
        {
          'wrench_balance': scalar,     # mean of ||G f + w_ext||^2 over batch
          'friction_cone': scalar,      # mean sum of cone violations
          'collision': scalar,          # mean fingertip-object proximity penalty
          'self_collision': scalar,     # mean fingertip self-collision penalty
          'total': weighted sum
        }
    """
    if object_points.dim() == 2:
        object_points = object_points.unsqueeze(0)
    if object_points.shape[-1] == 3:
        pts = object_points
    else:
        pts = object_points.transpose(1, 2)

    object_com = pts.mean(dim=1, keepdim=True)
    centered_contacts = pred_contacts - object_com
    object_extent = (pts.max(dim=1).values - pts.min(dim=1).values).norm(dim=-1).clamp(min=1e-3)

    # Wrench balance: (B,) residual -> scalar
    wb_residual = wrench_balance_residual(
        centered_contacts, pred_normals, pred_forces, valid_mask,
        object_mass=object_mass,
        torque_scale=object_extent,
    )  # (B,)
    wrench_balance = (wb_residual ** 2).mean()

    # Friction-cone penalty: (B,)  ->  scalar
    if pred_force_coords is None:
        fc_penalty = friction_cone_penalty(pred_forces, pred_normals, valid_mask, mu=mu)
    else:
        fc_penalty = friction_cone_penalty_local(pred_force_coords, valid_mask, mu=mu)
    friction_cone = fc_penalty.mean()

    # Collision penalty: (B,)  ->  scalar
    # Signed-distance penetration penalty against the object surface,
    # computed on Drake-FK-derived sphere positions (NOT the contact-decoder
    # output, which lives ~10cm above the mesh by data convention).
    #
    # Two paths:
    # (a) Multi-sphere (preferred): pred_collision_spheres has link midpoint
    #     spheres AND fingertip spheres.  Per-sphere margin = radius - dead_zone,
    #     where dead_zone is 2mm for fingertips (near-surface by design) and
    #     0mm for intermediate link spheres (should never touch the mesh).
    # (b) Fingertip-only (backward compat): pred_fingertips, single radius.
    if pred_collision_spheres is not None and collision_sphere_radii is not None and object_point_normals is not None:
        dead_zone_tip = 0.002   # 2mm for fingertip spheres
        dead_zone_link = 0.0    # 0mm for link spheres (no excuse to be near the mesh)
        n_link = collision_sphere_radii.shape[0] - 4  # last 4 are fingertips
        margins = collision_sphere_radii.clone().detach()
        margins[:n_link] -= dead_zone_link
        margins[n_link:] -= dead_zone_tip
        coll_penalty = collision_penalty(
            pred_collision_spheres, object_points, object_point_normals,
            margin=margins,
        )
    elif pred_fingertips is not None and object_point_normals is not None:
        if isinstance(fingertip_radius, torch.Tensor):
            radius_val = float(fingertip_radius.detach().mean())
        else:
            radius_val = float(fingertip_radius)
        margin = radius_val - 0.002
        coll_penalty = collision_penalty(
            pred_fingertips, object_points, object_point_normals,
            margin=margin,
        )
    else:
        coll_penalty = collision_penalty(
            pred_contacts, object_points, object_point_normals,
            margin=-0.003,
        )
    collision = coll_penalty.mean()

    # Self-collision penalty: (B,)  ->  scalar
    sc_penalty = self_collision_penalty(pred_contacts)
    self_coll = sc_penalty.mean()

    total = (
        alpha_w    * wrench_balance
        + alpha_mu * friction_cone
        + alpha_coll * collision
        + alpha_self * self_coll
    )

    return {
        "wrench_balance": wrench_balance,
        "friction_cone":  friction_cone,
        "collision":      collision,
        "self_collision": self_coll,
        "total":          total,
    }
