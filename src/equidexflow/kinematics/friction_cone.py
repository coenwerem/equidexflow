"""
Differentiable friction-cone penalties for dexterous grasping.

Linearised / soft Coulomb model:

    For contact i with inward surface normal n̂_i and force f_i:

        f_n  = dot(f_i, n̂_i)          [normal component, should be ≥ 0]
        f_t  = f_i - f_n * n̂_i        [tangential component]

    Friction-cone constraint:
        ‖f_t‖ ≤ μ · f_n    (and f_n ≥ 0)

    Penalty (soft, differentiable):
        penalty_i = ReLU(‖f_t‖ - μ · f_n)   +   ReLU(-f_n)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def friction_cone_penalty(
    forces: torch.Tensor,        # (B, M, 3)
    normals: torch.Tensor,       # (B, M, 3) inward unit normals
    valid_mask: torch.Tensor,    # (B, M) bool
    mu: float = 0.5,
) -> torch.Tensor:               # (B,)
    """Differentiable friction-cone penalty summed over valid contacts.

    Penalises:
    * tangential force exceeding the friction cone  (sliding)
    * pulling force  (f_n < 0)

    Args:
        forces:     (B, M, 3) contact forces (world frame)
        normals:    (B, M, 3) inward unit normals
        valid_mask: (B, M) boolean; invalid contacts are excluded
        mu:         coefficient of friction

    Returns:
        penalty: (B,) total penalty per sample (≥ 0)
    """
    if forces.dim() == 2:
        forces = forces.unsqueeze(0)
    if normals.dim() == 2:
        normals = normals.unsqueeze(0)
    if valid_mask.dim() == 1:
        valid_mask = valid_mask.unsqueeze(0)

    # Normal force component: scalar, (B, M)
    f_n = (forces * normals).sum(dim=-1)

    # Tangential force component: (B, M, 3)
    f_t = forces - f_n.unsqueeze(-1) * normals

    # Tangential magnitude: (B, M)
    f_t_norm = torch.norm(f_t, dim=-1)

    # Cone violation: tangential exceeds mu * normal
    cone_viol = F.relu(f_t_norm - mu * f_n)

    # Pulling penalty: normal force should be non-negative
    pull_viol = F.relu(-f_n)

    penalty = cone_viol + pull_viol  # (B, M)

    # Mask out invalid contacts
    penalty = penalty * valid_mask.float()

    return penalty.sum(dim=-1)  # (B,)


def friction_cone_penalty_local(
    force_coords: torch.Tensor,  # (B, M, 3), ordered (f_t1, f_t2, f_n)
    valid_mask: torch.Tensor,    # (B, M) bool
    mu: float = 0.5,
) -> torch.Tensor:              # (B,)
    """Differentiable friction-cone penalty in local contact coordinates."""
    if force_coords.dim() == 2:
        force_coords = force_coords.unsqueeze(0)
    if valid_mask.dim() == 1:
        valid_mask = valid_mask.unsqueeze(0)

    f_t_norm = torch.norm(force_coords[..., :2], dim=-1)
    f_n = force_coords[..., 2]
    penalty = F.relu(f_t_norm - mu * f_n) + F.relu(-f_n)
    return (penalty * valid_mask.float()).sum(dim=-1)


def friction_cone_violation_rate(
    forces: torch.Tensor,       # (B, M, 3)
    normals: torch.Tensor,      # (B, M, 3)
    valid_mask: torch.Tensor,   # (B, M) bool
    mu: float = 0.5,
) -> torch.Tensor:              # (B,)
    """Fraction of valid contacts that violate the friction cone.

    Non-differentiable; intended for evaluation / logging only.

    Args:
        forces:     (B, M, 3) contact forces
        normals:    (B, M, 3) inward unit normals
        valid_mask: (B, M) boolean mask
        mu:         coefficient of friction

    Returns:
        rate: (B,) fraction in [0, 1]
    """
    if forces.dim() == 2:
        forces = forces.unsqueeze(0)
    if normals.dim() == 2:
        normals = normals.unsqueeze(0)
    if valid_mask.dim() == 1:
        valid_mask = valid_mask.unsqueeze(0)

    f_n = (forces * normals).sum(dim=-1)          # (B, M)
    f_t = forces - f_n.unsqueeze(-1) * normals    # (B, M, 3)
    f_t_norm = torch.norm(f_t, dim=-1)            # (B, M)

    # Closed cone: boundary is feasible. Tolerance absorbs ULP rounding
    # from cone_project placing forces exactly on the boundary.
    violating = (f_t_norm > mu * f_n + 1e-6) | (f_n < -1e-6)  # (B, M) bool

    valid_f = valid_mask.float()
    n_valid = valid_f.sum(dim=-1).clamp(min=1.0)   # (B,)
    n_violating = (violating.float() * valid_f).sum(dim=-1)  # (B,)

    return n_violating / n_valid  # (B,)


def friction_cone_violation_rate_local(
    force_coords: torch.Tensor,  # (B, M, 3), ordered (f_t1, f_t2, f_n)
    valid_mask: torch.Tensor,    # (B, M) bool
    mu: float = 0.5,
) -> torch.Tensor:              # (B,)
    """Fraction of valid contacts violating the local-coordinate cone."""
    if force_coords.dim() == 2:
        force_coords = force_coords.unsqueeze(0)
    if valid_mask.dim() == 1:
        valid_mask = valid_mask.unsqueeze(0)

    f_t_norm = torch.norm(force_coords[..., :2], dim=-1)
    f_n = force_coords[..., 2]
    violating = (f_t_norm > mu * f_n + 1e-6) | (f_n < -1e-6)
    valid_f = valid_mask.float()
    n_valid = valid_f.sum(dim=-1).clamp(min=1.0)
    return (violating.float() * valid_f).sum(dim=-1) / n_valid
