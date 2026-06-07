"""
Grasp map and wrench-balance utilities.

The grasp map G ∈ ℝ^{6 x 3M} maps contact force vectors to object wrenches:

    w = G @ f_vec

where f_vec = [f_1; f_2; ...; f_M] ∈ ℝ^{3M} stacks all contact forces and
w = [F; τ] ∈ ℝ^6 is the net wrench on the object.

For point-contact-with-friction (general 3-DOF force at each contact):

    G[:, 3i:3i+3] = [[I_3],
                     [skew(p_i)]]

so the contribution of contact i is [f_i; p_i x f_i].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skew(v: torch.Tensor) -> torch.Tensor:
    """Batch skew-symmetric (cross-product) matrix.

    Args:
        v: (..., 3)

    Returns:
        S: (..., 3, 3) such that S @ u == v x u
    """
    z = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    row0 = torch.stack([ z,  -vz,  vy], dim=-1)
    row1 = torch.stack([ vz,   z, -vx], dim=-1)
    row2 = torch.stack([-vy,  vx,   z], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)  # (..., 3, 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_grasp_map(
    contact_points: torch.Tensor,   # (B, M, 3) or (M, 3)
    contact_normals: torch.Tensor,  # (B, M, 3) or (M, 3)  [not used in general form]
) -> torch.Tensor:                  # (B, 6, 3M)
    """Build the grasp map G (general point-contact-with-friction model).

    Each contact contributes 3 force DOF.  Column block 3i:3i+3 is::

        G_i = [  I_3    ]   (force rows)
              [ skew(p_i)]   (torque rows)

    so that  w = G @ vec(F)  gives the net wrench.

    Args:
        contact_points:  (B, M, 3) contact positions (world frame)
        contact_normals: (B, M, 3) inward unit normals [not used in G itself,
                         included for API consistency]

    Returns:
        G: (B, 6, 3M) grasp map
    """
    if contact_points.dim() == 2:
        contact_points = contact_points.unsqueeze(0)
    if contact_normals.dim() == 2:
        contact_normals = contact_normals.unsqueeze(0)

    B, M, _ = contact_points.shape
    dtype, device = contact_points.dtype, contact_points.device

    G = torch.zeros(B, 6, 3 * M, dtype=dtype, device=device)
    I3 = torch.eye(3, dtype=dtype, device=device)

    S = _skew(contact_points)  # (B, M, 3, 3)

    for i in range(M):
        col = slice(3 * i, 3 * i + 3)
        G[:, :3, col] = I3.unsqueeze(0).expand(B, 3, 3)  # force
        G[:, 3:, col] = S[:, i, :, :]                     # torque = p_i x f_i

    return G


def wrench_balance_residual(
    contact_points: torch.Tensor,   # (B, M, 3)
    contact_normals: torch.Tensor,  # (B, M, 3)
    forces: torch.Tensor,           # (B, M, 3)
    valid_mask: torch.Tensor,       # (B, M) bool
    w_ext: torch.Tensor | None = None,  # (B, 6) external wrench
    object_mass: float = 0.2,
    torque_scale: torch.Tensor | float | None = None,
) -> torch.Tensor:                  # (B,)
    """‖G(C) @ vec(F) + w_ext‖₂ per sample.

    Args:
        contact_points:  (B, M, 3) contact positions
        contact_normals: (B, M, 3) inward unit normals
        forces:          (B, M, 3) contact forces
        valid_mask:      (B, M) boolean mask (invalid contacts are zeroed out)
        w_ext:           (B, 6) external wrench; defaults to gravity
                         [0, 0, -m·g, 0, 0, 0]
        object_mass:     mass in kg (used only when w_ext is None)
        torque_scale:    object length scale used to compare force and torque
                 units. If omitted, torque is left unscaled.

    Returns:
        residual: (B,) ℓ2 norm of wrench imbalance per sample
    """
    if contact_points.dim() == 2:
        contact_points = contact_points.unsqueeze(0)
    if contact_normals.dim() == 2:
        contact_normals = contact_normals.unsqueeze(0)
    if forces.dim() == 2:
        forces = forces.unsqueeze(0)
    if valid_mask.dim() == 1:
        valid_mask = valid_mask.unsqueeze(0)

    B, M, _ = forces.shape
    dtype, device = forces.dtype, forces.device

    G = compute_grasp_map(contact_points, contact_normals)  # (B, 6, 3M)

    # Zero out invalid contacts
    mask = valid_mask.float().unsqueeze(-1).expand_as(forces)  # (B, M, 3)
    f_masked = forces * mask
    f_vec = f_masked.reshape(B, 3 * M)  # (B, 3M)

    wrench = torch.bmm(G, f_vec.unsqueeze(-1)).squeeze(-1)  # (B, 6)

    if w_ext is None:
        w_ext = torch.zeros(B, 6, dtype=dtype, device=device)
        w_ext[:, 2] = -object_mass * 9.81  # gravity along -z

    residual = wrench + w_ext
    if torque_scale is not None:
        if not torch.is_tensor(torque_scale):
            torque_scale = torch.tensor(torque_scale, dtype=dtype, device=device)
        torque_scale = torque_scale.to(dtype=dtype, device=device).clamp(min=1e-6)
        while torque_scale.dim() < 2:
            torque_scale = torque_scale.unsqueeze(-1)
        residual = torch.cat([residual[:, :3], residual[:, 3:] / torque_scale], dim=-1)

    return torch.norm(residual, dim=-1)  # (B,)
