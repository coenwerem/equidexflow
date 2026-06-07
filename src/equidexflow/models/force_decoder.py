"""SO(3)-equivariant force decoder with contact conditioning and cone projection.

Per-finger force vectors are predicted by tiling global VN-DGCNN features with
each finger's predicted contact position (as an additional equivariant vector
channel), then running per-finger VN layers. The contact channel gives the
decoder spatial grounding — without it, the global features are identical for
all fingers and the network converges to a uniform radial-push solution.

Forces are projected into the Coulomb friction cone post-hoc.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from equidexflow.models.vn_layers import VNLinear, VNLinearLeakyReLU


def cone_project(
    f_raw: torch.Tensor,    # (B, M, 3) raw force vectors
    normals: torch.Tensor,  # (B, M, 3) estimated inward unit normals
    mu: float = 0.5,
) -> torch.Tensor:          # (B, M, 3) forces inside friction cone
    """Project force vectors into the Coulomb friction cone.

    Guarantees f_n >= 0 (compressive) and ||f_t|| <= mu * f_n by construction.
    """
    f_dot_n = (f_raw * normals).sum(dim=-1, keepdim=True)   # (B, M, 1)
    f_n = F.softplus(f_dot_n)                                # compressive

    f_t = f_raw - f_dot_n * normals                          # (B, M, 3)
    f_t_norm = f_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    scale = torch.clamp(mu * f_n / f_t_norm, max=1.0)
    return f_n * normals + scale * f_t


# ---------------------------------------------------------------------------
# Legacy helpers (used by eval scripts, kept for backward compatibility)
# ---------------------------------------------------------------------------

def contact_frame_from_normals(normals: torch.Tensor) -> torch.Tensor:
    """Build a deterministic contact frame B=[t1,t2,n] from unit normals."""
    n_hat = F.normalize(normals + 1e-8, dim=-1)
    z_axis = torch.zeros_like(n_hat)
    z_axis[..., 2] = 1.0
    y_axis = torch.zeros_like(n_hat)
    y_axis[..., 1] = 1.0
    ref = torch.where(n_hat[..., 2:3].abs() < 0.9, z_axis, y_axis)
    t1 = F.normalize(torch.cross(ref, n_hat, dim=-1), dim=-1, eps=1e-8)
    t2 = torch.cross(n_hat, t1, dim=-1)
    return torch.stack([t1, t2, n_hat], dim=-1)


def local_to_global_forces(force_coords: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Convert local force coordinates alpha to object-frame force vectors."""
    frames = contact_frame_from_normals(normals)
    return torch.matmul(frames, force_coords.unsqueeze(-1)).squeeze(-1)


def global_to_local_forces(forces: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Convert object-frame force vectors to local contact-frame coordinates."""
    frames = contact_frame_from_normals(normals)
    return torch.matmul(frames.transpose(-1, -2), forces.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Normal decoder
# ---------------------------------------------------------------------------

class NormalDecoder(nn.Module):
    """SO(3)-equivariant per-finger surface normal prediction with contact conditioning.

    Architecture
    ------------
    1. Tile global features (B, C, 3) to per-finger (B, n_f, C, 3)
    2. Concatenate predicted contact position as +1 equivariant channel -> (B, n_f, C+1, 3)
    3. Reshape to (B*n_f, C+1, 3) and run per-finger VN layers:
       VNLinearLeakyReLU(C+1, hidden) -> VNLinear(hidden, 1) -> (B*n_f, 1, 3)
    4. Reshape to (B, n_f, 3) and L2-normalize to unit vectors
    """

    def __init__(
        self,
        in_channels: int = 341,
        hidden_channels: int = 64,
        n_fingers: int = 4,
    ) -> None:
        super().__init__()
        self.n_fingers = n_fingers
        self.vn_hidden = VNLinearLeakyReLU(
            in_channels + 1, hidden_channels, dim=4, use_bn=False
        )
        self.vn_out = VNLinear(hidden_channels, 1)

    def forward(
        self,
        features: torch.Tensor,            # (B, C, 3)
        contact_positions: torch.Tensor,    # (B, n_fingers, 3)
    ) -> torch.Tensor:
        if features.dim() == 4:
            features = features.squeeze(-1)

        B, C, _ = features.shape
        nf = self.n_fingers

        z_per = features.unsqueeze(1).expand(B, nf, C, 3)    # (B, nf, C, 3)
        c_chan = contact_positions.unsqueeze(2)                 # (B, nf, 1, 3)
        z_cond = torch.cat([z_per, c_chan], dim=2)             # (B, nf, C+1, 3)

        z_flat = z_cond.reshape(B * nf, C + 1, 3)
        x = self.vn_hidden(z_flat)                             # (B*nf, hidden, 3)
        n_raw = self.vn_out(x).squeeze(1)                      # (B*nf, 3)
        n_raw = n_raw.reshape(B, nf, 3)                        # (B, nf, 3)

        return F.normalize(n_raw, dim=-1)


# ---------------------------------------------------------------------------
# Force decoder
# ---------------------------------------------------------------------------

class ForceDecoder(nn.Module):
    """SO(3)-equivariant per-finger force prediction with contact conditioning.

    Architecture
    ------------
    1. Tile global features (B, C, 3) to per-finger (B, n_f, C, 3)
    2. Concatenate contact positions as +1 equivariant channel -> (B, n_f, C+1, 3)
    3. Reshape to (B*n_f, C+1, 3) and run through per-finger VN layers:
       VNLinearLeakyReLU(C+1, hidden) -> VNLinear(hidden, 1) -> (B*n_f, 1, 3)
    4. Reshape to (B, n_f, 3) and project into friction cone
    """

    def __init__(
        self,
        in_channels: int = 341,
        hidden_channels: int = 64,
        n_fingers: int = 4,
        mu: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_fingers = n_fingers
        self.mu = mu

        self.vn_hidden = VNLinearLeakyReLU(
            in_channels + 1, hidden_channels, dim=4, use_bn=False
        )
        self.vn_out = VNLinear(hidden_channels, 1)

    def forward(
        self,
        features: torch.Tensor,          # (B, C, 3) or (B, C, 3, 1)
        contact_positions: torch.Tensor,  # (B, n_fingers, 3)
        contact_normals: torch.Tensor,    # (B, n_fingers, 3)
        return_local: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        if features.dim() == 4:
            features = features.squeeze(-1)

        B, C, _ = features.shape
        nf = self.n_fingers

        # Tile global features per finger and concatenate contact position
        z_per = features.unsqueeze(1).expand(B, nf, C, 3)    # (B, nf, C, 3)
        c_chan = contact_positions.unsqueeze(2)                # (B, nf, 1, 3)
        z_cond = torch.cat([z_per, c_chan], dim=2)            # (B, nf, C+1, 3)

        # Per-finger VN path
        z_flat = z_cond.reshape(B * nf, C + 1, 3)
        x = self.vn_hidden(z_flat)                            # (B*nf, hidden, 3)
        f_raw = self.vn_out(x).squeeze(1)                     # (B*nf, 3)
        f_raw = f_raw.reshape(B, nf, 3)                       # (B, nf, 3)

        forces = cone_project(f_raw, contact_normals, self.mu)

        if return_local:
            return forces, None
        return forces
