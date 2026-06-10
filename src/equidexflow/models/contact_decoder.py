"""
ContactDecoder with wrist-rotation + per-finger conditioning.

The previous version had VNLinear(hidden, n_fingers) producing 4 projections
of the same global feature, anchored to the same wrist translation. With no
per-finger differentiation, all 4 contacts clustered within a few mm.

This version: tile the global features per finger, concatenate the wrist
rotation columns (3 equivariant vectors) and a learned per-finger embedding
(1 equivariant vector). Shared per-finger VN layers then produce a distinct
displacement per finger, anchored to the wrist translation.

Why this works:
  - Wrist rotation columns supply directional context, so the decoder knows
    which way "up", "forward", "side" point in the world relative to the hand.
  - Per-finger embedding breaks the symmetry between the 4 finger outputs.
  - Sharing weights across fingers (vs four independent heads) preserves
    parameter efficiency and lets the model leverage all training data for
    every finger.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from equidexflow.models.vn_layers import VNLinear, VNLinearLeakyReLU


class ContactDecoder(nn.Module):
    """Decodes per-fingertip contact positions from VN features + wrist pose.

    Input:
      features   : (B, C, 3)  encoder global features
      wrist_pose : (B, 4, 4)  current wrist SE(3)

    Output:
      contact_positions : (B, n_fingers, 3)  predicted contacts, object frame
      contact_logits    : (B, n_fingers)     per-finger confidence logits

    Per-finger VN path input layout (channel dim):
      C global feature channels + 3 wrist rotation columns + 1 finger embedding
      = C + 4
    Each of those is a 3-vector (last dim 3), so VN equivariance is preserved.
    """

    def __init__(
        self,
        in_channels: int = 341,
        hidden_channels: int = 64,
        n_fingers: int = 4,
    ) -> None:
        super().__init__()
        self.n_fingers = n_fingers

        # Per-finger VN path: shared weights, 1 output vector per finger.
        self.vn_hidden = VNLinearLeakyReLU(
            in_channels + 4, hidden_channels, dim=4, use_bn=False
        )
        self.vn_out = VNLinear(hidden_channels, 1)

        # Learned per-finger equivariant identity signal (small init so the
        # net starts close to the previous symmetric solution).
        self.finger_embed = nn.Parameter(torch.randn(n_fingers, 1, 3) * 0.1)

        # Scalar confidence head - eats SO(3)-invariant per-channel norms
        # (B, C) so it doesn't degenerate under augmentation.
        self.logit_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, n_fingers),
        )

    def forward(
        self,
        features: torch.Tensor,    # (B, C, 3) or (B, C, 3, 1)
        wrist_pose: torch.Tensor,  # (B, 4, 4)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.dim() == 4:
            features = features.squeeze(-1)

        B, C, _ = features.shape
        nf = self.n_fingers

        # Wrist rotation columns: 3 equivariant 3-vectors.
        wrist_R = wrist_pose[:, :3, :3]                 # (B, 3, 3) - 3 columns of R
        wrist_t = wrist_pose[:, :3, 3].unsqueeze(1)     # (B, 1, 3)

        # Per-finger conditioning: tile + concat.
        z_per = features.unsqueeze(1).expand(B, nf, C, 3)            # (B, nf, C, 3)
        R_per = wrist_R.unsqueeze(1).expand(B, nf, 3, 3)             # (B, nf, 3, 3)
        e_per = self.finger_embed.unsqueeze(0).expand(B, nf, 1, 3)   # (B, nf, 1, 3)
        z_cond = torch.cat([z_per, R_per, e_per], dim=2)             # (B, nf, C+4, 3)

        # Shared per-finger VN path.
        z_flat = z_cond.reshape(B * nf, C + 4, 3)
        x = self.vn_hidden(z_flat)                                   # (B*nf, hidden, 3)
        contact_per = self.vn_out(x).squeeze(1)                      # (B*nf, 3)
        contact_positions = contact_per.reshape(B, nf, 3)

        # Anchor displacements to the wrist origin.
        contact_positions = contact_positions + wrist_t              # (B, nf, 3)

        # Scalar confidence head on SO(3)-invariant per-channel norms.
        feat_inv = features.norm(dim=-1)                             # (B, C)
        contact_logits = self.logit_mlp(feat_inv)                    # (B, nf)

        return contact_positions, contact_logits
