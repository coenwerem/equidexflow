"""
HandQDecoder with FiLM (Feature-wise Linear Modulation) wrist conditioning.

Why FiLM instead of concat:
  Two prior attempts failed.
    v1 (concat z_flat | wrist_12)        -> encoder weights collapsed under
                                            SO(3) augmentation; hand_q std
                                            ~0.005 rad.
    v2 (concat ||z_c|| invariant | w12)  -> encoder pathway recovered (cross-
                                            object std up 17x to 0.085 rad)
                                            but per-sample std stuck at
                                            ~0.005 rad because 341 encoder
                                            dims dominate 12 wrist dims by
                                            ~34x in the first-layer arithmetic.

  Capacity-matching the wrist branch (12 -> 128) doesn't fix this: the loss
  is happy with a low-effort solution where the encoder explains per-object
  variance and the wrist does nothing. FiLM forces wrist into the computation
  by multiplying every hidden dimension with a wrist-derived scale and adding
  a wrist-derived shift. Setting scale=0/shift=0 has worse gradient flow than
  using the wrist, so the optimizer can't ignore it.

Architecture
------------
  encoder_proj : Linear(C, hidden)                    encoder pathway
  film         : Linear(12, 2*hidden)                 (scale | shift) generator
  mlp          : ReLU -> Linear(hidden, hidden) -> ReLU -> Linear(hidden, dof)
  output       : scaled sigmoid to per-joint Allegro limits

The FiLM layer is small-initialized so the network starts approximately
in the "ignore wrist" regime (output close to encoder-only) and learns to
use the wrist gradually - avoids early-training instability.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Wrist conditioning dim: R (3x3) flattened + t (3) = 12 floats per pose.
WRIST_FEAT_DIM = 12


def _wrist_to_features(wrist_pose: torch.Tensor) -> torch.Tensor:
    """Flatten (B, 4, 4) -> (B, 12): R flattened (9) + t (3)."""
    R_flat = wrist_pose[:, :3, :3].reshape(wrist_pose.shape[0], 9)
    t = wrist_pose[:, :3, 3]
    return torch.cat([R_flat, t], dim=-1)


class HandQDecoder(nn.Module):
    """FiLM-conditioned hand_q decoder.

    Inputs:
      z          : (B, C, 3) raw VN-DGCNN equivariant features
      wrist_pose : (B, 4, 4) SE(3) wrist pose (GT in training, ODE-sampled in inference)

    Output: (B, hand_dof) joint angles in [lower, upper] per joint.
    """

    ALLEGRO_LOWER = [
        -0.470, -0.196, -0.174, -0.227,  # index
        -0.470, -0.196, -0.174, -0.227,  # middle
        -0.470, -0.196, -0.174, -0.227,  # ring
         0.263, -0.105, -0.189, -0.162,  # thumb
    ]
    ALLEGRO_UPPER = [
         0.470,  1.610,  1.709,  1.618,  # index
         0.470,  1.610,  1.709,  1.618,  # middle
         0.470,  1.610,  1.709,  1.618,  # ring
         1.396,  1.163,  1.644,  1.719,  # thumb
    ]

    def __init__(
        self,
        in_dim: int,                    # number of VN channels C (e.g. 341)
        hidden_dim: int = 256,
        hand_dof: int = 16,
        wrist_dim: int = WRIST_FEAT_DIM,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.wrist_dim = wrist_dim
        self.hidden_dim = hidden_dim

        # LayerNorm on SO(3)-invariant per-channel norms so the encoder
        # pathway gets unit-variance input (raw norms are ~1e-3).
        self.feat_norm = nn.LayerNorm(in_dim)

        # Encoder pathway: invariant features -> hidden.
        self.encoder_proj = nn.Linear(in_dim, hidden_dim)

        # FiLM generator: wrist features -> (scale, shift) for the hidden state.
        self.film = nn.Linear(wrist_dim, 2 * hidden_dim)

        # Post-FiLM MLP.
        self.mlp = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hand_dof),
        )

        # Small init on the FiLM layer so we start close to (scale=0, shift=0),
        # i.e. encoder-only behavior. Gradual learning prevents early-training
        # instability where wrist noise overwhelms the encoder signal.
        nn.init.xavier_uniform_(self.film.weight, gain=0.1)
        nn.init.zeros_(self.film.bias)

        self.register_buffer(
            "joint_lower",
            torch.tensor(self.ALLEGRO_LOWER[:hand_dof], dtype=torch.float32),
        )
        self.register_buffer(
            "joint_upper",
            torch.tensor(self.ALLEGRO_UPPER[:hand_dof], dtype=torch.float32),
        )

    def forward(
        self,
        z: torch.Tensor,            # (B, C, 3)  NOT flattened
        wrist_pose: torch.Tensor,   # (B, 4, 4)
    ) -> torch.Tensor:
        if z.dim() == 4:
            z = z.squeeze(-1)

        # SO(3)-invariant features -> LayerNorm -> projection to hidden
        z_inv = z.norm(dim=-1)                  # (B, C)
        z_inv = self.feat_norm(z_inv)           # (B, C)
        h = self.encoder_proj(z_inv)            # (B, hidden)

        # FiLM modulation: every hidden dim is scaled+shifted by wrist
        wrist_feat = _wrist_to_features(wrist_pose)        # (B, 12)
        scale_shift = self.film(wrist_feat)                # (B, 2*hidden)
        scale, shift = scale_shift.chunk(2, dim=-1)        # each (B, hidden)
        h = h * (1.0 + scale) + shift

        raw = self.mlp(h)
        return self.joint_lower + (self.joint_upper - self.joint_lower) * torch.sigmoid(raw)
