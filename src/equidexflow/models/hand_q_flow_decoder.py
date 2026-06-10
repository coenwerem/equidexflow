"""
Conditional RealNVP normalizing flow for hand joint angles.

Replaces the deterministic FiLM MLP with a generative model that handles
multimodal grasp distributions - multiple valid finger configurations exist
for the same object, and MSE averages them into a single closed-fist mean.

Architecture
------------
  conditioning : z_inv (SO(3)-invariant norms) + wrist (R_flat|t) -> cond vec
  flow         : n_coupling_layers affine coupling layers in logit space
  bounds       : sigmoid maps flow output to per-joint [lower, upper]

Training: log_prob(hand_q_gt, z, wrist_pose) -> NLL loss
Inference: sample(z, wrist_pose) -> diverse joint angle samples
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


WRIST_FEAT_DIM = 12


def _wrist_to_features(wrist_pose: torch.Tensor) -> torch.Tensor:
    R_flat = wrist_pose[:, :3, :3].reshape(wrist_pose.shape[0], 9)
    t = wrist_pose[:, :3, 3]
    return torch.cat([R_flat, t], dim=-1)


class _CouplingLayer(nn.Module):
    """Affine coupling layer with FiLM-modulated conditioning.

    The conditioning vector multiplicatively modulates the hidden
    representation via FiLM (gamma/beta) after the first hidden layer.
    This prevents the network from learning to zero out the conditioning
    signal, which happens with simple concatenation.
    """

    def __init__(
        self,
        dim: int,
        mask: torch.Tensor,
        cond_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        n_masked = int(mask.sum().item())
        n_transform = dim - n_masked

        self.fc1 = nn.Linear(n_masked + cond_dim, hidden_dim)
        self.film_gamma = nn.Linear(cond_dim, hidden_dim)
        self.film_beta = nn.Linear(cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 2 * n_transform)

        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
        nn.init.ones_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.bias)

    def _st(
        self, x_half: torch.Tensor, cond: torch.Tensor,
    ) -> torch.Tensor:
        h = torch.relu(self.fc1(torch.cat([x_half, cond], dim=-1)))
        gamma = self.film_gamma(cond)
        beta = self.film_beta(cond)
        h = h * gamma + beta
        h = torch.relu(self.fc2(h))
        return self.fc_out(h)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalizing direction (data -> base). Returns (z, log_det)."""
        mask = self.mask.bool()
        x_masked = x[:, mask]
        x_unmasked = x[:, ~mask]

        st = self._st(x_masked, cond)
        s, t = st.chunk(2, dim=-1)
        s = s.clamp(-5, 5)

        z_unmasked = x_unmasked * torch.exp(s) + t
        log_det = s.sum(dim=-1)

        z = torch.empty_like(x)
        z[:, mask] = x_masked
        z[:, ~mask] = z_unmasked
        return z, log_det

    def inverse(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Generative direction (base -> data)."""
        mask = self.mask.bool()
        z_masked = z[:, mask]
        z_unmasked = z[:, ~mask]

        st = self._st(z_masked, cond)
        s, t = st.chunk(2, dim=-1)
        s = s.clamp(-5, 5)

        x_unmasked = (z_unmasked - t) * torch.exp(-s)

        x = torch.empty_like(z)
        x[:, mask] = z_masked
        x[:, ~mask] = x_unmasked
        return x


class HandQFlowDecoder(nn.Module):
    """Conditional RealNVP flow for hand_q prediction.

    Inputs (same interface as HandQDecoder):
      z          : (B, C, 3) raw VN-DGCNN equivariant features
      wrist_pose : (B, 4, 4) SE(3) wrist pose

    Outputs:
      forward()  : (B, hand_dof) - mode (z=0) for FK collision during training
      log_prob() : (B,) - per-sample log-likelihood for NLL loss
      sample()   : (B, hand_dof) - stochastic sample for inference
    """

    ALLEGRO_LOWER = [
        -0.470, -0.196, -0.174, -0.227,
        -0.470, -0.196, -0.174, -0.227,
        -0.470, -0.196, -0.174, -0.227,
         0.263, -0.105, -0.189, -0.162,
    ]
    ALLEGRO_UPPER = [
         0.470,  1.610,  1.709,  1.618,
         0.470,  1.610,  1.709,  1.618,
         0.470,  1.610,  1.709,  1.618,
         1.396,  1.163,  1.644,  1.719,
    ]

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        hand_dof: int = 16,
        wrist_dim: int = WRIST_FEAT_DIM,
        n_coupling_layers: int = 8,
        cond_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hand_dof = hand_dof
        self.cond_dim = cond_dim

        # --- Conditioning network ---
        self.feat_norm = nn.LayerNorm(in_dim)
        self.encoder_proj = nn.Sequential(
            nn.Linear(in_dim, cond_dim), nn.ReLU(inplace=True),
        )
        self.wrist_proj = nn.Sequential(
            nn.Linear(wrist_dim, cond_dim), nn.ReLU(inplace=True),
            nn.Linear(cond_dim, cond_dim), nn.ReLU(inplace=True),
        )
        self.fuse_gamma = nn.Linear(cond_dim, cond_dim)
        self.fuse_beta = nn.Linear(cond_dim, cond_dim)

        # --- Coupling layers with alternating masks ---
        masks = []
        for i in range(n_coupling_layers):
            mask = torch.zeros(hand_dof)
            if i % 2 == 0:
                mask[::2] = 1.0
            else:
                mask[1::2] = 1.0
            masks.append(mask)

        self.layers = nn.ModuleList([
            _CouplingLayer(hand_dof, masks[i], cond_dim, hidden_dim)
            for i in range(n_coupling_layers)
        ])

        # --- Joint limits ---
        self.register_buffer(
            "joint_lower",
            torch.tensor(self.ALLEGRO_LOWER[:hand_dof], dtype=torch.float32),
        )
        self.register_buffer(
            "joint_upper",
            torch.tensor(self.ALLEGRO_UPPER[:hand_dof], dtype=torch.float32),
        )

    def _condition(
        self, z: torch.Tensor, wrist_pose: torch.Tensor,
    ) -> torch.Tensor:
        if z.dim() == 4:
            z = z.squeeze(-1)
        z_inv = z.norm(dim=-1)
        z_inv = self.feat_norm(z_inv)
        z_proj = self.encoder_proj(z_inv)
        w_feat = _wrist_to_features(wrist_pose)
        w_proj = self.wrist_proj(w_feat)
        gamma = self.fuse_gamma(w_proj)
        beta = self.fuse_beta(w_proj)
        return z_proj * (1.0 + gamma) + beta

    def _to_logit(self, hand_q: torch.Tensor) -> torch.Tensor:
        u = (hand_q - self.joint_lower) / (self.joint_upper - self.joint_lower)
        u = u.clamp(1e-6, 1.0 - 1e-6)
        return torch.logit(u)

    def _from_logit(self, x: torch.Tensor) -> torch.Tensor:
        return self.joint_lower + (self.joint_upper - self.joint_lower) * torch.sigmoid(x)

    def _log_det_logit(self, hand_q: torch.Tensor) -> torch.Tensor:
        """Log |dx/dq| for the logit change-of-variables."""
        u = (hand_q - self.joint_lower) / (self.joint_upper - self.joint_lower)
        u = u.clamp(1e-6, 1.0 - 1e-6)
        return (
            -torch.log(self.joint_upper - self.joint_lower)
            - torch.log(u)
            - torch.log(1.0 - u)
        ).sum(dim=-1)

    def log_prob(
        self,
        hand_q: torch.Tensor,
        z: torch.Tensor,
        wrist_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Log p(hand_q | z, wrist_pose). Returns (B,)."""
        cond = self._condition(z, wrist_pose)
        x = self._to_logit(hand_q)

        log_det_flow = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, ld = layer(x, cond)
            log_det_flow = log_det_flow + ld

        log_p_base = -0.5 * (
            x.pow(2).sum(dim=-1) + self.hand_dof * math.log(2.0 * math.pi)
        )

        log_det_bounds = self._log_det_logit(hand_q)

        return log_p_base + log_det_flow + log_det_bounds

    def forward(
        self,
        z: torch.Tensor,
        wrist_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Mode prediction (z=0 in base space). Used for FK collision in training."""
        cond = self._condition(z, wrist_pose)
        x = torch.zeros(cond.shape[0], self.hand_dof, device=cond.device)
        for layer in reversed(self.layers):
            x = layer.inverse(x, cond)
        return self._from_logit(x)

    def sample(
        self,
        z: torch.Tensor,
        wrist_pose: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Stochastic sample. Returns (B, hand_dof)."""
        cond = self._condition(z, wrist_pose)
        x = torch.randn(cond.shape[0], self.hand_dof, device=cond.device) * temperature
        for layer in reversed(self.layers):
            x = layer.inverse(x, cond)
        return self._from_logit(x)
