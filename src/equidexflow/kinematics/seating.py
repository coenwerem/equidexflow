"""Seat a decoded grasp onto the object (test-time optimization).

The decoder's contacts are surface-accurate, but the sampled wrist + ``hand_q``
do not realize them -- the raw fingertips can sit centimetres off the object.
:func:`seat_grasp` closes that gap in task space by optimizing the wrist (6 DOF,
6D-rotation parameterization) **and** ``hand_q`` jointly so the FK fingertips
land on the predicted contacts, under a trust region (stay near the sampled
wrist), a joint-limit barrier, and an optional mesh-penetration term.

Pure torch (autograd through :class:`AllegroRightHandFK`), framework-agnostic.
This is the same routine the paper's figure pipeline uses; it is far stronger
than a wrist-only adjustment, which cannot bring four fingertips to four
contacts with a single rigid move.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _rot6d_to_R(x: torch.Tensor) -> torch.Tensor:
    """(...,6) -> (...,3,3) via Gram-Schmidt (differentiable)."""
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _R_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def seat_grasp(
    fk,
    hand_q: torch.Tensor,       # (B, hand_dof)
    wrist: torch.Tensor,        # (B, 4, 4) base-frame wrist
    contacts: torch.Tensor,     # (B, n_fingers, 3) target contacts (world)
    lo: torch.Tensor,           # (hand_dof,) joint lower limits
    hi: torch.Tensor,           # (hand_dof,) joint upper limits
    mesh_pts: torch.Tensor | None = None,   # (M, 3) object surface points (world)
    mesh_nrm: torch.Tensor | None = None,   # (M, 3) outward normals
    n_steps: int = 200,
    lr: float = 0.02,
    trust_w: float = 0.05,
    pen_w: float = 50.0,
    selfcoll_w: float = 0.0,
    tip_margin: float = 0.002,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (wrist (B,4,4), hand_q (B,hand_dof)) seated onto the contacts.

    Inputs/outputs are in the base (palm) frame -- the same convention as
    :meth:`EquiDexFlow.sample`. Detached.
    """
    device = hand_q.device
    B = hand_q.shape[0]
    nf = fk.N_FINGERS
    lo = lo.to(device)
    hi = hi.to(device)

    rot6d = _R_to_rot6d(wrist[:, :3, :3]).detach().clone().requires_grad_(True)
    trans = wrist[:, :3, 3].detach().clone().requires_grad_(True)
    q = hand_q.detach().clone().requires_grad_(True)
    rot0, trans0 = rot6d.detach().clone(), trans.detach().clone()
    opt = torch.optim.Adam([rot6d, trans, q], lr=lr)
    eye = torch.eye(4, device=device).repeat(B, 1, 1)

    with torch.enable_grad():
        for _ in range(n_steps):
            opt.zero_grad()
            W = eye.clone()
            W[:, :3, :3] = _rot6d_to_R(rot6d)
            W[:, :3, 3] = trans
            sph, radii = fk.forward_all_spheres(q, W)
            tips = sph[:, -nf:]

            reach = ((tips - contacts) ** 2).sum(-1).mean()
            trust = (((trans - trans0) ** 2).sum(-1).mean()
                     + ((rot6d - rot0) ** 2).sum(-1).mean())
            lim = (torch.relu(q - hi) ** 2 + torch.relu(lo - q) ** 2).sum(-1).mean()
            loss = reach + trust_w * trust + 10.0 * lim

            if selfcoll_w > 0.0:
                tipr = fk.fingertip_radius
                tipr = tipr.mean() if torch.is_tensor(tipr) else float(tipr)
                dmat = torch.cdist(tips, tips)
                ovr = torch.relu(2.0 * tipr - dmat)
                ovr = ovr.masked_fill(
                    torch.eye(nf, device=device, dtype=torch.bool), 0.0
                )
                loss = loss + selfcoll_w * 0.5 * (ovr ** 2).sum(dim=(1, 2)).mean()

            if mesh_pts is not None and mesh_nrm is not None:
                dsq = (sph.unsqueeze(2) - mesh_pts.view(1, 1, -1, 3)).pow(2).sum(-1)
                ni = dsq.argmin(-1)
                signed = ((sph - mesh_pts[ni]) * mesh_nrm[ni]).sum(-1)  # >0 outside
                margins = radii.clone()
                margins[-nf:] = (margins[-nf:] - tip_margin).clamp(min=0.0)
                pen = torch.relu(margins.view(1, -1) - signed).pow(2).sum(-1).mean()
                loss = loss + pen_w * pen

            loss.backward()
            opt.step()

    with torch.no_grad():
        W = eye.clone()
        W[:, :3, :3] = _rot6d_to_R(rot6d)
        W[:, :3, 3] = trans
        qf = q.clamp(lo, hi)
    return W.detach(), qf.detach()
