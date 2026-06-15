"""Seat a decoded grasp onto the object (test-time optimization).

The decoder's contacts are surface-accurate, but the sampled wrist + ``hand_q``
do not realize them -- the raw fingertips can sit centimetres off the object.
:func:`seat_grasp` closes that gap in task space by optimizing the wrist (6 DOF,
6D-rotation parameterization) **and** ``hand_q`` jointly so the FK fingertips
land on the predicted contacts, under a trust region (stay near the sampled
wrist), a joint-limit barrier, and an optional mesh-penetration term.

Pure torch (autograd through :class:`AllegroRightHandFK`), framework-agnostic.
This is the same routine the paper's figure pipeline uses. It is far stronger
than a wrist-only adjustment, which cannot bring four fingertips to four
contacts with a single rigid move.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from equidexflow.kinematics.collision import (
    inter_finger_clustering_penalty,
    link_self_collision_penalty,
)


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
    coll_points_fn=None,                    # (hand_q, wrist) -> (B, K, 3) world pts
    n_steps: int = 200,
    lr: float = 0.02,
    trust_w: float = 0.05,
    pen_w: float = 50.0,
    selfcoll_w: float = 0.0,
    linkcoll_w: float = 5.0,
    cluster_w: float = 2.0,
    link_clearance: float = 0.002,
    personal_space: float = 0.015,
    tip_margin: float = 0.002,
    reach_gate_tau: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (wrist (B,4,4), hand_q (B,hand_dof)) seated onto the contacts.

    Inputs/outputs are in the base (palm) frame -- the same convention as
    :meth:`EquiDexFlow.sample`. Detached.

    ``contacts`` are the per-finger reach targets for the fingertip *centers*.
    If the caller wants the pad surface (not the center) to meet a surface point,
    it should offset the targets outward by the pad radius along the surface
    normal *before* calling -- otherwise the reach term (center -> surface) fights
    the penetration term (which holds the hand outside the surface).

    Self-collision / clustering: ``linkcoll_w`` (>0 by default) penalizes
    overlapping finger-link *capsules* (full phalanges, adjacent same-finger
    links excluded) and ``cluster_w`` penalizes fingers bunching together via a
    per-finger-pair minimum-distance hinge with margin ``personal_space``. Both
    act on the realized FK at every step, so seating actively separates fingers.
    The legacy ``selfcoll_w`` (fingertip pairs only, default 0) is subsumed by
    ``linkcoll_w`` and kept off for back-compat.

    Penetration term: with ``coll_points_fn`` (preferred) the SDF penalty runs
    over a dense full-hand point cloud (palm + every link), so the *whole* hand is
    pushed out -- not just the sparse ``forward_all_spheres`` markers. Falls back
    to those spheres when only ``mesh_pts``/``mesh_nrm`` are given.

    ``reach_gate_tau`` (EXPERIMENTAL, default 0 = off): when > 0 and
    ``coll_points_fn`` exposes ``finger_ids``, each finger's reach weight is scaled
    by ``1/(1+(pen_f/tau)^2)`` so a finger that can only reach by penetrating
    relaxes and is pushed to the surface. In practice the coupled wrist+hand_q
    optimizer tends to *redistribute* rather than remove penetration (the root
    cause is an infeasible predicted contact, not seating), so this is off by
    default. Selection (GraspScorer) is the reliable lever. Kept as a knob.
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

            # --- Full-hand penetration (computed first: per-finger depth gates
            #     the reach term below).
            pen_loss = None
            per_finger_w = None
            if coll_points_fn is not None and mesh_pts is not None and mesh_nrm is not None:
                pts_w = coll_points_fn(q, W)                              # (B, K, 3)
                dmat = torch.cdist(pts_w, mesh_pts.unsqueeze(0).expand(B, -1, -1))
                ni = dmat.argmin(-1)                                      # (B, K)
                signed = ((pts_w - mesh_pts[ni]) * mesh_nrm[ni]).sum(-1)  # (B,K) >0 out
                pen_loss = torch.relu(-signed).pow(2).mean()

                fids = getattr(coll_points_fn, "finger_ids", None)
                if fids is not None and reach_gate_tau and reach_gate_tau > 0.0:
                    fids = fids.to(signed.device)
                    depth = torch.relu(-signed).detach()                 # (B,K)
                    pen_f = torch.zeros(B, nf, device=signed.device, dtype=signed.dtype)
                    for f in range(nf):
                        m = fids == f
                        if bool(m.any()):
                            pen_f[:, f] = depth[:, m].max(dim=1).values
                    # A finger that can only reach by penetrating relaxes its reach
                    # so the penetration term pushes it to the surface instead.
                    per_finger_w = 1.0 / (1.0 + (pen_f * 1000.0 / reach_gate_tau) ** 2)

            if per_finger_w is not None:
                reach = (per_finger_w * ((tips - contacts) ** 2).sum(-1)).mean()
            else:
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

            # --- Link-link self-collision + inter-finger clustering, on the
            #     full per-link capsule geometry (not just the fingertips).
            if linkcoll_w > 0.0 or cluster_w > 0.0:
                seg_a, seg_b, caps_r = fk.forward_link_capsules(q, W)
                if linkcoll_w > 0.0:
                    loss = loss + linkcoll_w * link_self_collision_penalty(
                        seg_a, seg_b, caps_r, fk._caps_pair_mask,
                        clearance=link_clearance,
                    ).mean()
                if cluster_w > 0.0:
                    loss = loss + cluster_w * inter_finger_clustering_penalty(
                        seg_a, seg_b, caps_r, fk._caps_finger,
                        personal_space=personal_space,
                    ).mean()

            if pen_loss is not None:
                loss = loss + pen_w * pen_loss
            elif mesh_pts is not None and mesh_nrm is not None:
                # Fallback: sparse collision-sphere SDF (phalanges + tips only).
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
