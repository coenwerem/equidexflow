"""
EquiDexFlow (dexterous variant): SE(3)-equivariant flow-matching model for
dexterous grasp generation with the RealHand L6.

Generates the joint distribution  p(q, C, F | O)  where:
  q - wrist SE(3) pose (via flow backbone) + hand joint angles (11 DOF)
  C - fingertip contact positions (5 x 3)
  F - contact forces (5 x 3)

The flow backbone is identical to the original EquiDexFlow in equi_grasp_flow.py
(SE(3) ODE on wrist pose).  Three additional decoder heads branch off the
VN-DGCNN encoder features to predict hand_q, C, and F.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from equidexflow.models.equi_grasp_flow import get_traj


# ---------------------------------------------------------------------------
# Differentiable surface projection
# ---------------------------------------------------------------------------

def _project_to_surface(
    contacts: torch.Tensor,     # (B, nf, 3) predicted contacts
    surface_pts: torch.Tensor,  # (B, N, 3) object surface points
    temperature: float = 0.005,
    surface_normals: torch.Tensor | None = None,  # (B, N, 3) outward normals
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Snap predicted contacts to the nearest surface point (differentiable).

    Returns on-surface contacts and (optionally) interpolated inward normals.
    """
    dists = torch.cdist(contacts, surface_pts)            # (B, nf, N)
    weights = F.softmax(-dists / temperature, dim=-1)     # (B, nf, N)
    projected = torch.einsum('bfn,bnd->bfd', weights, surface_pts)

    normals_out = None
    if surface_normals is not None:
        normals_out = torch.einsum('bfn,bnd->bfd', weights, surface_normals)
        normals_out = -F.normalize(normals_out, dim=-1)   # inward normals

    return projected, normals_out


# ---------------------------------------------------------------------------
# Helper: aggregate MAX_CONTACTS contacts down to n_fingers per-finger means
# ---------------------------------------------------------------------------

def _aggregate_by_finger(
    data: torch.Tensor,       # (B, M, D)
    finger_ids: torch.Tensor, # (B, M) long, -1 for padding
    valid_mask: torch.Tensor, # (B, M) bool
    n_fingers: int = 4,
) -> torch.Tensor:            # (B, n_fingers, D)
    """Average contact-level data per finger using valid_mask."""
    B, M, D = data.shape
    device, dtype = data.device, data.dtype

    result = torch.zeros(B, n_fingers, D, device=device, dtype=dtype)
    counts = torch.zeros(B, n_fingers, device=device, dtype=dtype)

    for f in range(n_fingers):
        mask = (finger_ids == f) & valid_mask           # (B, M)
        masked = data * mask.unsqueeze(-1).to(dtype)    # (B, M, D)
        result[:, f] = masked.sum(dim=1)
        counts[:, f] = mask.to(dtype).sum(dim=1)

    counts = counts.clamp(min=1.0)
    return result / counts.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class EquiDexFlow(nn.Module):
    """SE(3)-equivariant flow-matching model for dexterous grasps.

    Generates the joint distribution  p(q, C, F | O)  where:
      q = wrist SE(3) pose (flow backbone) + hand joint angles (11 DOF, FK-grounded decoder)
      C = fingertip contact positions (5 x 3)
      F = contact forces (5 x 3)

    Parameters
    ----------
    encoder        : VNDGCNNEncoder - (B,3,N)  ->  (B,C,3)
    vector_field   : VNVectorFields - SE(3) velocity field for wrist flow
    ode_solver     : SE3_Euler | SE3_RK4_MK
    contact_decoder: ContactDecoder
    force_decoder  : ForceDecoder
    hand_q_decoder : HandQDecoder - (B,C*3)  ->  (B,11)
    n_fingers      : int (default 5)
    hand_dof       : int (default 11)
    p_uncond       : classifier-free guidance drop probability
    guidance       : guidance scale
    init_dist      : callable(n, device)  ->  (n,4,4) initial SE(3) distribution
    """

    def __init__(
        self,
        encoder: nn.Module,
        vector_field: nn.Module,
        ode_solver,
        contact_decoder: nn.Module,
        force_decoder: nn.Module,
        hand_q_decoder: nn.Module,
        normal_decoder: nn.Module | None = None,
        n_fingers: int = 4,
        hand_dof: int = 16,
        p_uncond: float = 0.1,
        guidance: float = 2.0,
        init_dist=None,
        surface_proj_tau: float = 0.005,
        wrist_frame: str = "base",
        hand: str = "allegro",
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.vector_field = vector_field
        self.ode_solver = ode_solver
        self.contact_decoder = contact_decoder
        self.force_decoder = force_decoder
        self.hand_q_decoder = hand_q_decoder
        self.normal_decoder = normal_decoder
        self.n_fingers = n_fingers
        self.hand_dof = hand_dof
        self.p_uncond = p_uncond
        self.guidance = guidance
        self.init_dist = init_dist
        self.surface_proj_tau = surface_proj_tau
        self.wrist_frame = wrist_frame
        self.hand = hand

        # Differentiable forward kinematics. Allegro is the v1-complete path;
        # LEAP is wired here but its joint limits / wrist offsets are completed
        # when the machine-B model code is merged (see docs/MACHINE_B_HANDOFF.md).
        if hand == "leap":
            from equidexflow.kinematics.leap import LeapFK
            self.fk = LeapFK()
        else:
            from equidexflow.kinematics.allegro_fk import AllegroRightHandFK
            self.fk = AllegroRightHandFK()

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        """Training forward pass.

        Parameters
        ----------
        batch : dict from DexGraspDBDataset / GraspBatch schema
            Required keys: object_points (B,3,N), wrist_pose (B,4,4),
            hand_q (B,11), contacts (B,M,3), normals (B,M,3),
            forces (B,M,3), finger_ids (B,M), valid_mask (B,M)

        Returns
        -------
        dict containing scalar losses plus prediction tensors used by the
        trainer for differentiable physics regularization.
        """
        pc          = batch['object_points']  # (B, 3, N)
        wrist_pose  = batch['wrist_pose']     # (B, 4, 4)
        hand_q_gt   = batch['hand_q']         # (B, 11)
        contacts_gt = batch['contacts']       # (B, M, 3)
        normals_gt  = batch['normals']        # (B, M, 3)
        forces_gt   = batch['forces']         # (B, M, 3)
        finger_ids  = batch['finger_ids']     # (B, M) long
        valid_mask  = batch['valid_mask']     # (B, M) bool

        B      = pc.shape[0]
        device = pc.device

        # ---- SE(3) flow trajectory ----------------------------------------
        t   = torch.rand(B, 1, device=device)
        x_0 = self.init_dist(B, device)
        x_t, u_t = get_traj(x_0, wrist_pose, t)

        # ---- Encode point cloud -------------------------------------------
        z = self.encoder(pc)  # (B, C, 3)

        # ---- Classifier-free guidance masking -----------------------------
        mask_uncond = torch.bernoulli(
            torch.full((B,), self.p_uncond, device=device)
        ).bool()
        z_cond = z.clone()
        z_cond[mask_uncond] = 0.0

        # ---- Velocity field loss (SE(3) flow) -----------------------------
        v_t        = self.vector_field(z_cond, t, x_t)  # (B, 6)
        flow_loss  = F.mse_loss(v_t, u_t)

        # ---- Hand joint angle loss ----------------------------------------
        # Pass raw (B, C, 3) features so the decoder can compute SO(3)-invariant
        # per-channel norms internally; flattened features would be augmentation
        # noise.
        hand_q_pred = self.hand_q_decoder(z, wrist_pose)           # (B, hand_dof)
        if hasattr(self.hand_q_decoder, 'log_prob'):
            hand_q_loss = -self.hand_q_decoder.log_prob(
                hand_q_gt, z, wrist_pose,
            ).mean() / self.hand_dof
        else:
            hand_q_loss = F.mse_loss(hand_q_pred, hand_q_gt)

        # ---- Per-finger ground-truth aggregation --------------------------
        finger_contacts_gt = _aggregate_by_finger(
            contacts_gt, finger_ids, valid_mask, self.n_fingers
        )  # (B, n_f, 3)
        finger_normals_gt  = _aggregate_by_finger(
            normals_gt, finger_ids, valid_mask, self.n_fingers
        )  # (B, n_f, 3)
        finger_forces_gt   = _aggregate_by_finger(
            forces_gt, finger_ids, valid_mask, self.n_fingers
        )  # (B, n_f, 3)

        # Per-finger validity: True where at least one GT contact exists
        finger_valid = finger_contacts_gt.norm(dim=-1) > 1e-6  # (B, n_f)

        # ---- Contact prediction loss --------------------------------------
        contact_pred_raw, _ = self.contact_decoder(z, wrist_pose)  # (B, n_f, 3)

        # Differentiable surface projection: snap to nearest surface point.
        surface_pts = pc.transpose(1, 2)                          # (B, N, 3)
        surface_normals = batch.get('object_point_normals')       # (B, 3, N) or None
        if surface_normals is not None:
            surface_normals = surface_normals.transpose(1, 2)     # (B, N, 3)

        contact_pred, normals_from_surface = _project_to_surface(
            contact_pred_raw, surface_pts,
            temperature=self.surface_proj_tau,
            surface_normals=surface_normals,
        )

        if finger_valid.any():
            contact_loss = F.mse_loss(
                contact_pred[finger_valid], finger_contacts_gt[finger_valid]
            )
        else:
            contact_loss = torch.tensor(0.0, device=device)

        # ---- Normal prediction -------------------------------------------
        if normals_from_surface is not None:
            if finger_valid.any():
                cos_sim = F.cosine_similarity(
                    normals_from_surface, finger_normals_gt, dim=-1
                )
                normal_loss = 1.0 - cos_sim[finger_valid].mean()
            else:
                normal_loss = torch.tensor(0.0, device=device)
            normals_input = normals_from_surface
        elif self.normal_decoder is not None:
            normals_pred = self.normal_decoder(z, contact_pred)
            if finger_valid.any():
                cos_sim = F.cosine_similarity(
                    normals_pred, finger_normals_gt, dim=-1
                )
                normal_loss = 1.0 - cos_sim[finger_valid].mean()
            else:
                normal_loss = torch.tensor(0.0, device=device)
            normals_input = normals_pred
        else:
            object_centroid = pc.mean(dim=-1).unsqueeze(1)
            normals_input = F.normalize(
                object_centroid - contact_pred.detach() + 1e-8, dim=-1
            )
            normal_loss = torch.tensor(0.0, device=device)

        # ---- Force prediction loss (global frame, equivariant) -------------
        force_pred = self.force_decoder(z, contact_pred, normals_input)
        if finger_valid.any():
            force_loss = F.mse_loss(
                force_pred[finger_valid], finger_forces_gt[finger_valid]
            )
        else:
            force_loss = torch.tensor(0.0, device=device)

        # ---- Forward kinematics: all collision sphere positions -------------
        # Use GT wrist pose during training (the model's wrist prediction is
        # supervised by the flow loss; injecting the ODE-sampled wrist here
        # would be expensive and add noise to the collision signal).
        # FK operates in the palm (base) frame; convert if training in GC frame.
        wrist_pose_fk = wrist_pose
        if self.wrist_frame == "grasp_center":
            from equidexflow.kinematics.allegro_fk import shift_wrist_frame
            wrist_pose_fk = shift_wrist_frame(wrist_pose, to_base=True)
        pred_collision_spheres, collision_radii = self.fk.forward_all_spheres(
            hand_q_pred, wrist_pose_fk,
        )  # (B, S, 3), (S,)
        pred_fingertips = pred_collision_spheres[:, -self.n_fingers:]  # (B, n_f, 3)

        # ---- Reach loss at flow-interpolant wrist x_t -------------------------
        # The old reach loss used GT wrist for FK, but at inference the wrist
        # comes from the ODE - so the decoder never learned to reach from an
        # imperfect wrist. Using x_t (already computed, free) teaches the
        # decoder to produce hand_q that reaches contacts from whatever wrist
        # the flow produces at arbitrary denoising progress t.
        # Detach x_t: gradient flows only through hand_q_decoder, not the flow.
        hand_q_at_xt = self.hand_q_decoder(z, x_t.detach())
        wrist_xt_fk = x_t.detach()
        if self.wrist_frame == "grasp_center":
            wrist_xt_fk = shift_wrist_frame(wrist_xt_fk, to_base=True)
        sph_xt, _ = self.fk.forward_all_spheres(hand_q_at_xt, wrist_xt_fk)
        tips_xt = sph_xt[:, -self.n_fingers:]
        if finger_valid.any():
            reach_loss = F.mse_loss(
                tips_xt[finger_valid],
                contact_pred.detach()[finger_valid],
            )
        else:
            reach_loss = torch.tensor(0.0, device=device)

        return {
            'flow':    flow_loss,
            'hand_q':  hand_q_loss,
            'contact': contact_loss,
            'normal':  normal_loss,
            'force':   force_loss,
            'reach':   reach_loss,
            'pred_contacts': contact_pred,
            'pred_normals':  normals_input,
            'pred_forces':   force_pred,
            'pred_force_coords': None,
            'pred_fingertips': pred_fingertips,
            'fingertip_radius': self.fk.fingertip_radius,
            'pred_collision_spheres': pred_collision_spheres,
            'collision_sphere_radii': collision_radii,
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_grasps(
        self,
        object_points: torch.Tensor,  # (B, 3, N) or (3, N)
        num_samples: int = 10,
        center: bool = True,
    ) -> dict:
        """Shared decode path for :meth:`sample` and :meth:`sample_seated`.

        Runs encode -> SE(3) ODE -> decoder heads -> surface projection and
        returns the intermediates both consumers need, as stacked tensors of
        length ``total = B * num_samples``:

            'z'              : (total, C, 3)  encoder features
            'wrist_gc'       : (total, 4, 4)  ODE wrist in the model's native
                               (centered) frame -- grasp_center if
                               ``wrist_frame == 'grasp_center'`` else base.
                               This is what ``refine_wrist`` expects as its init.
            'wrist_base'     : (total, 4, 4)  un-centered, base-frame wrist
                               (ready for FK / output)
            'hand_q'         : (total, hand_dof)
            'contacts'       : (total, n_fingers, 3)  world frame, un-centered
            'forces'         : (total, n_fingers, 3)
            'contact_logits' : (total, n_fingers)
            'pc_mean'        : (total, 3) or None
        """
        if object_points.dim() == 2:
            object_points = object_points.unsqueeze(0)  # (1, 3, N)

        B      = object_points.shape[0]
        device = object_points.device
        total  = B * num_samples

        # ---- R^3 equivariance: center the point cloud before feeding the
        # encoder, then add the mean back to the predicted wrist + contacts.
        # Matches the training-time mean-subtraction in dexgrasp_db.__getitem__.
        if center:
            pc_mean = object_points.mean(dim=-1, keepdim=True)        # (B, 3, 1)
            object_points_in = object_points - pc_mean                # (B, 3, N)
            pc_mean_per_sample = pc_mean.squeeze(-1).repeat_interleave(
                num_samples, dim=0
            )  # (total, 3)
        else:
            object_points_in = object_points
            pc_mean_per_sample = None

        # Tile point cloud for all samples
        pc_rep = object_points_in.repeat_interleave(num_samples, dim=0)  # (total, 3, N)

        # Initial SE(3) distribution
        x_0 = self.init_dist(total, device)  # (total, 4, 4)

        # Encode
        z = self.encoder(pc_rep)  # (total, C, 3)

        # ODE integration  ->  final wrist poses
        traj    = self.ode_solver(z, x_0, self.guided_vector_field)  # (total, T+1, 4, 4)
        x_1_hat = traj[:, -1]                                         # (total, 4, 4)

        # Decode decoders from final features + predicted wrist pose
        # Pass raw (total, C, 3) to hand_q_decoder so it computes SO(3)-invariant
        # per-channel norms; flattened features are not invariant under augmentation.
        if hasattr(self.hand_q_decoder, 'sample'):
            hand_q = self.hand_q_decoder.sample(z, x_1_hat)            # (total, hand_dof)
        else:
            hand_q = self.hand_q_decoder(z, x_1_hat)                   # (total, hand_dof)
        contacts_raw, logits = self.contact_decoder(z, x_1_hat)          # (total,5,3), (total,5)

        # Project contacts to nearest surface point
        surface_pts = pc_rep.transpose(1, 2)                              # (total, N, 3)
        contacts, normals_from_proj = _project_to_surface(
            contacts_raw, surface_pts,
            temperature=self.surface_proj_tau,
        )

        # Normals: prefer surface projection, then learned decoder, then centroid
        if normals_from_proj is not None:
            normals_est = normals_from_proj
        elif self.normal_decoder is not None:
            normals_est = self.normal_decoder(z, contacts)
        else:
            object_centroid = pc_rep.mean(dim=-1).unsqueeze(1)
            normals_est = F.normalize(object_centroid - contacts + 1e-8, dim=-1)
        forces      = self.force_decoder(z, contacts, normals_est)      # (total, 5, 3)

        # Un-center: add the point-cloud mean back to wrist translation and contacts.
        # Forces and hand_q are translation-intrinsic; do not shift.
        if pc_mean_per_sample is not None:
            wrist_out = x_1_hat.clone()
            wrist_out[:, :3, 3] = wrist_out[:, :3, 3] + pc_mean_per_sample      # (total, 3)
            contacts_out = contacts + pc_mean_per_sample.unsqueeze(1)           # (total, n_f, 3)
        else:
            wrist_out = x_1_hat
            contacts_out = contacts

        # Convert grasp-center frame back to palm (base) frame so downstream
        # consumers (FK, renderers, q_star assembly) always get base-frame poses.
        if self.wrist_frame == "grasp_center":
            from equidexflow.kinematics.allegro_fk import shift_wrist_frame
            wrist_base = shift_wrist_frame(wrist_out, to_base=True)
        else:
            wrist_base = wrist_out

        return {
            'z':              z,
            'wrist_gc':       x_1_hat,
            'wrist_base':     wrist_base,
            'hand_q':         hand_q,
            'contacts':       contacts_out,
            'forces':         forces,
            'contact_logits': logits,
            'pc_mean':        pc_mean_per_sample,
        }

    @torch.no_grad()
    def sample(
        self,
        object_points: torch.Tensor,  # (B, 3, N) or (3, N)
        num_samples: int = 10,
        center: bool = True,
    ) -> list[dict]:
        """Generate num_samples x B grasp candidates (raw decoder output).

        Returns a flat list of dicts (length = B x num_samples), each with:
            'wrist_pose'    : (4, 4)
            'hand_q'        : (hand_dof,)
            'contacts'      : (n_fingers, 3)
            'forces'        : (n_fingers, 3)
            'contact_logits': (n_fingers,)

        The wrist/hand_q are the unseated decoder output: contacts are snapped
        to the object surface, but the hand pose is not adjusted to reach them.
        Use :meth:`sample_seated` for an FK-consistent, object-seated grasp.
        """
        dec = self._decode_grasps(object_points, num_samples, center)
        total = dec['hand_q'].shape[0]
        results = []
        for i in range(total):
            results.append({
                'wrist_pose':     dec['wrist_base'][i],
                'hand_q':         dec['hand_q'][i],
                'contacts':       dec['contacts'][i],
                'forces':         dec['forces'][i],
                'contact_logits': dec['contact_logits'][i],
            })
        return results

    def sample_seated(
        self,
        object_points: torch.Tensor,  # (B, 3, N) or (3, N)
        num_samples: int = 10,
        center: bool = True,
        n_steps: int = 250,
        lr: float = 0.02,
        trust_w: float = 0.05,
        selfcoll_w: float = 0.0,
        mesh_pts: torch.Tensor | None = None,   # (M, 3) object surface pts (object frame)
        mesh_nrm: torch.Tensor | None = None,   # (M, 3) outward unit normals
        coll_points_fn=None,                    # (hand_q, wrist) -> (B, K, 3) full-hand pts
        pen_w: float = 50.0,
        tip_margin: float = 0.002,
        reach_gate_tau: float = 0.0,
        return_raw: bool = False,
    ) -> list[dict]:
        """Sample grasps and *seat* the hand onto the object.

        Same return schema as :meth:`sample`, but ``wrist_pose`` / ``hand_q``
        are the seated pose whose FK fingertips reach the predicted contacts.
        Seating optimizes the wrist (6 DOF) **and** ``hand_q`` jointly in task
        space (:func:`equidexflow.kinematics.seating.seat_grasp`) under a trust
        region and the decoder's joint limits -- far stronger than a wrist-only
        adjustment, which cannot bring four fingertips to four contacts with a
        single rigid move. ``contacts`` / ``forces`` / ``contact_logits`` are the
        same surface-projected predictions as :meth:`sample`.

        **Penetration awareness:** pass ``mesh_pts`` (and ``mesh_nrm``) -- object
        surface points and outward normals in the SAME (object) frame as the
        returned contacts -- to enable the signed-distance penetration term.
        Without them, seating is penetration-unaware and the reach objective can
        drag proximal links through a thin object. The demo always passes them.

        ``n_steps`` / ``lr`` / ``trust_w`` / ``selfcoll_w`` / ``pen_w`` /
        ``tip_margin`` are forwarded to :func:`seat_grasp`. With
        ``return_raw=True`` each dict additionally carries
        ``'wrist_pose_raw'`` / ``'hand_q_raw'`` (the unseated decoder output).
        """
        from equidexflow.kinematics.seating import seat_grasp

        # Decode under no_grad; seat_grasp runs its own inner autograd, so it
        # must NOT execute inside a torch.no_grad() context.
        dec = self._decode_grasps(object_points, num_samples, center)

        lo = getattr(self.hand_q_decoder, "joint_lower", None)
        hi = getattr(self.hand_q_decoder, "joint_upper", None)
        if lo is None or hi is None:
            # Wide fallback limits if the decoder doesn't expose them.
            lo = torch.full((self.hand_dof,), -3.14159, device=dec['hand_q'].device)
            hi = torch.full((self.hand_dof,),  3.14159, device=dec['hand_q'].device)

        if mesh_pts is not None:
            mesh_pts = mesh_pts.to(dec['hand_q'].device, dec['hand_q'].dtype)
        if mesh_nrm is not None:
            mesh_nrm = mesh_nrm.to(dec['hand_q'].device, dec['hand_q'].dtype)

        # Reach targets for the fingertip CENTERS. The decoder contacts sit ON the
        # surface, but the FK fingertip is the pad center -- so target the contact
        # offset OUTWARD by the pad radius along the surface normal. The pad then
        # touches the contact instead of the reach term (center->surface) fighting
        # the penetration term (which holds the hand outside the surface).
        reach_targets = dec['contacts']
        if mesh_pts is not None and mesh_nrm is not None:
            with torch.no_grad():
                d = torch.cdist(dec['contacts'], mesh_pts)        # (B, nf, M)
                n_out = mesh_nrm[d.argmin(-1)]                    # (B, nf, 3)
                r_tip = float(self.fk.fingertip_radius)
                reach_targets = dec['contacts'] + r_tip * n_out

        wrist_seat, hand_q_seat = seat_grasp(
            self.fk, dec['hand_q'], dec['wrist_base'], reach_targets,
            lo, hi, mesh_pts=mesh_pts, mesh_nrm=mesh_nrm, coll_points_fn=coll_points_fn,
            n_steps=n_steps, lr=lr, trust_w=trust_w, selfcoll_w=selfcoll_w,
            pen_w=pen_w, tip_margin=tip_margin, reach_gate_tau=reach_gate_tau,
        )

        total = hand_q_seat.shape[0]
        results = []
        for i in range(total):
            g = {
                'wrist_pose':     wrist_seat[i].detach(),
                'hand_q':         hand_q_seat[i].detach(),
                'contacts':       dec['contacts'][i],
                'forces':         dec['forces'][i],
                'contact_logits': dec['contact_logits'][i],
            }
            if return_raw:
                g['wrist_pose_raw'] = dec['wrist_base'][i]
                g['hand_q_raw']     = dec['hand_q'][i]
            results.append(g)
        return results

    # ------------------------------------------------------------------
    # Phase 3: Test-time wrist refinement via gradient descent
    # ------------------------------------------------------------------

    def refine_wrist(
        self,
        z: torch.Tensor,            # (B, C, 3) encoder features (detached)
        wrist_init: torch.Tensor,   # (B, 4, 4) ODE output (grasp_center frame)
        contacts: torch.Tensor,     # (B, n_fingers, 3) predicted contacts (world)
        pc_mean: torch.Tensor | None = None,  # (B, 3) point cloud mean offset
        n_steps: int = 200,
        n_rounds: int = 3,
        lr: float = 3e-3,
        reg_weight: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Refine wrist pose to minimize FK-contact gap (alternating optimization).

        Strategy: alternating rounds of (1) optimize wrist with fixed hand_q,
        (2) re-decode hand_q at refined wrist. Each round the decoder sees a
        better wrist and produces better joints, improving the next alignment.

        Returns (refined_wrist, refined_hand_q) in the same frame as wrist_init.
        """
        import roma
        from equidexflow.kinematics.allegro_fk import shift_wrist_frame

        B = wrist_init.shape[0]
        device = wrist_init.device
        z_detached = z.detach()
        wrist_current = wrist_init.detach().clone()

        for rnd in range(n_rounds):
            # Decode hand_q at current wrist (fixed for this round)
            with torch.no_grad():
                if hasattr(self.hand_q_decoder, 'sample'):
                    hand_q_fixed = self.hand_q_decoder.sample(z_detached, wrist_current)
                else:
                    hand_q_fixed = self.hand_q_decoder(z_detached, wrist_current)

            # Optimize se(3) perturbation relative to current wrist
            delta = torch.zeros(B, 6, device=device, requires_grad=True)
            opt = torch.optim.Adam([delta], lr=lr)

            for step in range(n_steps):
                opt.zero_grad()

                rotvec = delta[:, :3]
                trans = delta[:, 3:]
                R_delta = roma.rotvec_to_rotmat(rotvec)

                wrist_refined = torch.zeros_like(wrist_current)
                wrist_refined[:, :3, :3] = R_delta @ wrist_current[:, :3, :3]
                t_cur = wrist_current[:, :3, 3].unsqueeze(-1)
                wrist_refined[:, :3, 3] = (R_delta @ t_cur).squeeze(-1) + trans
                wrist_refined[:, 3, 3] = 1.0

                wrist_fk = wrist_refined.clone()
                if pc_mean is not None:
                    wrist_fk[:, :3, 3] = wrist_fk[:, :3, 3] + pc_mean
                if self.wrist_frame == "grasp_center":
                    wrist_fk = shift_wrist_frame(wrist_fk, to_base=True)

                sph_all, _ = self.fk.forward_all_spheres(hand_q_fixed, wrist_fk)
                tips = sph_all[:, -self.n_fingers:]

                loss = ((tips - contacts.detach()) ** 2).sum(dim=-1).mean()
                if reg_weight > 0:
                    loss = loss + reg_weight * (delta ** 2).sum(dim=-1).mean()

                loss.backward()
                opt.step()

            # Apply the optimized delta to get the new wrist_current
            with torch.no_grad():
                rotvec = delta[:, :3]
                trans = delta[:, 3:]
                R_delta = roma.rotvec_to_rotmat(rotvec)

                new_wrist = torch.zeros_like(wrist_current)
                new_wrist[:, :3, :3] = R_delta @ wrist_current[:, :3, :3]
                t_cur = wrist_current[:, :3, 3].unsqueeze(-1)
                new_wrist[:, :3, 3] = (R_delta @ t_cur).squeeze(-1) + trans
                new_wrist[:, 3, 3] = 1.0
                wrist_current = new_wrist

        # Final hand_q at the refined wrist
        with torch.no_grad():
            if hasattr(self.hand_q_decoder, 'sample'):
                hand_q_out = self.hand_q_decoder.sample(z_detached, wrist_current)
            else:
                hand_q_out = self.hand_q_decoder(z_detached, wrist_current)

        return wrist_current, hand_q_out

    # ------------------------------------------------------------------
    # Guided vector field (classifier-free guidance)
    # ------------------------------------------------------------------

    def guided_vector_field(
        self,
        z: torch.Tensor,    # (B, C, 3)
        t: torch.Tensor,    # (B, 1)
        x_t: torch.Tensor,  # (B, 4, 4)
    ) -> torch.Tensor:      # (B, 6)
        v_null = self.vector_field(torch.zeros_like(z), t, x_t)
        v_cond = self.vector_field(z,                   t, x_t)
        return (1.0 - self.guidance) * v_null + self.guidance * v_cond

    # ------------------------------------------------------------------
    # Training / validation step helper
    # ------------------------------------------------------------------

    def step(
        self,
        batch: dict,
        losses_dict: dict,
        split: str,
        optimizer=None,
    ) -> dict:
        """One training/validation step.

        Parameters
        ----------
        batch       : data batch from DexGraspDBDataset
        losses_dict : accumulator dict that gets updated in-place (copy returned)
        split       : 'train' or 'val'
        optimizer   : if provided, performs backward + step

        Returns
        -------
        Updated losses_dict with new scalar entries.
        """
        if optimizer is not None:
            optimizer.zero_grad()

        losses = self.forward(batch)
        total  = sum(losses.values())

        if optimizer is not None:
            total.backward()
            optimizer.step()

        updated = dict(losses_dict)
        for k, v in losses.items():
            updated[f'scalar/{split}/{k}'] = v.item()
        updated[f'scalar/{split}/loss'] = total.item()

        return updated

    # ------------------------------------------------------------------
    # SE(3) interpolation (mirrors original get_traj for convenience)
    # ------------------------------------------------------------------

    @staticmethod
    def get_traj(x_0, x_1, t):
        """Geodesic interpolation on SE(3); re-exported for trainer use."""
        return get_traj(x_0, x_1, t)
