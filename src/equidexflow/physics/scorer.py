"""
Grasp scoring and ranking for dexterous grasp candidates.

Implements the combined quality formula from the EquiDexFlow problem formulation:

    J(q, C, F; O) = β1·Q_geom + β2·Q_phys + β3·Q_task - β4·Q_risk

where:
    Q_geom  = -collision_penalty - self_collision_penalty
    Q_phys  = -wrench_balance_residual - friction_cone_violation_rate
    Q_task  = min singular value of grasp map G  (GWS quality proxy)
    Q_risk  = mean contact spread variance  (uncertainty proxy)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from equidexflow.kinematics.grasp_map import compute_grasp_map, wrench_balance_residual
from equidexflow.kinematics.friction_cone import friction_cone_violation_rate
from equidexflow.kinematics.collision import (
    collision_penalty,
    self_collision_penalty,
    link_self_collision_penalty,
    inter_finger_clustering_penalty,
)


class GraspScorer:
    """Scores and ranks dexterous grasp candidates.

    Scoring formula (Equation 9 from problem formulation):
        J(q, C, F; O) = β1*Q_geom + β2*Q_phys + β3*Q_task - β4*Q_risk
                        + β5*Q_consist - β6*Q_cluster

    where:
        Q_geom  = -collision_penalty - self_collision_penalty (geometric feasibility),
                  minus FK link-link self-collision (capsule overlap) when FK +
                  hand_q are available
        Q_phys  = -wrench_balance_residual - friction_cone_violation_rate (physics quality)
        Q_task  = min singular value of G  (GWS quality proxy)
        Q_risk  = contact_spread_variance of PREDICTED contacts (uncertainty proxy)
        Q_consist = -mean FK fingertip-to-predicted-contact gap (kinematic consistency)
        Q_cluster = inter-finger min-distance hinge on the REALIZED (seated) FK
                    geometry (catches post-seat clustering that Q_risk cannot)

    FK collision (optional):
        When ``fk_module`` is provided, Q_geom also includes a heavy penalty
        for FK-derived finger-body sphere penetration through the object mesh.
        This catches grasps where the contact decoder predicts plausible
        surface contacts but the hand_q actually sends fingers through the
        object.
    """

    def __init__(
        self,
        beta1: float = 1.0,   # geometry weight
        beta2: float = 2.0,   # physics weight
        beta3: float = 1.0,   # task weight
        beta4: float = 0.5,   # risk weight
        beta5: float = 0.0,   # FK-contact consistency weight
        beta6: float = 1.0,   # realized-pose clustering weight
        mu: float = 0.5,
        object_mass: float = 0.2,
        fk_module: torch.nn.Module | None = None,
        fk_collision_weight: float = 5.0,
        linkcoll_weight: float = 5.0,
    ) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.beta4 = beta4
        self.beta5 = beta5
        self.beta6 = beta6
        self.mu = mu
        self.object_mass = object_mass
        self.fk = fk_module
        self.fk_collision_weight = fk_collision_weight
        self.linkcoll_weight = linkcoll_weight

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor(x, dtype=torch.float32) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.as_tensor(x, dtype=dtype)

    @staticmethod
    def _pts_to_n3(pts: torch.Tensor) -> torch.Tensor:
        """Ensure object point cloud has shape (N, 3)."""
        if pts.dim() == 3:
            pts = pts.squeeze(0)
        if pts.shape[-1] == 3:
            return pts        # (N, 3)
        return pts.permute(1, 0)  # (3, N)  ->  (N, 3)

    @staticmethod
    def _pts_to_batch(pts: torch.Tensor, K: int) -> torch.Tensor:
        """Return object_points as (K, ?, ?) with preserved channel format."""
        if pts.dim() == 2:
            pts = pts.unsqueeze(0)   # (1, N, 3) or (1, 3, N)
        # Expand to (K, ...) without copying data
        return pts.expand(K, *pts.shape[1:])

    def _estimate_normals(
        self,
        contacts: torch.Tensor,     # (K, 5, 3) or (5, 3)
        pts_n3: torch.Tensor,        # (N, 3) - shared centroid
    ) -> torch.Tensor:
        """Estimate inward normals as direction from each contact toward centroid."""
        centroid = pts_n3.mean(dim=0)   # (3,)
        # centroid - contacts: points inward toward the object center
        inward = centroid - contacts    # (..., 3)
        return F.normalize(inward, dim=-1)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def batch_physics_score(
        self,
        candidates: list[dict],
        object_points: torch.Tensor,  # (N, 3) or (3, N)
        object_point_normals: torch.Tensor | None = None,  # (M, 3) or (3, M)
        fk_mesh_points: torch.Tensor | None = None,        # (M, 3) matched to normals
    ) -> torch.Tensor:                # (K,) scores
        """Batch scoring for efficiency.

        All candidates are scored in a single vectorised forward pass.

        When ``self.fk`` is set and candidates contain ``hand_q`` and
        ``wrist_pose``, FK-derived collision spheres are checked against the
        object mesh. ``fk_mesh_points`` and ``object_point_normals`` must be
        paired (same N, same ordering). If ``fk_mesh_points`` is None, falls
        back to ``object_points`` (which may have fewer samples).
        """
        K = len(candidates)
        device = object_points.device if isinstance(object_points, torch.Tensor) \
            else torch.device("cpu")

        pts = self._to_tensor(object_points).to(device)
        pts_n3 = self._pts_to_n3(pts)                         # (N, 3)
        pts_batch = self._pts_to_batch(pts, K)                # (K, ?, ?)

        # Stack contacts and forces: (K, 5, 3)
        contacts = torch.stack(
            [self._to_tensor(c["contacts"]) for c in candidates]
        ).to(device)
        forces = torch.stack(
            [self._to_tensor(c["forces"]) for c in candidates]
        ).to(device)

        normals = self._estimate_normals(contacts, pts_n3)     # (K, N_FINGERS, 3)
        from equidexflow.loaders.schema import N_FINGERS as _NF
        valid_mask = torch.ones(K, _NF, dtype=torch.bool, device=device)

        # -- Q_geom --
        coll = collision_penalty(contacts, pts_batch)          # (K,)
        sc   = self_collision_penalty(contacts)                # (K,)
        Q_geom = -coll - sc                                    # (K,)

        # FK collision: penalize actual finger-mesh penetration
        if self.fk is not None and object_point_normals is not None:
            has_fk = all("hand_q" in c and "wrist_pose" in c for c in candidates)
            if has_fk:
                hand_q_batch = torch.stack(
                    [self._to_tensor(c["hand_q"]) for c in candidates]
                ).to(device)                                       # (K, 11)
                wrist_batch = torch.stack(
                    [self._to_tensor(c["wrist_pose"]) for c in candidates]
                ).to(device)                                       # (K, 4, 4)

                sphere_pos, sphere_radii = self.fk.forward_all_spheres(
                    hand_q_batch, wrist_batch,
                )  # (K, S, 3), (S,)

                nrm = self._to_tensor(object_point_normals).to(device)
                if nrm.dim() == 2:
                    nrm = nrm.unsqueeze(0)
                if nrm.shape[-1] != 3:
                    nrm = nrm.permute(0, 2, 1)                    # (1, M, 3)
                nrm_batch = nrm.expand(K, -1, -1)                 # (K, M, 3)

                if fk_mesh_points is not None:
                    fk_pts = self._to_tensor(fk_mesh_points).to(device)
                    fk_pts = self._pts_to_n3(fk_pts)              # (M, 3)
                else:
                    fk_pts = pts_n3                                # fallback
                fk_pts_batch = fk_pts.unsqueeze(0).expand(K, -1, -1)

                n_link = sphere_radii.shape[0] - 4
                margins = sphere_radii.clone()
                margins[-4:] = (margins[-4:] - 0.002).clamp(min=0.0)

                fk_coll = collision_penalty(
                    sphere_pos, fk_pts_batch, nrm_batch, margin=margins,
                )  # (K,)
                Q_geom = Q_geom - self.fk_collision_weight * fk_coll

        # -- Link-link self-collision + realized-pose clustering --
        # Needs FK + per-candidate hand_q/wrist (NOT object normals), so it is
        # gated separately from the object-penetration term above.
        Q_cluster = torch.zeros(K, device=device)
        if self.fk is not None and all(
            "hand_q" in c and "wrist_pose" in c for c in candidates
        ):
            hq = torch.stack(
                [self._to_tensor(c["hand_q"]) for c in candidates]
            ).to(device)
            wr = torch.stack(
                [self._to_tensor(c["wrist_pose"]) for c in candidates]
            ).to(device)
            seg_a, seg_b, caps_r = self.fk.forward_link_capsules(hq, wr)
            link_sc = link_self_collision_penalty(
                seg_a, seg_b, caps_r, self.fk._caps_pair_mask,
            )                                                  # (K,)
            Q_geom = Q_geom - self.linkcoll_weight * link_sc
            Q_cluster = inter_finger_clustering_penalty(
                seg_a, seg_b, caps_r, self.fk._caps_finger,
            )                                                  # (K,)

        # -- Q_phys --
        wb   = wrench_balance_residual(
            contacts, normals, forces, valid_mask,
            object_mass=self.object_mass,
        )                                                      # (K,)
        fc   = friction_cone_violation_rate(
            forces, normals, valid_mask, mu=self.mu,
        )                                                      # (K,)
        Q_phys = -wb - fc                                      # (K,)

        # -- Q_task: min singular value of the grasp map --
        G = compute_grasp_map(contacts, normals)               # (K, 6, 15)
        sv = torch.linalg.svdvals(G)                           # (K, 6)
        Q_task = sv[:, -1]                                     # (K,) smallest SV

        # -- Q_risk: mean variance of contact positions --
        Q_risk = contacts.var(dim=1).mean(dim=-1)              # (K,)

        # -- Q_consist: FK fingertip-to-predicted-contact consistency --
        Q_consist = torch.zeros(K, device=device)
        if self.beta5 != 0.0 and self.fk is not None:
            has_fk_data = all("hand_q" in c and "wrist_pose" in c for c in candidates)
            if has_fk_data:
                hand_q_batch = torch.stack(
                    [self._to_tensor(c["hand_q"]) for c in candidates]
                ).to(device)
                wrist_batch = torch.stack(
                    [self._to_tensor(c["wrist_pose"]) for c in candidates]
                ).to(device)
                sph_all, _ = self.fk.forward_all_spheres(hand_q_batch, wrist_batch)
                tips = sph_all[:, -4:]  # (K, 4, 3) - last 4 spheres are fingertips
                gap = (tips - contacts).norm(dim=-1).mean(dim=-1)  # (K,)
                Q_consist = -gap

        scores = (
            self.beta1 * Q_geom
            + self.beta2 * Q_phys
            + self.beta3 * Q_task
            - self.beta4 * Q_risk
            + self.beta5 * Q_consist
            - self.beta6 * Q_cluster
        )
        return scores  # (K,)

    @torch.no_grad()
    def physics_score(
        self,
        candidate: dict,
        object_points: torch.Tensor,  # (N, 3) or (3, N)
        object_point_normals: torch.Tensor | None = None,
        fk_mesh_points: torch.Tensor | None = None,
    ) -> float:
        """Scalar score for a single candidate. Higher is better."""
        scores = self.batch_physics_score(
            [candidate], object_points, object_point_normals, fk_mesh_points,
        )
        return scores[0].item()

    def rank_candidates(
        self,
        candidates: list[dict],
        object_points: torch.Tensor,  # (N, 3) or (3, N)
        object_point_normals: torch.Tensor | None = None,
        fk_mesh_points: torch.Tensor | None = None,
    ) -> list[dict]:
        """Return candidates sorted by score (best first), with 'score' key added."""
        scores = self.batch_physics_score(
            candidates, object_points, object_point_normals, fk_mesh_points,
        )  # (K,)
        order = torch.argsort(scores, descending=True).tolist()

        ranked = []
        for idx in order:
            entry = dict(candidates[idx])
            entry["score"] = scores[idx].item()
            ranked.append(entry)
        return ranked
