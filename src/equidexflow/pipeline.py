"""Shared grasp-synthesis pipeline: sample -> penetration-aware seat -> rank.

Both the public demo (``cli/demo.py``) and any external/figure tooling call this,
so the seating + selection logic lives in exactly one place.
"""

from __future__ import annotations

import numpy as np
import torch


def sample_surface_with_normals(mesh, n_points: int, rng: np.random.Generator):
    """Surface points + outward unit normals (object frame)."""
    import trimesh  # noqa: F401  (mesh is already a trimesh)

    pts, fid = trimesh.sample.sample_surface(mesh, n_points, seed=int(rng.integers(2**31)))
    return np.asarray(pts, dtype=np.float32), np.asarray(mesh.face_normals[fid], dtype=np.float32)


def rank_grasps_quality(grasps, fk, coll_fn, mesh_pts, mesh_nrm, object_pc) -> list[int]:
    """Rank grasps by the repo's force-closure ``GraspScorer`` (the same one
    ``online_inference`` uses) plus a light penetration penalty for cleanliness.

    The released checkpoint's ``contact_logits`` are near-constant across samples,
    so mean-logit ranking is non-discriminative. ``GraspScorer`` rewards
    opposition (``Q_task`` = min singular value of the grasp map) and penalizes
    clustered same-side contacts (``Q_risk`` = contact-spread variance). We add a
    small penalty for max full-hand surface penetration. Lower key = better.

    NOTE: selection picks the best of N -- it cannot invent opposition the model
    never predicted (bare primitives stay weakly opposed regardless).
    """
    from equidexflow.physics.scorer import GraspScorer

    scorer = GraspScorer(mu=0.5, object_mass=0.2)
    phys = scorer.batch_physics_score([dict(g) for g in grasps], object_pc)  # (K,) higher better
    from equidexflow.kinematics.collision import link_self_collision_penalty

    keys = []
    for i, g in enumerate(grasps):
        pen_mm = 0.0
        if coll_fn is not None:
            with torch.no_grad():
                P = coll_fn(g["hand_q"].unsqueeze(0).to(mesh_pts.device),
                            g["wrist_pose"].unsqueeze(0).to(mesh_pts.device))[0]
                ni = torch.cdist(P, mesh_pts).argmin(1)
                signed = ((P - mesh_pts[ni]) * mesh_nrm[ni]).sum(-1)
                pen_mm = float(torch.relu(-signed).max()) * 1000.0
        # Worst finger-link self-collision (capsule overlap) in mm, as a small
        # tie-breaker so the rendered top grasp is geometrically clean.
        link_mm = 0.0
        with torch.no_grad():
            seg_a, seg_b, caps_r = fk.forward_link_capsules(
                g["hand_q"].unsqueeze(0).to(mesh_pts.device),
                g["wrist_pose"].unsqueeze(0).to(mesh_pts.device),
            )
            link_pen = link_self_collision_penalty(
                seg_a, seg_b, caps_r, fk._caps_pair_mask, clearance=0.0,
            )
            link_mm = float(link_pen[0]) * 1000.0
        keys.append(-float(phys[i]) + 0.1 * pen_mm + 0.1 * link_mm)
    return list(np.argsort(keys))


def generate_seated_grasps(
    model,
    mesh,
    *,
    num_samples: int = 32,
    num_points: int = 512,
    seat_steps: int = 250,
    seat_top_k: int = 4,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Sample many -> force-closure rank -> seat only the best few -> re-rank.

    Two-stage by design: ``model.sample`` is cheap, so we draw ``num_samples``
    candidates and rank them with ``GraspScorer`` on the RAW contacts/forces
    (which seating does not change). Only the top ``seat_top_k`` are then run
    through the expensive penetration-aware seating, and re-ranked by
    ``GraspScorer`` minus penetration. This recovers paper-quality grasps (the
    decoder's per-object best needs a decent candidate pool) at a fraction of the
    cost of seating every sample -- undersampling + non-discriminative selection,
    not the model or the seating, was what produced clustered/penetrating grasps.

    Returns a dict: ``grasps`` (the seated top-k dicts), ``order`` (ranked indices
    into ``grasps``, best first), ``pc`` (3, N), ``mesh_pts`` / ``mesh_nrm``,
    ``coll_fn``, ``fk``.
    """
    from equidexflow.kinematics.allegro_fk import AllegroRightHandFK
    from equidexflow.kinematics.seating import seat_grasp

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    pc_np, _ = sample_surface_with_normals(mesh, num_points, rng)        # (N, 3)
    pc = torch.from_numpy(np.ascontiguousarray(pc_np.T)).to(device)      # (3, N)
    seat_pts, seat_nrm = sample_surface_with_normals(mesh, max(num_points * 4, 2048), rng)
    mesh_pts = torch.from_numpy(seat_pts).to(device)
    mesh_nrm = torch.from_numpy(seat_nrm).to(device)

    fk = AllegroRightHandFK().to(device).eval()
    coll_fn = None
    if str(getattr(model, "hand", "allegro")).lower() == "allegro":
        try:
            from equidexflow.kinematics.collision_points import build_allegro_collision_fn
            coll_fn = build_allegro_collision_fn(fk, device=device)
        except Exception:
            coll_fn = None

    # Stage 1: sample a pool (cheap) and rank RAW by force closure.
    raw = model.sample(pc, num_samples=num_samples)
    k = max(1, min(seat_top_k, len(raw)))
    try:
        from equidexflow.physics.scorer import GraspScorer
        phys = GraspScorer(mu=0.5, object_mass=0.2).batch_physics_score(
            [dict(g) for g in raw], pc)
        cand = torch.argsort(phys, descending=True).tolist()[:k]
    except Exception:
        cand = list(range(k))

    # Stage 2: seat only the candidates (batched), with pad-offset reach targets.
    hand_q = torch.stack([raw[i]["hand_q"] for i in cand]).to(device)
    wrist = torch.stack([raw[i]["wrist_pose"] for i in cand]).to(device)
    contacts = torch.stack([raw[i]["contacts"] for i in cand]).to(device)
    lo = getattr(model.hand_q_decoder, "joint_lower", None)
    hi = getattr(model.hand_q_decoder, "joint_upper", None)
    if lo is None or hi is None:
        lo = torch.full((hand_q.shape[-1],), -3.14159, device=device)
        hi = torch.full((hand_q.shape[-1],), 3.14159, device=device)
    r_tip = float(fk.fingertip_radius)
    n_out = mesh_nrm[torch.cdist(contacts, mesh_pts).argmin(-1)]         # (k, nf, 3)
    reach_targets = contacts + r_tip * n_out
    wrist_s, hand_q_s = seat_grasp(
        fk, hand_q, wrist, reach_targets, lo, hi,
        mesh_pts=mesh_pts, mesh_nrm=mesh_nrm, coll_points_fn=coll_fn, n_steps=seat_steps,
    )

    grasps = []
    for j, i in enumerate(cand):
        grasps.append({
            "wrist_pose": wrist_s[j].detach(),
            "hand_q": hand_q_s[j].detach(),
            "contacts": raw[i]["contacts"],
            "forces": raw[i]["forces"],
            "contact_logits": raw[i]["contact_logits"],
        })

    try:
        order = rank_grasps_quality(grasps, fk, coll_fn, mesh_pts, mesh_nrm, pc)
    except Exception:
        order = list(range(len(grasps)))

    return dict(grasps=grasps, order=order, pc=pc, mesh_pts=mesh_pts,
                mesh_nrm=mesh_nrm, coll_fn=coll_fn, fk=fk)
