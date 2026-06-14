#!/usr/bin/env python3
"""Generate grasp candidates for paper figures (Figures 4, 5, 6).

Runs model.sample() on selected test objects, ranks with GraspScorer,
and saves top-K grasps as JSON files ready for rendering on any machine.

USAGE (from the repo root, in the project venv):
    python scripts/generate_figure_grasps.py \
        --checkpoint allegro_full --variant full \
        --objects cube mustard_bottle bleach_cleanser sphere B2 F4 \
        --num-samples 10 --top-k 2 --device 0 \
        --out outputs/figure_grasps/

    For ablation comparison, run once per variant (full pose_only contact_only
    geom_only). Object meshes resolve from EQUIDEXFLOW_OBJECTS_DIR (default
    assets/objects); EGAD meshes from EQUIDEXFLOW_EGAD_ROOT (default
    ~/.cache/equidexflow/egad), both populated by scripts/download_assets.py.

VRAM: ~4GB peak (single object inference)

OUTPUT per grasp:
    <out>/<variant>/<object>/<object>__grasp__<rank>.json
    Fields: wrist_pose (4x4), hand_q (11,), contacts (4x3), forces (4x3),
            score (float), object_name (str), variant (str)
"""
from __future__ import annotations
import os
import sys

_EDF_DIR = os.environ.get("EQUIDEXFLOW_OUTPUTS_DIR", os.getcwd())

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation


def _to_serializable(v):
    if isinstance(v, torch.Tensor):
        return v.cpu().numpy().tolist()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


REPO_ROOT = Path(__file__).resolve().parents[1]
_OBJ_MESH_DIR = Path(
    os.environ.get("EQUIDEXFLOW_OBJECTS_DIR", str(REPO_ROOT / "assets" / "objects"))
)
_EGAD_MESH_DIR = Path(
    os.environ.get("EQUIDEXFLOW_EGAD_ROOT", os.path.expanduser("~/.cache/equidexflow/egad"))
)

_MESH_STEM = {
    "box": "graspit/box", "cube": "graspit/cube",
    "cylinder": "graspit/cylinder", "graspit_box": "graspit/box",
    "graspit_cylinder": "graspit/cylinder", "sphere": "graspit/sphere",
    "phydex": "graspit/phydex", "sns_cup": "graspit/sns_cup",
    "chips_can": "frogger_ycb/001_chips_can",
    "master_chef_can": "frogger_ycb/002_master_chef_can",
    "cracker_box": "frogger_ycb/003_cracker_box",
    "sugar_box": "frogger_ycb/004_sugar_box",
    "tomato_soup_can": "frogger_ycb/005_tomato_soup_can",
    "mustard_bottle": "frogger_ycb/006_mustard_bottle",
    "tuna_fish_can": "frogger_ycb/007_tuna_fish_can",
    "pudding_box": "frogger_ycb/008_pudding_box",
    "gelatin_box": "frogger_ycb/009_gelatin_box",
    "potted_meat_can": "frogger_ycb/010_potted_meat_can",
    "banana": "frogger_ycb/011_banana",
    "bleach_cleanser": "frogger_ycb/021_bleach_cleanser",
    "tennis_ball": "frogger_ycb/056_tennis_ball",
    "foam_brick": "frogger_ycb/061_foam_brick",
}


def _optimize_wrist(
    hand_q: torch.Tensor,       # (11,)
    wrist_pose: torch.Tensor,   # (4, 4)
    fk,                          # AllegroRightHandFK
    mesh_pts: torch.Tensor,     # (M, 3)
    mesh_nrm: torch.Tensor,     # (M, 3)
    n_steps: int = 300,
    lr: float = 0.005,
    collision_w: float = 100.0,
    surface_w: float = 10.0,
) -> torch.Tensor:
    """Optimize wrist translation to minimize collision while keeping tips near surface.

    Uses gradient descent on the wrist position (3 DOF, rotation frozen) to
    minimize: collision_w * FK_penetration + surface_w * tip_surface_distance.
    """
    # Check if already collision-free
    with torch.no_grad():
        sph0, radii0 = fk.forward_all_spheres(hand_q.unsqueeze(0), wrist_pose.unsqueeze(0))
        sph0 = sph0[0]
        d0 = sph0.unsqueeze(1) - mesh_pts.unsqueeze(0)
        dsq0 = (d0 ** 2).sum(-1)
        ni0 = dsq0.argmin(dim=-1)
        np0 = mesh_pts[ni0]
        nn0 = mesh_nrm[ni0]
        s0 = ((sph0 - np0) * nn0).sum(-1)
        m0 = radii0.clone()
        m0[-4:] = (m0[-4:] - 0.002).clamp(min=0.0)
        if (m0 - s0).max().item() <= 0.001:
            return wrist_pose

    with torch.enable_grad():
        delta = torch.zeros(3, device=wrist_pose.device, dtype=wrist_pose.dtype,
                            requires_grad=True)
        opt = torch.optim.Adam([delta], lr=lr)

        hand_q_b = hand_q.unsqueeze(0).detach()
        radii_cache = None

        for step in range(n_steps):
            opt.zero_grad()
            T = wrist_pose.clone().detach()
            T[:3, 3] = T[:3, 3] + delta
            T = T.unsqueeze(0)

            sph, radii = fk.forward_all_spheres(hand_q_b, T)
            sph = sph[0]  # (S, 3)
            if radii_cache is None:
                radii_cache = radii.detach()

            # Signed distance for each sphere
            diffs = sph.unsqueeze(1) - mesh_pts.unsqueeze(0)  # (S, M, 3)
            dist_sq = (diffs ** 2).sum(-1)                     # (S, M)
            nearest_idx = dist_sq.argmin(dim=-1).detach()      # stop grad through argmin
            nearest_pts = mesh_pts[nearest_idx]
            nearest_nrm = mesh_nrm[nearest_idx]
            diff_near = sph - nearest_pts
            signed = (diff_near * nearest_nrm).sum(-1)         # (S,)

            margins = radii_cache.clone()
            margins[-4:] = (margins[-4:] - 0.002).clamp(min=0.0)
            coll_loss = F.relu(margins - signed).sum()

            # Keep ALL spheres close to the surface (penalize over-retraction)
            all_dists = dist_sq.min(dim=-1).values.sqrt()  # (S,)
            proximity_loss = ((all_dists - radii_cache) ** 2).sum()

            loss = collision_w * coll_loss + surface_w * proximity_loss
            loss.backward()
            opt.step()

            max_pen = (margins - signed.detach()).max().item()
            if max_pen <= 0.0 and step > 20:
                break

    T_out = wrist_pose.clone()
    T_out[:3, 3] = T_out[:3, 3] + delta.detach()
    return T_out


def _rot6d_to_R(x):  # (...,6) -> (...,3,3), Gram-Schmidt (differentiable)
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _R_to_rot6d(R):
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def _tto_refine(fk, hand_q, wrist, contacts, lo, hi,
                mesh_pts=None, mesh_nrm=None,
                n_steps=200, lr=0.02, trust_w=0.05, pen_w=50.0, selfcoll_w=0.0,
                tip_margin=0.002):
    """Test-time optimization: refine (wrist 6-DOF + hand_q) to pull the FK
    fingertips onto the model's predicted contacts. The decoder's contacts are
    surface-accurate (contact-loss ~1e-3) but the sampled wrist+hand_q don't
    realize them; TTO closes that gap in task space. A trust region keeps the
    wrist near the sampled pose, a joint-limit barrier keeps hand_q feasible,
    and an optional signed-distance term penalises mesh penetration.
    Batched over B grasps. Returns (wrist (B,4,4), hand_q (B,16)) detached."""
    device = hand_q.device
    B = hand_q.shape[0]
    nf = fk.N_FINGERS
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
            trust = ((trans - trans0) ** 2).sum(-1).mean() + ((rot6d - rot0) ** 2).sum(-1).mean()
            lim = (torch.relu(q - hi) ** 2 + torch.relu(lo - q) ** 2).sum(-1).mean()
            loss = reach + trust_w * trust + 10.0 * lim
            # Self-collision: penalize inter-fingertip sphere overlap. FRoGGeR
            # excludes all hand-hand pairs (leap.py) so finger overlap is never
            # penalized anywhere; on small objects the fingers crowd and overlap.
            tipr = fk.fingertip_radius
            tipr = tipr.mean() if torch.is_tensor(tipr) else float(tipr)
            dmat = torch.cdist(tips, tips)                          # (B, nf, nf)
            ovr = torch.relu(2.0 * tipr - dmat)
            ovr = ovr.masked_fill(torch.eye(nf, device=device, dtype=torch.bool), 0.0)
            self_coll = 0.5 * (ovr ** 2).sum(dim=(1, 2)).mean()
            loss = loss + selfcoll_w * self_coll
            if mesh_pts is not None:
                dsq = (sph.unsqueeze(2) - mesh_pts.view(1, 1, -1, 3)).pow(2).sum(-1)  # (B,S,M)
                ni = dsq.argmin(-1)                                                   # (B,S)
                signed = ((sph - mesh_pts[ni]) * mesh_nrm[ni]).sum(-1)               # (B,S) >0 outside
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


def _load_mesh_with_normals(
    obj_name: str, n_pts: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load centroid-centered mesh and sample (pts, normals) for collision."""
    stem = _MESH_STEM.get(obj_name, obj_name)
    search_paths = (
        [(_OBJ_MESH_DIR / (stem + ext), False) for ext in (".stl", ".obj", ".ply")]
        + [(_EGAD_MESH_DIR / (obj_name + ext), True) for ext in (".obj", ".stl", ".ply")]
    )
    for path, is_egad in search_paths:
        if path.exists():
            mesh = trimesh.load(str(path), force="mesh")
            if is_egad:
                mesh.apply_scale(0.001)
            # Body-frame centroid (offset of body origin from centroid). The
            # model/loader work in the centered frame; this is what must be
            # added back to put model outputs in the object body frame used by
            # the renderer/scorer. Captured BEFORE centering the sample points.
            body_centroid = torch.from_numpy(mesh.centroid.astype(np.float32)).clone()
            mesh.apply_translation(-mesh.centroid)
            samples, face_idx = trimesh.sample.sample_surface(mesh, n_pts)
            pts = torch.from_numpy(samples.astype(np.float32))
            nrm = torch.from_numpy(
                mesh.face_normals[face_idx].astype(np.float32)
            )
            nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return pts, nrm, body_centroid  # (N,3), (N,3), (3,)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hand", choices=["allegro", "leap"], default="allegro",
                        help="Which hand's FK + dataset to use")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_best.pt")
    parser.add_argument("--variant", required=True, help="Label: full, pose_only, etc.")
    parser.add_argument("--objects", nargs="+", required=True,
                        help="Object names to generate grasps for")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=2,
                        help="Save top-K ranked grasps per object")
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fk-collision", action="store_true",
                        help="Disable FK-based collision penalty in scoring")
    parser.add_argument("--flow", action="store_true",
                        help="Use HandQFlowDecoder (Conditional RealNVP) instead of deterministic")
    parser.add_argument("--n-coupling-layers", type=int, default=8)
    parser.add_argument("--wrist-frame", choices=["base", "grasp_center"],
                        default="base",
                        help="grasp_center: model predicts the finger-centric "
                             "frame; shift back to base before FK/q_star")
    parser.add_argument("--cond-norm", action="store_true",
                        help="Rescale object features into the vector field "
                             "(must match the checkpoint's training setting)")
    parser.add_argument("--refine-tto", action="store_true",
                        help="Test-time optimization: refine all sampled "
                             "candidates (wrist 6-DOF + hand_q) onto predicted "
                             "contacts before ranking")
    parser.add_argument("--tto-steps", type=int, default=200)
    parser.add_argument("--tto-lr", type=float, default=0.02)
    # Contact-IK hardening knobs (defaults reproduce the original behavior).
    parser.add_argument("--tto-pen-w", type=float, default=50.0,
                        help="Mesh-penetration penalty weight in TTO. Raise to "
                             "push fingers/body out of the object surface.")
    parser.add_argument("--tto-selfcoll-w", type=float, default=0.0,
                        help="Inter-fingertip self-collision penalty weight in TTO "
                             "(>0 stops fingers crowding/overlapping on small objects).")
    parser.add_argument("--tto-tip-margin", type=float, default=0.002,
                        help="Allowed fingertip penetration depth (m) in TTO. "
                             "Lower (->0) for a harder no-penetration constraint.")
    parser.add_argument("--tto-mesh-pts", type=int, default=1024,
                        help="Surface points used by the TTO penetration term. "
                             "Higher = finer penetration detection (less leakage "
                             "between sampled points).")
    parser.add_argument("--tto-trust-w", type=float, default=0.05,
                        help="Trust-region weight keeping the wrist near its decoded "
                             "pose. Lower lets the IK relocate the wrist to back the "
                             "palm/body out of the object (fixes deep body penetration "
                             "from an imperfectly-placed wrist).")
    parser.add_argument("--save-pre-tto", action="store_true",
                        help="Also write the pre-TTO (raw decoded) grasp for each "
                             "saved candidate as <obj>__grasp_pre__NN.json, so the "
                             "same candidate can be rendered before/after refinement")
    parser.add_argument("--surface-proj-tau", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Flow sampling temperature (lower = more deterministic)")
    args = parser.parse_args()

    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    torch.manual_seed(args.seed)

    from equidexflow.models import get_dex_model
    model = get_dex_model(
        p_uncond=0.1, guidance=2.0, num_ode_steps=10,
        hand_q_decoder_type="flow" if args.flow else "deterministic",
        n_coupling_layers=args.n_coupling_layers,
        surface_proj_tau=args.surface_proj_tau,
        hand=args.hand,
        wrist_frame=args.wrist_frame,
        cond_norm=args.cond_norm,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint/decoder mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected keys. Wrong --flow? "
            f"missing[:3]={missing[:3]} unexpected[:3]={unexpected[:3]}")
    model.eval()
    print(f"Loaded {args.checkpoint}")

    if args.hand == "allegro":
        from equidexflow.kinematics.allegro_fk import AllegroRightHandFK
        fk = AllegroRightHandFK().to(device) if not args.no_fk_collision else None
    else:
        from equidexflow.kinematics.leap import LeapFK
        fk = LeapFK().to(device) if not args.no_fk_collision else None
    if fk is not None:
        n_spheres = getattr(fk, "n_collision_spheres", "n/a")
        print(f"FK collision scoring enabled ({n_spheres} spheres)")

    from equidexflow.physics.scorer import GraspScorer
    scorer = GraspScorer(mu=0.5, object_mass=0.2,
                         beta1=1.0, beta2=2.0, beta3=1.0, beta4=0.5,
                         fk_module=fk, fk_collision_weight=5.0)

    mesh_normals: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    obj_centroids: dict[str, torch.Tensor] = {}
    for obj_name in args.objects:
        result = _load_mesh_with_normals(obj_name)
        if result is not None:
            mesh_normals[obj_name] = (result[0], result[1])
            obj_centroids[obj_name] = result[2]
            print(f"  Loaded mesh+normals for {obj_name} ({result[0].shape[0]} pts)")
        else:
            print(f"  WARNING: no mesh found for {obj_name}, FK collision disabled")

    from equidexflow.loaders import get_dataloader
    dataset_subdir = {"allegro": "allegro", "leap": "leap"}[args.hand]
    _data_env = os.environ.get("EQUIDEXFLOW_DATA_DIR")
    _grasp_db_dir = (
        str(Path(_data_env) / "dexgraspdb" / "v3" / dataset_subdir) if _data_env
        else str(REPO_ROOT / "data" / "dexgraspdb" / "v3" / dataset_subdir)
    )
    ds_cfg = {
        "name": "dexgrasp",
        "grasp_db_dir": _grasp_db_dir,
        "object_mesh_dir": str(_OBJ_MESH_DIR),
        "n_object_points": 512, "max_contacts": 64,
        "mu": 0.5, "object_mass": 0.2,
        "augment": False, "split": "test", "object_names": None,
    }
    if args.hand == "leap":
        ds_cfg["use_frogger_primitive_specs"] = True
    test_cfg = OmegaConf.create({
        "dataset": ds_cfg,
        "batch_size": 1, "num_workers": 0, "shuffle": False,
    })
    loader = get_dataloader("test", test_cfg)

    pts_by_obj: dict[str, torch.Tensor] = {}
    for batch in loader:
        names = batch["object_name"]
        for b in range(batch["object_points"].shape[0]):
            name = names[b]
            if name in args.objects and name not in pts_by_obj:
                pts_by_obj[name] = batch["object_points"][b]
        if all(o in pts_by_obj for o in args.objects):
            break

    missing = set(args.objects) - set(pts_by_obj.keys())
    if missing:
        print(f"WARNING: objects not found in test set: {missing}")

    with torch.no_grad():
        for obj_name in args.objects:
            if obj_name not in pts_by_obj:
                continue

            obj_pts = pts_by_obj[obj_name].unsqueeze(0).to(device)
            preds = model.sample(obj_pts, args.num_samples)

            # The model predicts the finger-centric frame; shift every wrist
            # back to leap_hand_base before ranking/FK/q_star (all of which
            # assume the base pose).
            if args.wrist_frame == "grasp_center":
                from equidexflow.kinematics.leap import shift_wrist_frame
                for pred in preds:
                    pred["wrist_pose"] = shift_wrist_frame(pred["wrist_pose"], to_base=True)

            obj_nrm = None
            fk_pts = None
            if obj_name in mesh_normals:
                mesh_pts, mesh_nrm = mesh_normals[obj_name]
                fk_pts = mesh_pts.to(device)   # (M, 3)
                obj_nrm = mesh_nrm.to(device)  # (M, 3)

            # Frame note: the encoder centers the point cloud, so model outputs,
            # predicted contacts, and the _load_mesh_with_normals points (fk_pts)
            # all live in the SAME pc-centered frame. TTO and ranking therefore
            # run consistently in the centered frame here; the body-frame centroid
            # is added only when writing q_star (below), so the saved grasp lands
            # in the object body frame the renderer/scorer use.

            # Test-time optimization: refine ALL candidates onto their predicted
            # contacts before ranking (TTO changes which candidate is best).
            if args.refine_tto and fk is not None:
                lo = model.hand_q_decoder.joint_lower.to(device)
                hi = model.hand_q_decoder.joint_upper.to(device)
                hq_b = torch.stack([p["hand_q"].to(device).float() for p in preds])
                W_b = torch.stack([
                    (p["wrist_pose"] if isinstance(p["wrist_pose"], torch.Tensor)
                     else torch.tensor(p["wrist_pose"])).to(device).float()
                    for p in preds])
                C_b = torch.stack([p["contacts"].to(device).float()[:fk.N_FINGERS] for p in preds])
                mp = fk_pts; mn = obj_nrm
                if mp is not None and mp.shape[0] > args.tto_mesh_pts:  # cap for TTO memory
                    sel = torch.randperm(mp.shape[0], device=device)[:args.tto_mesh_pts]
                    mp, mn = mp[sel], mn[sel]
                W_ref, q_ref = _tto_refine(fk, hq_b, W_b, C_b, lo, hi,
                                           mesh_pts=mp, mesh_nrm=mn,
                                           n_steps=args.tto_steps, lr=args.tto_lr,
                                           trust_w=args.tto_trust_w,
                                           pen_w=args.tto_pen_w,
                                           selfcoll_w=args.tto_selfcoll_w,
                                           tip_margin=args.tto_tip_margin)
                for i, p in enumerate(preds):
                    if args.save_pre_tto:   # stash raw decoded pose before refine
                        wr = p["wrist_pose"]
                        p["_wrist_raw"] = (wr.detach().clone() if isinstance(wr, torch.Tensor)
                                           else torch.tensor(wr))
                        hr = p["hand_q"]
                        p["_hand_q_raw"] = (hr.detach().clone() if isinstance(hr, torch.Tensor)
                                            else torch.tensor(hr))
                    p["wrist_pose"] = W_ref[i]
                    p["hand_q"] = q_ref[i]

            if args.refine_tto and fk is not None and obj_name in mesh_normals:
                # Rank by render-frame reach: the physics scorer does not select
                # for fingers actually reaching the object, so rank TTO-refined
                # candidates by (n fingers in contact, then mean tip gap), and
                # reject body penetration. fk_pts/obj_nrm are the object mesh.
                ranked = []
                for p in preds:
                    W = p["wrist_pose"].unsqueeze(0)
                    sph, radii = fk.forward_all_spheres(p["hand_q"].unsqueeze(0).to(device), W)
                    sph = sph[0]
                    dsq = (sph.unsqueeze(1) - fk_pts.unsqueeze(0)).pow(2).sum(-1)  # (S, M)
                    ni = dsq.argmin(-1)
                    signed = ((sph - fk_pts[ni]) * obj_nrm[ni]).sum(-1)            # (S,) >0 outside
                    gap = (signed - radii)                                          # sphere-surface gap (m)
                    tip_gap = gap[-fk.N_FINGERS:]
                    n_touch = int((tip_gap.abs() <= 0.013).sum())
                    body_pen = float((-gap[:-fk.N_FINGERS]).clamp(min=0).max()) * 1000  # mm into surface
                    mean_tip = float(tip_gap.abs().mean()) * 1000
                    p["score"] = -(n_touch - 0.001 * mean_tip - (1.0 if body_pen > 3 else 0.0))
                    p["_rank_key"] = (-(n_touch), body_pen > 3, mean_tip)
                    ranked.append(p)
                ranked.sort(key=lambda p: p["_rank_key"])
            else:
                ranked = scorer.rank_candidates(
                    preds, obj_pts.permute(0, 2, 1)[0], obj_nrm, fk_pts,
                )

            out_dir = args.out / args.variant / obj_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for rank, pred in enumerate(ranked[:args.top_k]):
                wrist_T_raw = pred["wrist_pose"]
                hand_q_raw = pred["hand_q"]

                if fk is not None and obj_name in mesh_normals and not args.refine_tto:
                    mesh_pts_t, mesh_nrm_t = mesh_normals[obj_name]
                    wrist_T_adj = _optimize_wrist(
                        hand_q_raw.to(device), wrist_T_raw.to(device),
                        fk, mesh_pts_t.to(device), mesh_nrm_t.to(device),
                    )
                    shift_d = (wrist_T_adj[:3, 3] - wrist_T_raw.to(device)[:3, 3]).norm().item()
                    if shift_d > 0.001:
                        print(f"    [{rank}] wrist optimized, shift={shift_d*1000:.1f}mm")
                    wrist_T_raw = wrist_T_adj

                wrist_T = wrist_T_raw.cpu().numpy() if isinstance(wrist_T_raw, torch.Tensor) else wrist_T_raw
                hand_q = hand_q_raw.cpu().numpy() if isinstance(hand_q_raw, torch.Tensor) else hand_q_raw
                contacts_out = (pred["contacts"].cpu().numpy()
                                if isinstance(pred["contacts"], torch.Tensor) else np.asarray(pred["contacts"]))

                # Centered -> object body frame: add the body centroid so the
                # saved grasp lands in the frame the renderer/scorer use.
                if obj_name in obj_centroids:
                    c = obj_centroids[obj_name].cpu().numpy()
                    wrist_T = wrist_T.copy()
                    wrist_T[:3, 3] = wrist_T[:3, 3] + c
                    contacts_out = contacts_out + c

                R = wrist_T[:3, :3]
                p = wrist_T[:3, 3]
                quat_xyzw = Rotation.from_matrix(R).as_quat()
                q_star = np.concatenate([
                    [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
                    p,
                    hand_q,
                ])

                data = {
                    "object_name": obj_name,
                    # `object` + `hand` + `X_WO` make the meta directly consumable
                    # by frogger.renderer.render_from_meta and the validators
                    # (gagrasp_drake / smoke_test). q_star and contacts are saved
                    # in the object body frame (centroid added above), so the
                    # object sits at the identity pose in that same frame.
                    "object": obj_name,
                    "hand": args.hand,
                    "X_WO": {"translation": [0.0, 0.0, 0.0],
                             "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
                    "variant": args.variant,
                    "rank": rank,
                    "score": pred["score"],
                    "wrist_pose": _to_serializable(wrist_T),
                    "hand_q": _to_serializable(hand_q),
                    "q_star": _to_serializable(q_star),
                    "contacts": _to_serializable(contacts_out),
                    "forces": _to_serializable(pred["forces"]),
                }

                path = out_dir / f"{obj_name}__grasp__{rank:02d}.json"
                path.write_text(json.dumps(data, indent=2))
                print(f"  {path.name}: score={pred['score']:.4f}")

                # Paired pre-TTO (raw decoded) grasp for the same candidate.
                if args.save_pre_tto and "_wrist_raw" in pred:
                    wpre = pred["_wrist_raw"]
                    wpre = wpre.cpu().numpy() if isinstance(wpre, torch.Tensor) else np.asarray(wpre)
                    hqpre = pred["_hand_q_raw"]
                    hqpre = hqpre.cpu().numpy() if isinstance(hqpre, torch.Tensor) else np.asarray(hqpre)
                    wpre = wpre.copy()
                    if obj_name in obj_centroids:
                        wpre[:3, 3] = wpre[:3, 3] + c   # same body-frame shift as post
                    quat_pre = Rotation.from_matrix(wpre[:3, :3]).as_quat()
                    q_star_pre = np.concatenate([
                        [quat_pre[3], quat_pre[0], quat_pre[1], quat_pre[2]],
                        wpre[:3, 3], hqpre,
                    ])
                    data_pre = {
                        **data, "stage": "pre_tto",
                        "wrist_pose": _to_serializable(wpre),
                        "hand_q": _to_serializable(hqpre),
                        "q_star": _to_serializable(q_star_pre),
                    }
                    (out_dir / f"{obj_name}__grasp_pre__{rank:02d}.json").write_text(
                        json.dumps(data_pre, indent=2))

    print(f"\nGrasps saved to {args.out}")


if __name__ == "__main__":
    main()
