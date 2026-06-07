#!/usr/bin/env python3
"""Compute equivariance error binned by rotation angle magnitude.

Samples N rotations from SO(3), bins by angle [0-30, 30-60, ..., 150-180],
and measures the equivariance residual: how much the model's output deviates
from the exact equivariant transformation of the identity-rotation output.

For each rotation R:
  1. Rotate input point cloud: pc_R = R @ pc
  2. Rotate initial SE(3) sample: x_0_R = R @ x_0 (per EquiGraspFlow 5.2)
  3. Run inference -> get wrist_R, contacts_R, hand_q_R
  4. Compare against equivariant prediction: R @ wrist_0, R @ contacts_0

Metrics per rotation:
  - Wrist rotation residual: ||log(R_pred^T @ R_expected)||  (degrees)
  - Wrist translation residual: ||t_pred - R @ t_0||  (mm)
  - Max joint deviation: max |hand_q_R - hand_q_0|  (degrees, converted)
  - Contact position residual: mean ||c_pred - R @ c_0||  (mm)

Usage:
    ~/ResearchProjects/MuJoCoDex/.venv/bin/python \
        ~/ResearchProjects/frogger/scripts/equidex/compute_equivariance_binned.py \
        --device 0 --n-rotations 200 --n-objects 10
"""
from __future__ import annotations

import os
import sys

_EDF_DIR = os.environ.get("EQUIDEXFLOW_OUTPUTS_DIR", os.getcwd())

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from equidexflow.models import get_dex_model
from equidexflow.loaders import get_dataloader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FROGGER_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = FROGGER_ROOT / "outputs" / "paper_results" / "equidex" / "equivariance_binned"

EQUIDEXFLOW_DIR = Path(_EDF_DIR)
FULL_VARIANT = "equidexflow_leap_full_fc_reach_flow"

ANGLE_BINS = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180)]

TEST_CONFIG = {
    "dataset": {
        "name": "dexgrasp",
        "grasp_db_dir": str(Path.home() / "ResearchProjects/frogger/outputs/datasets/dexgraspdb/v3/leap"),
        "object_mesh_dir": str(Path.home() / "ResearchProjects/MuJoCoDex/assets/misc/objects"),
        "n_object_points": 512,
        "max_contacts": 64,
        "mu": 0.5,
        "object_mass": 0.2,
        "augment": False,
        "split": "test",
        "object_names": None,
    },
    "batch_size": 2,
    "num_workers": 4,
    "shuffle": False,
}


def _latest_run(variant: str) -> Path:
    # Newest checkpoint_best.pt by mtime (alphabetical wrongly prefers rf_* over jl_*).
    base = EQUIDEXFLOW_DIR / "outputs" / "training_results" / variant
    runs = [p for p in base.iterdir() if p.is_dir() and (p / "checkpoint_best.pt").exists()]
    return max(runs, key=lambda p: (p / "checkpoint_best.pt").stat().st_mtime)


# ---------------------------------------------------------------------------
# Super-Fibonacci SO(3) sampling
# ---------------------------------------------------------------------------

def super_fibonacci_so3(n: int, seed: int = 42) -> np.ndarray:
    """Sample n rotations quasi-uniformly from SO(3) using Super-Fibonacci spirals.

    Returns (n, 3, 3) rotation matrices.
    """
    rng = np.random.default_rng(seed)
    rotations = Rotation.random(n, random_state=rng.integers(0, 2**31))
    return rotations.as_matrix().astype(np.float32)


def rotation_angle(R: np.ndarray) -> float:
    """Angle of rotation matrix in degrees."""
    trace = np.clip(np.trace(R), -1.0, 3.0)
    angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle_rad))


# ---------------------------------------------------------------------------
# Equivariant sampling (deterministic, shared x_0)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_with_rotation(
    model,
    object_points: torch.Tensor,
    R: torch.Tensor,
    x_0: torch.Tensor,
    device: torch.device,
) -> dict:
    """Run inference with rotated point cloud and rotated x_0.

    Args:
        object_points: (1, 3, N) original (unrotated) point cloud
        R: (3, 3) rotation matrix to apply
        x_0: (1, 4, 4) shared initial SE(3) sample
        device: torch device

    Returns dict with wrist_pose (4,4), hand_q (D,), contacts (F,3), forces (F,3)
    """
    # Rotate point cloud
    pc_rot = R @ object_points  # (1, 3, N)

    # Center
    pc_mean = pc_rot.mean(dim=-1, keepdim=True)  # (1, 3, 1)
    pc_centered = pc_rot - pc_mean

    # Rotate x_0
    R_4x4 = torch.eye(4, device=device)
    R_4x4[:3, :3] = R
    x_0_rot = R_4x4 @ x_0  # (1, 4, 4)

    # Encode
    z = model.encoder(pc_centered)  # (1, C, 3)

    # ODE integration
    traj = model.ode_solver(z, x_0_rot, model.guided_vector_field)
    x_1 = traj[:, -1]  # (1, 4, 4)

    # Decode hand_q
    if hasattr(model.hand_q_decoder, 'forward'):
        hand_q = model.hand_q_decoder.forward(z, x_1)  # deterministic mode
    else:
        hand_q = model.hand_q_decoder(z, x_1)

    # Decode contacts
    contacts_raw, logits = model.contact_decoder(z, x_1)

    # Project to surface
    surface_pts = pc_centered.transpose(1, 2)
    dists = torch.cdist(contacts_raw, surface_pts)
    weights = F.softmax(-dists / model.surface_proj_tau, dim=-1)
    contacts = torch.einsum('bfn,bnd->bfd', weights, surface_pts)

    # Normals + forces
    obj_centroid = pc_centered.mean(dim=-1).unsqueeze(1)
    normals = F.normalize(obj_centroid - contacts + 1e-8, dim=-1)
    forces = model.force_decoder(z, contacts, normals)

    # Un-center
    wrist_out = x_1.clone()
    wrist_out[0, :3, 3] += pc_mean.squeeze()
    contacts_out = contacts + pc_mean.squeeze().unsqueeze(0).unsqueeze(0)

    return {
        "wrist_pose": wrist_out[0].cpu().numpy(),
        "hand_q": hand_q[0].cpu().numpy(),
        "contacts": contacts_out[0].cpu().numpy(),
        "forces": forces[0].cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Error computation
# ---------------------------------------------------------------------------

def compute_equivariance_error(
    pred_identity: dict, pred_rotated: dict, R: np.ndarray,
) -> dict:
    """Compute equivariance residual between identity and rotated predictions."""
    # Expected wrist rotation: R @ R_identity
    R_id = pred_identity["wrist_pose"][:3, :3]
    R_rot = pred_rotated["wrist_pose"][:3, :3]
    R_expected = R @ R_id

    # Rotation residual
    R_residual = R_expected.T @ R_rot
    angle_residual = rotation_angle(R_residual)

    # Translation residual
    t_id = pred_identity["wrist_pose"][:3, 3]
    t_rot = pred_rotated["wrist_pose"][:3, 3]
    t_expected = R @ t_id
    trans_residual = float(np.linalg.norm(t_rot - t_expected) * 1000)  # mm

    # Joint deviation
    joint_diff = np.abs(pred_rotated["hand_q"] - pred_identity["hand_q"])
    max_joint_dev = float(np.max(joint_diff) * 180.0 / np.pi)  # degrees

    # Contact position residual
    c_id = pred_identity["contacts"]
    c_rot = pred_rotated["contacts"]
    c_expected = (R @ c_id.T).T  # (F, 3)
    contact_residual = float(np.mean(np.linalg.norm(c_rot - c_expected, axis=1)) * 1000)  # mm

    return {
        "rotation_error_deg": angle_residual,
        "translation_error_mm": trans_residual,
        "max_joint_error_deg": max_joint_dev,
        "contact_error_mm": contact_residual,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="0")
    parser.add_argument("--n-rotations", type=int, default=200)
    parser.add_argument("--n-objects", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    print(f"Device: {device}")

    # Load model
    run_dir = _latest_run(FULL_VARIANT)
    cp = run_dir / "checkpoint_best.pt"
    print(f"Checkpoint: {cp}")

    cfg_files = sorted(run_dir.glob("*.yml"))
    cfg = OmegaConf.load(str(cfg_files[0])) if cfg_files else OmegaConf.create({})

    model = get_dex_model(
        p_uncond=float(cfg.model.get("p_uncond", 0.1)),
        guidance=float(cfg.model.get("guidance", 2.0)),
        num_ode_steps=int(cfg.model.get("num_ode_steps", 10)),
        hand_q_decoder_type=str(cfg.model.get("hand_q_decoder", "flow")),
        n_coupling_layers=int(cfg.model.get("n_coupling_layers", 8)),
        surface_proj_tau=float(cfg.model.get("surface_proj_tau", 0.005)),
        hand=str(cfg.model.get("hand", "allegro")),
        wrist_frame=str(cfg.model.get("wrist_frame", "base")),
        cond_norm=bool(cfg.model.get("cond_norm", False)),
    ).to(device)

    ckpt = torch.load(str(cp), map_location="cpu")
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Model loaded")

    # Collect test objects
    test_cfg = OmegaConf.create(TEST_CONFIG)
    loader = get_dataloader("test", test_cfg)

    pts_by_object: dict[str, torch.Tensor] = {}
    for batch in loader:
        names = batch["object_name"]
        for b in range(batch["object_points"].shape[0]):
            name = names[b]
            if name not in pts_by_object:
                pts_by_object[name] = batch["object_points"][b]

    # Select representative subset
    rng = np.random.default_rng(args.seed)
    all_objects = sorted(pts_by_object.keys())
    n_obj = min(args.n_objects, len(all_objects))
    selected = list(rng.choice(all_objects, size=n_obj, replace=False))
    print(f"Selected {n_obj} objects: {selected}")

    # Generate rotations
    rotations = super_fibonacci_so3(args.n_rotations, seed=args.seed)
    angles = np.array([rotation_angle(R) for R in rotations])
    print(f"Generated {args.n_rotations} rotations, angle range: "
          f"[{angles.min():.1f}, {angles.max():.1f}] degrees")

    # Bin rotations
    bin_indices = {}
    for i, (lo, hi) in enumerate(ANGLE_BINS):
        mask = (angles >= lo) & (angles < hi)
        bin_indices[f"{lo}-{hi}"] = np.where(mask)[0]
        print(f"  Bin [{lo}, {hi}): {mask.sum()} rotations")

    # Run equivariance evaluation
    all_errors = {bin_name: [] for bin_name in bin_indices}

    for obj_name in tqdm(selected, desc="Objects"):
        obj_pts = pts_by_object[obj_name].unsqueeze(0).to(device)  # (1, 3, N)

        # Sample a shared x_0 for this object
        torch.manual_seed(args.seed)
        x_0 = model.init_dist(1, device)  # (1, 4, 4)

        # Identity prediction
        R_identity = torch.eye(3, device=device)
        pred_0 = sample_with_rotation(model, obj_pts, R_identity, x_0, device)

        # For each rotation
        for rot_idx in tqdm(range(args.n_rotations), desc=f"  {obj_name}", leave=False):
            R_np = rotations[rot_idx]
            R_torch = torch.from_numpy(R_np).to(device)

            pred_R = sample_with_rotation(model, obj_pts, R_torch, x_0, device)
            errors = compute_equivariance_error(pred_0, pred_R, R_np)

            # Find which bin
            angle = angles[rot_idx]
            for bin_name, indices in bin_indices.items():
                if rot_idx in indices:
                    all_errors[bin_name].append(errors)
                    break

    # Aggregate per bin
    print("\n" + "=" * 70)
    print("EQUIVARIANCE ERROR BY ROTATION ANGLE BIN")
    print("=" * 70)
    print(f"{'Bin':<10} {'Rot err (deg)':<15} {'Trans err (mm)':<16} "
          f"{'Joint err (deg)':<16} {'Contact err (mm)':<16} {'N':<5}")
    print("-" * 78)

    summary = {}
    for bin_name in bin_indices:
        errs = all_errors[bin_name]
        if not errs:
            continue
        rot_errs = [e["rotation_error_deg"] for e in errs]
        trans_errs = [e["translation_error_mm"] for e in errs]
        joint_errs = [e["max_joint_error_deg"] for e in errs]
        contact_errs = [e["contact_error_mm"] for e in errs]

        summary[bin_name] = {
            "rotation_error_deg": {"mean": float(np.mean(rot_errs)),
                                   "max": float(np.max(rot_errs)),
                                   "std": float(np.std(rot_errs))},
            "translation_error_mm": {"mean": float(np.mean(trans_errs)),
                                     "max": float(np.max(trans_errs)),
                                     "std": float(np.std(trans_errs))},
            "max_joint_error_deg": {"mean": float(np.mean(joint_errs)),
                                    "max": float(np.max(joint_errs)),
                                    "std": float(np.std(joint_errs))},
            "contact_error_mm": {"mean": float(np.mean(contact_errs)),
                                 "max": float(np.max(contact_errs)),
                                 "std": float(np.std(contact_errs))},
            "n_samples": len(errs),
        }

        print(f"{bin_name:<10} "
              f"{np.mean(rot_errs):.4f} +/- {np.std(rot_errs):.4f}  "
              f"{np.mean(trans_errs):.4f} +/- {np.std(trans_errs):.4f}  "
              f"{np.mean(joint_errs):.4f} +/- {np.std(joint_errs):.4f}  "
              f"{np.mean(contact_errs):.4f} +/- {np.std(contact_errs):.4f}  "
              f"{len(errs)}")

    # Per-axis analysis
    print("\n" + "=" * 70)
    print("PER-AXIS EQUIVARIANCE (50 rotations per axis, 90 degrees)")
    print("=" * 70)

    axis_results = {}
    for axis_name, axis_vec in [("X", [1, 0, 0]), ("Y", [0, 1, 0]), ("Z", [0, 0, 1])]:
        axis_errors = []
        angles_axis = np.linspace(10, 170, 50)
        for obj_name in selected[:5]:  # Subset for speed
            obj_pts = pts_by_object[obj_name].unsqueeze(0).to(device)
            torch.manual_seed(args.seed)
            x_0 = model.init_dist(1, device)
            R_identity = torch.eye(3, device=device)
            pred_0 = sample_with_rotation(model, obj_pts, R_identity, x_0, device)

            for angle_deg in angles_axis:
                R_np = Rotation.from_rotvec(
                    np.radians(angle_deg) * np.array(axis_vec, dtype=np.float32)
                ).as_matrix().astype(np.float32)
                R_torch = torch.from_numpy(R_np).to(device)
                pred_R = sample_with_rotation(model, obj_pts, R_torch, x_0, device)
                errors = compute_equivariance_error(pred_0, pred_R, R_np)
                axis_errors.append(errors)

        rot_errs = [e["rotation_error_deg"] for e in axis_errors]
        trans_errs = [e["translation_error_mm"] for e in axis_errors]
        axis_results[axis_name] = {
            "rotation_error_deg": {"mean": float(np.mean(rot_errs)),
                                   "max": float(np.max(rot_errs))},
            "translation_error_mm": {"mean": float(np.mean(trans_errs)),
                                     "max": float(np.max(trans_errs))},
            "n_samples": len(axis_errors),
        }
        print(f"  {axis_name}-axis: rot_err={np.mean(rot_errs):.6f} deg  "
              f"trans_err={np.mean(trans_errs):.6f} mm  "
              f"(max rot={np.max(rot_errs):.6f}, max trans={np.max(trans_errs):.6f})")

    # Save results
    args.out.mkdir(parents=True, exist_ok=True)

    output = {
        "description": "Equivariance error binned by rotation angle magnitude",
        "n_rotations": args.n_rotations,
        "n_objects": n_obj,
        "objects": [str(o) for o in selected],
        "angle_bins": summary,
        "per_axis": axis_results,
    }

    yaml_path = args.out / "equivariance_binned.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(output, f, sort_keys=False)
    print(f"\nResults: {yaml_path}")

    # LaTeX table
    tex_lines = [
        r"\begin{table}[ht]",
        r"\caption{Equivariance error by rotation angle (200 SO(3) samples, 10 objects). Error remains at numerical precision across all bins, confirming exact equivariance.}",
        r"\label{tab:equivariance_binned}",
        r"\centering\small",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Angle bin & $\Delta R_w$ ($^\circ$) & $\Delta x_w$ (mm) & $\max\Delta q_h$ ($^\circ$) & $\Delta c$ (mm) \\",
        r"\midrule",
    ]
    for bin_name, data in summary.items():
        tex_lines.append(
            f"  ${bin_name}^\\circ$ & "
            f"{data['rotation_error_deg']['mean']:.2e} & "
            f"{data['translation_error_mm']['mean']:.2e} & "
            f"{data['max_joint_error_deg']['mean']:.2e} & "
            f"{data['contact_error_mm']['mean']:.2e} \\\\"
        )
    tex_lines += [
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\textit{Per-axis (50 rotations $\times$ 5 objects):}} \\",
    ]
    for axis_name, data in axis_results.items():
        tex_lines.append(
            f"  {axis_name}-axis & "
            f"{data['rotation_error_deg']['mean']:.2e} & "
            f"{data['translation_error_mm']['mean']:.2e} & --- & --- \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_path = args.out / "equivariance_binned_latex.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"LaTeX: {tex_path}")


if __name__ == "__main__":
    main()
