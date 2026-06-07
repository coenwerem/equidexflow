#!/usr/bin/env python3
"""Inference-time ablations for EquiDexFlow (no retraining required).

Three ablations on the Full-variant checkpoint:
  1. Cone projection OFF  — raw force vectors, no friction cone enforcement
  2. Wrist projection ON  — apply SDF-based wrist slide to each candidate
  3. Flow deterministic   — z=0 mode instead of stochastic sampling (hand_q)

All use the same 81-object test set and scoring as run_full_eval.py.

Usage (from any directory, uses MuJoCoDex venv):
    ~/ResearchProjects/MuJoCoDex/.venv/bin/python \
        ~/ResearchProjects/frogger/scripts/equidex/run_inference_ablations.py \
        --device 0
"""
from __future__ import annotations

import os
import sys

_EDF_DIR = os.environ.get("EQUIDEXFLOW_OUTPUTS_DIR", os.getcwd())

import argparse
import csv
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from equidexflow.models import get_dex_model
from equidexflow.models.force_decoder import cone_project
from equidexflow.loaders import get_dataloader
from equidexflow.physics.scorer import GraspScorer
from equidexflow.kinematics import friction_cone_violation_rate
from equidexflow.loaders.schema import N_FINGERS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FROGGER_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = FROGGER_ROOT / "outputs" / "paper_results" / "equidex" / "inference_ablations"

FULL_VARIANT = "equidexflow_leap_full_fc_reach_flow"
EQUIDEXFLOW_DIR = Path(_EDF_DIR)

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
    # Pick the run whose checkpoint_best.pt is newest by mtime (alphabetical
    # sort wrongly prefers stale rf_* over the fixed jl_* run).
    base = EQUIDEXFLOW_DIR / "outputs" / "training_results" / variant
    runs = [p for p in base.iterdir() if p.is_dir() and (p / "checkpoint_best.pt").exists()]
    return max(runs, key=lambda p: (p / "checkpoint_best.pt").stat().st_mtime)


# ---------------------------------------------------------------------------
# Ablation 1: Cone projection OFF
# ---------------------------------------------------------------------------

def patch_cone_off(model) -> Callable:
    """Monkey-patch ForceDecoder to skip cone projection. Returns restore fn."""
    fd = model.force_decoder
    original_forward = fd.forward

    def forward_no_cone(features, contact_positions, contact_normals, return_local=False):
        if features.dim() == 4:
            features = features.squeeze(-1)
        B, C, _ = features.shape
        nf = fd.n_fingers
        z_per = features.unsqueeze(1).expand(B, nf, C, 3)
        c_chan = contact_positions.unsqueeze(2)
        z_cond = torch.cat([z_per, c_chan], dim=2)
        z_flat = z_cond.reshape(B * nf, C + 1, 3)
        x = fd.vn_hidden(z_flat)
        f_raw = fd.vn_out(x).squeeze(1)
        f_raw = f_raw.reshape(B, nf, 3)
        # Skip cone_project — return raw forces
        if return_local:
            return f_raw, None
        return f_raw

    fd.forward = forward_no_cone
    return lambda: setattr(fd, 'forward', original_forward)


# ---------------------------------------------------------------------------
# Ablation 3: Flow deterministic (z=0 mode)
# ---------------------------------------------------------------------------

def patch_flow_deterministic(model) -> Callable:
    """Monkey-patch hand_q_decoder.sample to use forward (z=0 mode)."""
    hqd = model.hand_q_decoder
    if not hasattr(hqd, 'sample'):
        print("  WARNING: hand_q_decoder has no sample(); already deterministic")
        return lambda: None

    original_sample = hqd.sample

    def deterministic_sample(z, wrist_pose, temperature=1.0):
        return hqd.forward(z, wrist_pose)

    hqd.sample = deterministic_sample
    return lambda: setattr(hqd, 'sample', original_sample)


# ---------------------------------------------------------------------------
# Evaluation (same logic as run_full_eval.py)
# ---------------------------------------------------------------------------

def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def evaluate_model(
    model, loader, device: torch.device, num_samples: int, scorer: GraspScorer,
    mu: float = 0.5,
) -> dict:
    """Run physics-scoring eval identical to run_full_eval's rollout eval."""
    pts_by_object: dict[str, torch.Tensor] = {}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Collecting objects", leave=False):
            batch = _to_device(batch, device)
            names = batch["object_name"]
            obj_pts = batch["object_points"]
            for b in range(obj_pts.shape[0]):
                name = names[b]
                if name not in pts_by_object:
                    pts_by_object[name] = obj_pts[b].cpu()

        top1_scores: list[float] = []
        top3_scores: list[float] = []
        fvr_per_pred: list[float] = []
        objects = sorted(pts_by_object.keys())

        for name in tqdm(objects, desc="  Scoring objects", leave=False):
            obj_pts_1 = pts_by_object[name].unsqueeze(0).to(device)
            obj_pts_n3 = obj_pts_1.permute(0, 2, 1)

            preds = model.sample(obj_pts_1, num_samples)
            ranked = scorer.rank_candidates(preds, obj_pts_n3[0])
            scores = [c["score"] for c in ranked]

            top1_scores.append(scores[0])
            top3 = float(np.mean(scores[:min(3, len(scores))]))
            top3_scores.append(top3)

            best = ranked[0]
            contacts = best["contacts"].unsqueeze(0).to(device)
            forces = best["forces"].unsqueeze(0).to(device)
            obj_centroid = obj_pts_1.mean(dim=-1).unsqueeze(0)
            normals = F.normalize(obj_centroid - contacts + 1e-8, dim=-1)
            valid = torch.ones(1, N_FINGERS, dtype=torch.bool, device=device)
            rate = friction_cone_violation_rate(forces, normals, valid, mu=mu)
            fvr_per_pred.append(rate.item())

    return {
        "top1_scores": top1_scores,
        "top3_scores": top3_scores,
        "fvr_per_pred": fvr_per_pred,
        "n_objects": len(objects),
    }


def save_results(out_dir: Path, name: str, results: dict):
    """Save per-object CSV and summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}_rollout.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["object_idx", "top1_score", "top3_score", "friction_violation_rate"])
        for i, (s1, s3, fvr) in enumerate(zip(
            results["top1_scores"], results["top3_scores"], results["fvr_per_pred"]
        )):
            writer.writerow([i, s1, s3, fvr])
    print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="0")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--ablations", nargs="+",
                        default=["cone_off", "flow_deterministic"],
                        choices=["cone_off", "flow_deterministic", "baseline"],
                        help="Which ablations to run")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

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
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # Build test loader
    test_cfg = OmegaConf.create(TEST_CONFIG)
    loader = get_dataloader("test", test_cfg)
    print(f"Test set: {len(loader.dataset)} grasps, 81 objects")

    scorer = GraspScorer(mu=0.5, object_mass=0.2,
                         beta1=1.0, beta2=2.0, beta3=1.0, beta4=0.5)

    all_results = {}

    # --- Baseline (Full, unmodified) ---
    if "baseline" in args.ablations:
        print("\n=== Baseline (Full, unmodified) ===")
        results = evaluate_model(model, loader, device, args.num_samples, scorer)
        save_results(args.out, "baseline", results)
        all_results["baseline"] = {
            "top1_mean": float(np.mean(results["top1_scores"])),
            "top3_mean": float(np.mean(results["top3_scores"])),
            "fvr_mean": float(np.mean(results["fvr_per_pred"])),
            "n_objects": results["n_objects"],
        }
        print(f"  Top-1: {all_results['baseline']['top1_mean']:.4f}")
        print(f"  Top-3: {all_results['baseline']['top3_mean']:.4f}")
        print(f"  FVR:   {all_results['baseline']['fvr_mean']*100:.1f}%")

    # --- Ablation 1: Cone projection OFF ---
    if "cone_off" in args.ablations:
        print("\n=== Ablation: Cone Projection OFF ===")
        restore = patch_cone_off(model)
        results = evaluate_model(model, loader, device, args.num_samples, scorer)
        restore()
        save_results(args.out, "cone_off", results)
        all_results["cone_off"] = {
            "top1_mean": float(np.mean(results["top1_scores"])),
            "top3_mean": float(np.mean(results["top3_scores"])),
            "fvr_mean": float(np.mean(results["fvr_per_pred"])),
            "n_objects": results["n_objects"],
        }
        print(f"  Top-1: {all_results['cone_off']['top1_mean']:.4f}")
        print(f"  Top-3: {all_results['cone_off']['top3_mean']:.4f}")
        print(f"  FVR:   {all_results['cone_off']['fvr_mean']*100:.1f}%")

    # --- Ablation 3: Flow deterministic (z=0) ---
    if "flow_deterministic" in args.ablations:
        print("\n=== Ablation: Flow Deterministic (z=0 mode) ===")
        restore = patch_flow_deterministic(model)
        results = evaluate_model(model, loader, device, args.num_samples, scorer)
        restore()
        save_results(args.out, "flow_deterministic", results)
        all_results["flow_deterministic"] = {
            "top1_mean": float(np.mean(results["top1_scores"])),
            "top3_mean": float(np.mean(results["top3_scores"])),
            "fvr_mean": float(np.mean(results["fvr_per_pred"])),
            "n_objects": results["n_objects"],
        }
        print(f"  Top-1: {all_results['flow_deterministic']['top1_mean']:.4f}")
        print(f"  Top-3: {all_results['flow_deterministic']['top3_mean']:.4f}")
        print(f"  FVR:   {all_results['flow_deterministic']['fvr_mean']*100:.1f}%")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("INFERENCE ABLATION SUMMARY")
    print("=" * 60)

    # Load existing Full results for comparison
    existing_full = FROGGER_ROOT / "outputs" / "paper_results" / "equidex" / "equidex_results" / FULL_VARIANT / "rollout.csv"
    if existing_full.exists():
        full_top1, full_top3, full_fvr = [], [], []
        with open(existing_full) as f:
            reader = csv.DictReader(f)
            for row in reader:
                full_top1.append(float(row["top1_score"]))
                full_top3.append(float(row["top3_score"]))
                full_fvr.append(float(row["friction_violation_rate"]))
        ref = {
            "top1_mean": float(np.mean(full_top1)),
            "top3_mean": float(np.mean(full_top3)),
            "fvr_mean": float(np.mean(full_fvr)),
        }
        all_results["full_reference"] = ref
        print(f"\n  Full (reference):    Top-1={ref['top1_mean']:.4f}  "
              f"Top-3={ref['top3_mean']:.4f}  FVR={ref['fvr_mean']*100:.1f}%")

    for name, r in all_results.items():
        if name == "full_reference":
            continue
        print(f"  {name:22s}: Top-1={r['top1_mean']:.4f}  "
              f"Top-3={r['top3_mean']:.4f}  FVR={r['fvr_mean']*100:.1f}%")

    # Save summary YAML
    summary_path = args.out / "inference_ablations_summary.yaml"
    args.out.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        yaml.safe_dump({
            "description": "Inference-time ablations on Full checkpoint (no retraining)",
            "checkpoint": str(cp),
            "num_samples": args.num_samples,
            "test_set": "81 objects",
            "results": all_results,
        }, f, sort_keys=False)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
