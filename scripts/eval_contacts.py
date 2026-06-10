#!/usr/bin/env python3
"""Evaluate contact prediction quality of a trained EquiDexFlow model.

For each test object the script draws *num_samples* pose candidates via
model.sample(), then measures how closely the predicted per-finger contact
positions match the ground-truth contacts stored in the database.

Metrics reported
----------------
* Per-finger mean contact position error (metres)
* Per-finger std
* Contact coverage rate - fraction of fingers whose predicted contact
  lies within 1 cm of any GT contact for that finger

Usage
-----
    python eval_contacts.py \\
        --config configs/equidexflow_dex_full.yml \\
        --checkpoint path/to/checkpoint.pt \\
        --split test \\
        [--num_samples 10] \\
        [--output results_contacts.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from equidexflow.loaders.schema import N_FINGERS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _aggregate_per_finger(
    data: torch.Tensor,       # (B, M, D)
    finger_ids: torch.Tensor, # (B, M)
    valid_mask: torch.Tensor, # (B, M) bool
    n_fingers: int = N_FINGERS,
) -> torch.Tensor:
    """Returns (B, n_fingers, D), NaN where no GT for that finger."""
    B, M, D = data.shape
    result = torch.full((B, n_fingers, D), float("nan"), dtype=data.dtype, device=data.device)
    for f in range(n_fingers):
        mask = valid_mask & (finger_ids == f)
        count = mask.float().sum(dim=1)
        has = count > 0
        if has.any():
            weighted = (data * mask.unsqueeze(-1).float()).sum(dim=1)
            mean = weighted / count.unsqueeze(-1).clamp(min=1.0)
            result[has, f] = mean[has]
    return result


def evaluate_contacts(model, loader, device, num_samples: int,
                      scorer=None, coverage_threshold: float = 0.01):
    """Top-1 predicted grasp vs nearest GT grasp, per test object.

    Two-pass evaluation that handles multi-modal GT correctly:
      Pass 1 - collect all GT per-finger contacts grouped by object name.
      Pass 2 - for each unique object, generate candidates, rank with
               *scorer* to pick top-1, then measure distance to the
               nearest GT grasp (min over all GT grasps for that object).

    Reported as: "Contact error of the top-ranked candidate against the
    nearest ground-truth grasp for each test object."

    Returns
    -------
    finger_errors : (5,) mean position error per finger (metres)
    finger_stds   : (5,) std
    coverage_rate : (5,) fraction of objects within coverage_threshold
    """
    from collections import defaultdict

    gt_by_object: dict[str, list[torch.Tensor]] = defaultdict(list)
    pts_by_object: dict[str, torch.Tensor] = {}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting GT"):
            batch = _to_device(batch, device)
            obj_pts = batch["object_points"]  # (B, 3, N)
            B = obj_pts.shape[0]
            names = batch["object_name"]  # list[str], length B

            gt_per_finger = _aggregate_per_finger(
                batch["contacts"], batch["finger_ids"], batch["valid_mask"]
            )  # (B, 5, 3)

            for b_idx in range(B):
                name = names[b_idx]
                gt_by_object[name].append(gt_per_finger[b_idx].cpu())
                if name not in pts_by_object:
                    pts_by_object[name] = obj_pts[b_idx].cpu()

        finger_errors: list[list[float]] = [[] for _ in range(N_FINGERS)]
        coverage_hits: list[list[float]] = [[] for _ in range(N_FINGERS)]
        objects = sorted(gt_by_object.keys())
        print(f"Evaluating {len(objects)} unique objects, "
              f"{sum(len(v) for v in gt_by_object.values())} total GT grasps")

        for name in tqdm(objects, desc="Evaluating objects"):
            obj_pts_1 = pts_by_object[name].unsqueeze(0).to(device)  # (1, 3, N)
            obj_pts_n3 = obj_pts_1.permute(0, 2, 1)  # (1, N, 3)

            preds = model.sample(obj_pts_1, num_samples)

            if scorer is not None:
                ranked = scorer.rank_candidates(preds, obj_pts_n3[0])
                pred_c = ranked[0]["contacts"].to(device)  # (5, 3)
            else:
                pred_c = preds[0]["contacts"].to(device)

            all_gt = gt_by_object[name]  # list of (5, 3) tensors

            for f in range(N_FINGERS):
                pred_f = pred_c[f]
                min_err = float("inf")
                any_valid = False
                best_hit = 0.0

                for gt_c in all_gt:
                    gt_f = gt_c[f].to(device)
                    if not torch.isfinite(gt_f).all():
                        continue
                    any_valid = True
                    err = (pred_f - gt_f).norm().item()
                    if err < min_err:
                        min_err = err
                    if err < coverage_threshold:
                        best_hit = 1.0

                if any_valid:
                    finger_errors[f].append(min_err)
                    coverage_hits[f].append(best_hit)

    mean_errors = np.array([np.mean(e) if e else float("nan") for e in finger_errors])
    std_errors  = np.array([np.std(e)  if e else float("nan") for e in finger_errors])
    cov_rate    = np.array([np.mean(h) if h else float("nan") for h in coverage_hits])

    return mean_errors, std_errors, cov_rate


def print_results_table(mean_errors, std_errors, cov_rate):
    finger_names = ["Thumb", "Index", "Middle", "Ring"][:N_FINGERS]
    print("\n" + "=" * 68)
    print(f"{'Finger':<10} {'Mean Err (m)':>14} {'Std (m)':>10} {'Coverage (%)':>14}")
    print("-" * 68)
    for i, name in enumerate(finger_names):
        print(
            f"{name:<10} {mean_errors[i]:>14.4f} {std_errors[i]:>10.4f} {cov_rate[i]*100:>13.1f}%"
        )
    print("-" * 68)
    print(
        f"{'Overall':<10} {np.nanmean(mean_errors):>14.4f} "
        f"{np.nanmean(std_errors):>10.4f} {np.nanmean(cov_rate)*100:>13.1f}%"
    )
    print("=" * 68 + "\n")


def save_csv(path: str, mean_errors, std_errors, cov_rate):
    finger_names = ["Thumb", "Index", "Middle", "Ring"][:N_FINGERS]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["finger", "mean_error_m", "std_error_m", "coverage_rate"])
        for i, name in enumerate(finger_names):
            writer.writerow([name, mean_errors[i], std_errors[i], cov_rate[i]])
    print(f"Saved results to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate contact prediction quality.")
    parser.add_argument("--config",      required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint",  required=True, help="Path to .pt checkpoint")
    parser.add_argument("--split",       default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Override evaluation.num_samples from config")
    parser.add_argument("--device",      default="0")
    parser.add_argument("--output",      default=None, help="Optional CSV output path")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Device
    device = torch.device(
        "cpu" if args.device == "cpu" else f"cuda:{args.device}"
    )

    # Model
    from equidexflow.models import get_dex_model
    model = get_dex_model(
        p_uncond=float(cfg.model.get("p_uncond", 0.1)),
        guidance=float(cfg.model.get("guidance", 2.0)),
        num_ode_steps=int(cfg.model.get("num_ode_steps", 10)),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Data
    from equidexflow.loaders import get_dataloader
    split_cfg = cfg.data[args.split]
    loader = get_dataloader(args.split, split_cfg)

    num_samples = args.num_samples or int(cfg.evaluation.num_samples)

    from equidexflow.physics.scorer import GraspScorer
    pw = cfg.evaluation.get("physics_weights", {})
    scorer = GraspScorer(
        mu=float(cfg.data.get("mu", 0.5)),
        object_mass=float(cfg.data.get("object_mass", 0.2)),
        beta1=float(pw.get("beta1", 1.0)),
        beta2=float(pw.get("beta2", 2.0)),
        beta3=float(pw.get("beta3", 1.0)),
        beta4=float(pw.get("beta4", 0.5)),
    )

    # Evaluate (top-1 ranked candidate per object)
    mean_errors, std_errors, cov_rate = evaluate_contacts(
        model, loader, device, num_samples, scorer=scorer
    )

    print_results_table(mean_errors, std_errors, cov_rate)

    if args.output:
        save_csv(args.output, mean_errors, std_errors, cov_rate)


if __name__ == "__main__":
    main()
