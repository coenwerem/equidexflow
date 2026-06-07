#!/usr/bin/env python3
"""Evaluate grasp candidates via physics scoring (offline rollout proxy).

For each test object the script samples K candidate grasps with model.sample(),
ranks them with GraspScorer.rank_candidates(), and reports physics quality.

Metrics reported
----------------
* Top-1 physics score  (higher is better)
* Top-3 mean physics score
* Mean friction violation rate (%)

Usage
-----
    python eval_rollout.py \\
        --config configs/equidexflow_dex_full.yml \\
        --checkpoint path/to/checkpoint.pt \\
        [--num_samples 10] \\
        [--split test] \\
        [--output results_rollout.csv]
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

from equidexflow.physics.scorer import GraspScorer
from equidexflow.kinematics import friction_cone_violation_rate
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def evaluate_rollout(
    model,
    loader,
    device: torch.device,
    num_samples: int,
    scorer: GraspScorer,
    mu: float = 0.5,
):
    """Per-object physics-scoring evaluation.

    Collects unique objects from the loader, generates candidates once per
    object, ranks with *scorer*, and reports top-1/top-3 physics scores.

    Returns
    -------
    top1_scores  : list[float]  top-1 physics score per unique object
    top3_scores  : list[float]  mean of top-3 per unique object
    fvr_per_pred : list[float]  friction violation rate per top-1 candidate
    """
    from equidexflow.loaders.schema import N_FINGERS

    pts_by_object: dict[str, torch.Tensor] = {}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting objects"):
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
        print(f"Evaluating {len(objects)} unique objects")

        for name in tqdm(objects, desc="Scoring objects"):
            obj_pts_1 = pts_by_object[name].unsqueeze(0).to(device)
            obj_pts_n3 = obj_pts_1.permute(0, 2, 1)

            preds = model.sample(obj_pts_1, num_samples)
            ranked = scorer.rank_candidates(preds, obj_pts_n3[0])
            scores = [c["score"] for c in ranked]

            top1_scores.append(scores[0])
            top3 = float(np.mean(scores[: min(3, len(scores))]))
            top3_scores.append(top3)

            best = ranked[0]
            contacts = best["contacts"].unsqueeze(0).to(device)
            forces = best["forces"].unsqueeze(0).to(device)
            obj_centroid = obj_pts_1.mean(dim=-1).unsqueeze(0)
            normals = F.normalize(obj_centroid - contacts + 1e-8, dim=-1)
            valid = torch.ones(1, N_FINGERS, dtype=torch.bool, device=device)
            rate = friction_cone_violation_rate(forces, normals, valid, mu=mu)
            fvr_per_pred.append(rate.item())

    return top1_scores, top3_scores, fvr_per_pred


def print_results_table(top1_scores, top3_scores, fvr_per_pred):
    print("\n" + "=" * 48)
    print("Physics Scoring Results (offline rollout proxy)")
    print("-" * 48)
    print(f"  Objects evaluated    : {len(top1_scores)}")
    print(f"  Top-1 physics score  : {np.mean(top1_scores):.4f} ± {np.std(top1_scores):.4f}")
    print(f"  Top-3 physics score  : {np.mean(top3_scores):.4f} ± {np.std(top3_scores):.4f}")
    print(f"  Friction viol. rate  : {np.mean(fvr_per_pred)*100:.1f}%")
    print("=" * 48 + "\n")


def save_csv(path: str, top1_scores, top3_scores, fvr_per_pred):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["object_idx", "top1_score", "top3_score", "friction_violation_rate"])
        for i, (s1, s3, fvr) in enumerate(zip(top1_scores, top3_scores, fvr_per_pred)):
            writer.writerow([i, s1, s3, fvr])
    print(f"Saved results to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate grasp candidates via physics scoring.")
    parser.add_argument("--config",      required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--split",       default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--device",      default="0")
    parser.add_argument("--output",      default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")

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

    from equidexflow.loaders import get_dataloader
    split_cfg = cfg.data[args.split]
    loader = get_dataloader(args.split, split_cfg)

    num_samples = args.num_samples or int(cfg.evaluation.num_samples)
    mu          = float(cfg.data.get("mu", 0.5))
    obj_mass    = float(cfg.data.get("object_mass", 0.2))

    eval_cfg = cfg.evaluation
    phys_w   = eval_cfg.physics_weights
    scorer = GraspScorer(
        beta1=float(phys_w.beta1),
        beta2=float(phys_w.beta2),
        beta3=float(phys_w.beta3),
        beta4=float(phys_w.beta4),
        mu=mu,
        object_mass=obj_mass,
    )

    top1, top3, fvr = evaluate_rollout(model, loader, device, num_samples, scorer, mu=mu)

    print_results_table(top1, top3, fvr)

    if args.output:
        save_csv(args.output, top1, top3, fvr)


if __name__ == "__main__":
    main()
