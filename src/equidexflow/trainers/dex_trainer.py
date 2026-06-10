"""DexGraspTrainer - training and evaluation loop for EquiDexFlow.

Orchestrates:
  • full training loop with per-step weighted loss computation
  • periodic validation and evaluation passes
  • contact / force / friction metrics computed via model.sample()
  • checkpoint saving in the canonical format

Design notes
------------
* model.forward(batch) returns supervised losses plus predicted contact/force
    tensors used for differentiable physics regularization.
* The per-finger valid mask is derived from dataset finger assignments. If no
    per-finger data is available, physics losses evaluate to zero rather than
    producing a spurious training signal.
* evaluate() calls model.sample() (ODE-based - slow) only at eval_interval steps.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm

from equidexflow.losses.physics_loss import physics_loss
from equidexflow.kinematics import friction_cone_violation_rate
from equidexflow.physics.scorer import GraspScorer
from equidexflow.utils.average_meter import AverageMeter


# ---------------------------------------------------------------------------
# Public trainer class
# ---------------------------------------------------------------------------

class DexGraspTrainer:
    """Training / evaluation loop for dexterous grasp generation."""

    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader,
        val_loader,
        test_loader,
        logger,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.logger = logger

        self.device: torch.device = next(model.parameters()).device

        # ## training hyper-parameters ####################################
        tr = cfg.training
        self.num_epochs: int = int(tr.epochs)
        self.print_interval: int = int(tr.print_interval)
        self.val_interval: int = int(tr.val_interval)
        self.eval_interval: int = int(tr.eval_interval)
        self.save_interval: int = int(tr.save_interval)
        self.num_eval_samples: int = int(cfg.evaluation.num_samples)

        # Loss weights as plain Python floats
        lw = tr.loss_weights
        self.lw: dict[str, float] = {
            "flow":              float(lw.flow),
            "hand_q":            float(lw.hand_q),
            "contact":           float(lw.contact),
            "normal":            float(getattr(lw, "normal", 1.0)),
            "force":             float(lw.force),
            "reach":             float(getattr(lw, "reach", 0.0)),
            "physics_wrench":    float(lw.physics_wrench),
            "physics_friction":  float(lw.physics_friction),
            "physics_collision": float(lw.physics_collision),
        }

        # Data params (used for physics loss)
        self.mu: float = float(getattr(cfg.data, "mu", 0.5))
        self.object_mass: float = float(getattr(cfg.data, "object_mass", 0.2))

        # Checkpoint state
        self.best_val_loss: float = float("inf")
        self.logdir: str = logger.writer.file_writer.get_logdir()
        os.makedirs(self.logdir, exist_ok=True)

        # ## running-average meters ########################################
        self._loss_keys = [
            "flow", "hand_q", "contact", "normal", "force", "reach",
            "physics_wrench", "physics_friction", "physics_collision", "total",
        ]
        self.train_meters = {k: AverageMeter() for k in self._loss_keys}
        self.val_meters   = {k: AverageMeter() for k in self._loss_keys}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main epoch loop."""
        step = 0
        for epoch in range(1, self.num_epochs + 1):
            for batch in self.train_loader:
                step += 1

                self.train(batch)

                if step % self.print_interval == 0:
                    self._log_and_print_meters("train", epoch, step)

                if step % self.val_interval == 0:
                    self.validate(epoch, step)

                if step % self.eval_interval == 0:
                    self.evaluate(epoch, step)

                if step % self.save_interval == 0:
                    self.save(f"step_{step:07d}")

    def train(self, batch: dict) -> dict:
        """One training step.  Returns dict of scalar loss values."""
        self.model.train()
        self.optimizer.zero_grad()

        batch = self._to_device(batch)
        losses = self._compute_losses(batch)
        losses["total"].backward()
        self.optimizer.step()

        B = batch["object_points"].shape[0]
        for k in self._loss_keys:
            self.train_meters[k].update(losses[k].item(), n=B)

        return {k: losses[k].item() for k in self._loss_keys}

    def validate(self, epoch: int = 0, step: int = 0) -> None:
        """Full validation pass (no backward)."""
        self.model.eval()
        for m in self.val_meters.values():
            m.reset()

        t0 = time.time()
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                batch = self._to_device(batch)
                losses = self._compute_losses(batch)
                B = batch["object_points"].shape[0]
                for k in self._loss_keys:
                    self.val_meters[k].update(losses[k].item(), n=B)

        elapsed = time.time() - t0

        results = {f"scalar/val/{k}": self.val_meters[k].avg for k in self._loss_keys}
        self.logger.log(results, step)

        msg = (
            f"[ Validation ] epoch={epoch} step={step}  "
            + "  ".join(f"{k}={self.val_meters[k].avg:.4f}" for k in self._loss_keys)
            + f"  elapsed={elapsed:.1f}s"
        )
        print(msg)
        logging.info(msg)

        val_loss = self.val_meters["total"].avg
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save("best")

    def evaluate(self, epoch: int = 0, step: int = 0) -> None:
        """Full evaluation: sample from model, compute fidelity and physics metrics."""
        self.model.eval()
        loader = self.test_loader if self.test_loader is not None else self.val_loader

        eval_cfg = self.cfg.evaluation
        phys_w = eval_cfg.physics_weights
        scorer = GraspScorer(
            beta1=float(phys_w.beta1),
            beta2=float(phys_w.beta2),
            beta3=float(phys_w.beta3),
            beta4=float(phys_w.beta4),
            mu=float(eval_cfg.mu),
            object_mass=self.object_mass,
        )

        cf_vals, ff_vals, fvr_vals = [], [], []
        top1_scores, top3_scores = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                batch = self._to_device(batch)
                obj_pts = batch["object_points"]  # (B, 3, N)

                preds = self.model.sample(obj_pts, self.num_eval_samples)

                cf_vals.append(self.compute_contact_fidelity(preds, batch))
                ff_vals.append(self.compute_force_fidelity(preds, batch))
                fvr_vals.append(self.compute_friction_violation_rate(preds, batch))

                # Physics scoring on sampled candidates
                B = obj_pts.shape[0]
                obj_pts_n3 = obj_pts.permute(0, 2, 1)  # (B, N, 3)
                for b in range(B):
                    b_preds = [p for i, p in enumerate(preds) if i % B == b]
                    if b_preds:
                        ranked = scorer.rank_candidates(b_preds, obj_pts_n3[b])
                        scores = [c["score"] for c in ranked]
                        top1_scores.append(scores[0])
                        top3_scores.append(float(np.mean(scores[:3])))

        contact_fid = float(np.mean(cf_vals)) if cf_vals else 0.0
        force_fid   = float(np.mean(ff_vals)) if ff_vals else 0.0
        fvr         = float(np.mean(fvr_vals)) if fvr_vals else 0.0
        top1        = float(np.mean(top1_scores)) if top1_scores else 0.0
        top3        = float(np.mean(top3_scores)) if top3_scores else 0.0

        results = {
            "scalar/eval/contact_fidelity":       contact_fid,
            "scalar/eval/force_fidelity":          force_fid,
            "scalar/eval/friction_violation_rate": fvr,
            "scalar/eval/top1_physics_score":      top1,
            "scalar/eval/top3_physics_score":      top3,
        }
        self.logger.log(results, step)

        msg = (
            f"[  Evaluate  ] epoch={epoch} step={step}  "
            f"contact_fidelity={contact_fid:.4f}  "
            f"force_fidelity={force_fid:.4f}  "
            f"friction_violation_rate={fvr:.4f}  "
            f"top1_score={top1:.4f}  "
            f"top3_score={top3:.4f}"
        )
        print(msg)
        logging.info(msg)

    def save(self, tag: str) -> None:
        """Save checkpoint: {'epoch', 'model', 'optimizer', 'best_val_loss'}."""
        path = os.path.join(self.logdir, f"checkpoint_{tag}.pt")
        torch.save(
            {
                "epoch":         tag,
                "model":         self.model.state_dict(),
                "optimizer":     self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        logging.info(f"Saved checkpoint: {path}")

    # ------------------------------------------------------------------
    # Evaluation metrics
    # ------------------------------------------------------------------

    def compute_contact_fidelity(self, preds: list, batch: dict) -> float:
        """Mean per-finger contact position error (metres)."""
        B = batch["object_points"].shape[0]
        gt_per_finger = _aggregate_per_finger(
            batch["contacts"], batch["finger_ids"], batch["valid_mask"]
        )  # (B, 5, 3) - NaN where no GT contact for that finger

        errors: list[float] = []
        for k, pred in enumerate(preds):
            b = k % B
            pred_c = pred["contacts"].to(gt_per_finger.device)  # (5, 3)
            gt_c   = gt_per_finger[b]                           # (5, 3)
            mask = torch.isfinite(gt_c).all(dim=-1)             # (5,)
            if mask.sum() > 0:
                err = (pred_c[mask] - gt_c[mask]).norm(dim=-1).mean().item()
                errors.append(err)

        return float(np.mean(errors)) if errors else 0.0

    def compute_force_fidelity(self, preds: list, batch: dict) -> float:
        """Mean per-finger force magnitude error (Newtons)."""
        B = batch["object_points"].shape[0]
        gt_per_finger = _aggregate_per_finger(
            batch["forces"], batch["finger_ids"], batch["valid_mask"]
        )  # (B, 5, 3)

        errors: list[float] = []
        for k, pred in enumerate(preds):
            b = k % B
            pred_f = pred["forces"].to(gt_per_finger.device)  # (5, 3)
            gt_f   = gt_per_finger[b]                         # (5, 3)
            mask = torch.isfinite(gt_f).all(dim=-1)
            if mask.sum() > 0:
                err = (pred_f[mask].norm(dim=-1) - gt_f[mask].norm(dim=-1)).abs().mean().item()
                errors.append(err)

        return float(np.mean(errors)) if errors else 0.0

    def compute_friction_violation_rate(self, preds: list, batch: dict) -> float:
        """Fraction of finger-contacts violating the Coulomb friction cone."""
        obj_pts = batch["object_points"]  # (B, 3, N)
        B = obj_pts.shape[0]
        violations: list[float] = []
        for k, pred in enumerate(preds):
            b = k % B
            contacts = pred["contacts"].unsqueeze(0)  # (1, 4, 3)
            forces   = pred["forces"].unsqueeze(0)    # (1, 4, 3)
            obj_centroid = obj_pts[b].mean(dim=-1).unsqueeze(0).unsqueeze(0)
            normals  = torch.nn.functional.normalize(
                obj_centroid - contacts + 1e-8, dim=-1
            )
            from equidexflow.loaders.schema import N_FINGERS
            valid    = torch.ones(1, N_FINGERS, dtype=torch.bool, device=contacts.device)
            rate     = friction_cone_violation_rate(forces, normals, valid, mu=self.mu)
            violations.append(rate.item())

        return float(np.mean(violations)) if violations else 0.0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_losses(self, batch: dict) -> dict:
        """Compute all weighted losses for one batch.

        Returns a dict of scalar tensors (all on the model's device).
        """
        # ## 4 model losses from forward() ###############################
        model_losses: dict = self.model(batch)

        # ## per-finger validity mask for prediction-space physics losses ##
        _, _, _, valid_f = self._per_finger_tensors(batch)

        # Guard: if no per-finger data is available (finger_ids all –1), physics
        # losses evaluate to zero rather than producing a spurious signal.
        any_valid = valid_f.any()
        if any_valid:
            phys = physics_loss(
                model_losses["pred_contacts"],
                model_losses["pred_normals"],
                model_losses["pred_forces"],
                batch["object_points"],  # (B, 3, N) - collision_penalty handles both layouts
                valid_f,
                object_point_normals=batch.get("object_point_normals"),     # (B, 3, N) outward normals
                pred_fingertips=model_losses.get("pred_fingertips"),         # (B, 4, 3) Drake-FK derived
                fingertip_radius=model_losses.get("fingertip_radius", 0.012),
                pred_collision_spheres=model_losses.get("pred_collision_spheres"),  # (B, S, 3)
                collision_sphere_radii=model_losses.get("collision_sphere_radii"), # (S,)
                pred_force_coords=model_losses["pred_force_coords"],
                object_mass=self.object_mass,
                mu=self.mu,
            )
        else:
            zero = torch.tensor(0.0, device=self.device)
            phys = {
                "wrench_balance": zero,
                "friction_cone":  zero,
                "collision":      zero,
                "self_collision": zero,
                "total":          zero,
            }

        lw = self.lw
        total = (
            lw["flow"]              * model_losses["flow"]
            + lw["hand_q"]           * model_losses["hand_q"]
            + lw["contact"]          * model_losses["contact"]
            + lw["normal"]           * model_losses["normal"]
            + lw["force"]            * model_losses["force"]
            + lw["reach"]            * model_losses["reach"]
            + lw["physics_wrench"]   * phys["wrench_balance"]
            + lw["physics_friction"] * phys["friction_cone"]
            + lw["physics_collision"]* phys["collision"]
        )

        return {
            "flow":              model_losses["flow"],
            "hand_q":            model_losses["hand_q"],
            "contact":           model_losses["contact"],
            "normal":            model_losses["normal"],
            "force":             model_losses["force"],
            "reach":             model_losses["reach"],
            "physics_wrench":    phys["wrench_balance"],
            "physics_friction":  phys["friction_cone"],
            "physics_collision": phys["collision"],
            "total":             total,
        }

    def _per_finger_tensors(self, batch: dict):
        """Aggregate batch contacts/normals/forces per finger.

        Returns
        -------
        contacts_f : (B, 5, 3)
        normals_f  : (B, 5, 3) unit-normalised
        forces_f   : (B, 5, 3)
        valid_f    : (B, 5) bool - True where finger has ≥1 valid GT contact
        """
        B = batch["contacts"].shape[0]
        device = batch["contacts"].device

        contacts_f = _aggregate_per_finger(
            batch["contacts"], batch["finger_ids"], batch["valid_mask"]
        )
        normals_f  = _aggregate_per_finger(
            batch["normals"],  batch["finger_ids"], batch["valid_mask"]
        )
        forces_f   = _aggregate_per_finger(
            batch["forces"],   batch["finger_ids"], batch["valid_mask"]
        )

        # Per-finger valid mask: True where at least one contact belongs to that finger
        from equidexflow.loaders.schema import N_FINGERS
        valid_f = torch.zeros(B, N_FINGERS, dtype=torch.bool, device=device)
        for f in range(N_FINGERS):
            valid_f[:, f] = (batch["valid_mask"] & (batch["finger_ids"] == f)).any(dim=1)

        # Replace NaN with 0 (entries masked out by valid_f)
        contacts_f = torch.nan_to_num(contacts_f, nan=0.0)
        forces_f   = torch.nan_to_num(forces_f,   nan=0.0)
        normals_f  = torch.nn.functional.normalize(
            torch.nan_to_num(normals_f, nan=0.0), dim=-1, eps=1e-8
        )

        return contacts_f, normals_f, forces_f, valid_f

    def _to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _log_and_print_meters(self, split: str, epoch: int, step: int) -> None:
        meters = self.train_meters if split == "train" else self.val_meters
        results = {f"scalar/{split}/{k}": meters[k].avg for k in self._loss_keys}
        self.logger.log(results, step)

        msg = (
            f"[  Training  ] epoch={epoch} step={step}  "
            + "  ".join(f"{k}={meters[k].avg:.4f}" for k in self._loss_keys)
        )
        print(msg)
        logging.info(msg)

        for m in meters.values():
            m.reset()


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _aggregate_per_finger(
    data: torch.Tensor,        # (B, M, D)
    finger_ids: torch.Tensor,  # (B, M)  values 0..N_FINGERS-1, –1 = padding
    valid_mask: torch.Tensor,  # (B, M)  bool
    n_fingers: int | None = None,
) -> torch.Tensor:
    if n_fingers is None:
        from equidexflow.loaders.schema import N_FINGERS as _NF
        n_fingers = _NF
    """Average data per finger.

    Returns (B, n_fingers, D) with NaN for fingers that have no valid GT contact.
    """
    B, M, D = data.shape
    result = torch.full(
        (B, n_fingers, D), float("nan"), dtype=data.dtype, device=data.device
    )
    for f in range(n_fingers):
        mask = valid_mask & (finger_ids == f)  # (B, M)
        count = mask.float().sum(dim=1)        # (B,)
        has = count > 0                        # (B,)
        if has.any():
            weighted = (data * mask.unsqueeze(-1).float()).sum(dim=1)  # (B, D)
            mean = weighted / count.unsqueeze(-1).clamp(min=1.0)
            result[has, f] = mean[has]
    return result
