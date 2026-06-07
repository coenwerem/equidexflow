#!/usr/bin/env python3
"""Run full EquiDexFlow evaluation: all 4 variants on the 81-object test set.

USAGE (run from the EquiDexFlow directory):
    cd ~/ResearchProjects/MuJoCoDex/third_party/grasp_syn/EquiDexFlow
    ~/ResearchProjects/MuJoCoDex/.venv/bin/python \
        ~/ResearchProjects/frogger/scripts/equidex/run_full_eval.py \
        --device 0

PREREQUISITES:
    - MuJoCoDex venv with torch, omegaconf, tensorboard, tqdm
    - GPU with >=8GB VRAM (eval uses ~4GB for batch inference)
    - Dataset: ~/ResearchProjects/frogger/outputs/datasets/dexgraspdb/v3/allegro
    - Object meshes: ~/ResearchProjects/MuJoCoDex/assets/misc/objects
    - Checkpoints in: outputs/training_results/<variant>/<run>/checkpoint_best.pt

DATASET:
    81 objects | 8,099 grasps | Test split: 811 grasps
    All variants evaluated on identical 811-grasp test set (object_names=null).

METRIC (corrected):
    "Contact error of the top-ranked candidate against the nearest
     ground-truth grasp for each test object."
    - Prediction side: GraspScorer ranks 10 candidates, take top-1
    - GT side: min contact distance across all GT grasps for that object
    This handles multi-modal GT correctly (10 GT grasps × 81 objects).

VRAM: ~4GB peak (inference only, batch_size=1 per object, 10 ODE steps)
TIME: ~8 min per variant × 4 variants = ~32 min total on RTX 5070 Ti

OUTPUT:
    ~/ResearchProjects/frogger/outputs/paper_results/equidex/equidex_results/
        results_table_81obj.yaml     (aggregated table)
        <variant>/contacts.csv       (per-finger contact errors)
        <variant>/forces.csv         (per-finger force errors + FVR + WB)
        <variant>/rollout.csv        (top-1/top-3 physics scores)
"""
from __future__ import annotations

import os
import sys

# Ensure EquiDexFlow imports work regardless of CWD
_EDF_DIR = os.environ.get("EQUIDEXFLOW_OUTPUTS_DIR", os.getcwd())

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EQUIDEXFLOW_DIR = Path(_EDF_DIR)

# Standalone repo layout. Checkpoints are frozen locally under checkpoints/<key>/
# (see checkpoints/MANIFEST.yaml); resolution prefers these over any external tree.
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CKPT_DIR = REPO_ROOT / "checkpoints"

# Eval artifacts (CSVs, aggregated yaml) land in the repo-local, gitignored outputs/.
FROGGER_OUT = REPO_ROOT / "outputs" / "paper_results" / "equidex" / "equidex_results"

VARIANTS_BY_HAND = {
    # Local manifest keys (checkpoints/<key>/). These four are the frozen `_xhand`
    # generation that produced the preprint tab:results (MANIFEST.yaml,
    # reproduces_preprint_table: true; allegro_full sha256 == authoritative
    # equidexflow_dex_full_xhand/20260526-xhand checkpoint). NOT the superseded
    # `_retrain` generation.
    "allegro": [
        "allegro_full",
        "allegro_pose_only",
        "allegro_contact_only",
        "allegro_geom_only",
    ],
    # LEAP checkpoints are not yet frozen into this repo (see
    # docs/MACHINE_B_HANDOFF.md); once collected as checkpoints/leap_*/ they
    # resolve the same way.
    "leap": [
        "leap_full",
        "leap_pose_only",
        "leap_contact_only",
        "leap_geom_only",
    ],
}
VARIANTS = VARIANTS_BY_HAND["allegro"]  # default; main() overrides per --hand


def _grasp_db_dir(dataset_subdir: str) -> str:
    """Resolve the dexgraspdb dir: EQUIDEXFLOW_DATA_DIR env, then the repo-local
    copy (data/dexgraspdb/v3/<hand>), then the FRoGGeR source as a last resort."""
    candidates = []
    env = os.environ.get("EQUIDEXFLOW_DATA_DIR")
    if env:
        candidates.append(Path(env) / "dexgraspdb" / "v3" / dataset_subdir)
        candidates.append(Path(env) / dataset_subdir)  # env points straight at v3/
    candidates.append(REPO_ROOT / "data" / "dexgraspdb" / "v3" / dataset_subdir)
    candidates.append(Path.home() / f"ResearchProjects/frogger/outputs/datasets/dexgraspdb/v3/{dataset_subdir}")
    for c in candidates:
        if c.is_dir():
            return str(c)
    return str(candidates[-1])  # report the canonical fallback in the error path


def _object_mesh_dir() -> str:
    env = os.environ.get("EQUIDEXFLOW_OBJECTS_DIR")
    if env:
        return env
    local = REPO_ROOT / "assets" / "objects"
    if local.is_dir():
        return str(local)
    return str(Path.home() / "ResearchProjects/MuJoCoDex/assets/misc/objects")


def _test_config(hand: str) -> dict:
    """Build canonical test config for one hand. ALL 81 objects, no restriction."""
    dataset_subdir = {"allegro": "allegro", "leap": "leap"}[hand]
    cfg = {
        "dataset": {
            "name": "dexgrasp",
            "grasp_db_dir": _grasp_db_dir(dataset_subdir),
            "object_mesh_dir": _object_mesh_dir(),
            "n_object_points": 512,
            "max_contacts": 64,
            "mu": 0.5,
            "object_mass": 0.2,
            "augment": False,
            "split": "test",
            "object_names": None,  # ALL 81 objects
        },
        "batch_size": 2,
        "num_workers": 4,
        "shuffle": False,
    }
    if hand == "leap":
        # Loader override: rebuild graspit_box (axis-swap) and sphere (LEAP
        # 30mm vs disk 35mm) via trimesh.creation with FRoGGeR's specs.
        cfg["dataset"]["use_frogger_primitive_specs"] = True
    return cfg


TEST_CONFIG = _test_config("allegro")  # legacy alias; main() rebuilds per --hand


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_run(variant: str) -> Path | None:
    # Prefer the repo-local frozen checkpoint (checkpoints/<key>/checkpoint_best.pt
    # with its config.yml), which is the authoritative, sha-pinned artifact for the
    # standalone repo. This is a manifest key like "allegro_full".
    local = LOCAL_CKPT_DIR / variant
    if (local / "checkpoint_best.pt").is_file():
        return local
    # Fallback: an external EquiDexFlow training tree (legacy / dev), variant named
    # like "equidexflow_dex_full_xhand"; pick the newest run by checkpoint mtime.
    base = EQUIDEXFLOW_DIR / "outputs" / "training_results" / variant
    if not base.is_dir():
        return None
    runs = [p for p in base.iterdir() if p.is_dir() and (p / "checkpoint_best.pt").is_file()]
    if not runs:
        return None
    return max(runs, key=lambda p: (p / "checkpoint_best.pt").stat().st_mtime)


def _safe_float(v) -> float | None:
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Evaluation functions (inlined — no subprocess, same process, same GPU)
# ---------------------------------------------------------------------------

def evaluate_variant(variant: str, device: torch.device, num_samples: int,
                     test_config: dict | None = None):
    """Run all three evals for one variant. Returns dict of results."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {variant}")
    print(f"{'='*60}")

    run_dir = _latest_run(variant)
    if run_dir is None:
        print(f"  ERROR: No run directory for {variant}")
        return None

    cp = run_dir / "checkpoint_best.pt"
    if not cp.is_file():
        print(f"  ERROR: No checkpoint at {cp}")
        return None

    print(f"  Checkpoint: {cp}")

    # Load model
    from equidexflow.models import get_dex_model
    cfg_files = sorted(run_dir.glob("*.yml"))
    cfg = OmegaConf.load(str(cfg_files[0])) if cfg_files else OmegaConf.create({})

    # Build the model with the SAME architecture the checkpoint was trained with
    # (hand, hand_q decoder type, wrist frame, cond_norm). Reading these from the
    # run's config is essential: the LEAP `full` model uses a flow hand_q decoder
    # + grasp_center frame; building with defaults (deterministic/base) silently
    # random-inits the flow decoder under strict=False -> garbage hand_q.
    model = get_dex_model(
        p_uncond=float(cfg.model.get("p_uncond", 0.1)),
        guidance=float(cfg.model.get("guidance", 2.0)),
        num_ode_steps=int(cfg.model.get("num_ode_steps", 10)),
        hand_q_decoder_type=str(cfg.model.get("hand_q_decoder", "deterministic")),
        n_coupling_layers=int(cfg.model.get("n_coupling_layers", 8)),
        surface_proj_tau=float(cfg.model.get("surface_proj_tau", 0.005)),
        hand=str(cfg.model.get("hand", "allegro")),
        wrist_frame=str(cfg.model.get("wrist_frame", "base")),
        cond_norm=bool(cfg.model.get("cond_norm", False)),
    ).to(device)

    ckpt = torch.load(str(cp), map_location="cpu")
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint/arch mismatch: {len(missing)} missing, "
                           f"{len(unexpected)} unexpected (wrong decoder/hand?). "
                           f"missing[:3]={missing[:3]} unexpected[:3]={unexpected[:3]}")
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # Build CANONICAL test loader (81 objects, 811 grasps)
    from equidexflow.loaders import get_dataloader
    test_cfg = OmegaConf.create(test_config if test_config is not None else TEST_CONFIG)
    loader = get_dataloader("test", test_cfg)
    print(f"  Test set: {len(loader.dataset)} grasps")

    # Build scorer
    from equidexflow.physics.scorer import GraspScorer
    scorer = GraspScorer(mu=0.5, object_mass=0.2,
                         beta1=1.0, beta2=2.0, beta3=1.0, beta4=0.5)

    # --- Contact eval ---
    print("  Running contact eval...")
    from eval_contacts import evaluate_contacts, save_csv as save_contacts_csv
    mean_errs, std_errs, cov_rate = evaluate_contacts(
        model, loader, device, num_samples, scorer=scorer)

    out_dir = FROGGER_OUT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    save_contacts_csv(str(out_dir / "contacts.csv"), mean_errs, std_errs, cov_rate)

    # --- Force eval ---
    print("  Running force eval...")
    from eval_forces import evaluate_forces, save_csv as save_forces_csv
    mean_mag, mean_dir, mean_fvr, mean_wb = evaluate_forces(
        model, loader, device, num_samples, scorer=scorer, mu=0.5, object_mass=0.2)

    save_forces_csv(str(out_dir / "forces.csv"), mean_mag, mean_dir, mean_fvr, mean_wb)

    # --- Rollout eval ---
    print("  Running rollout eval...")
    from eval_rollout import evaluate_rollout, save_csv as save_rollout_csv
    top1, top3, fvr = evaluate_rollout(
        model, loader, device, num_samples, scorer, mu=0.5)

    save_rollout_csv(str(out_dir / "rollout.csv"), top1, top3, fvr)

    result = {
        "variant": variant,
        "checkpoint": str(cp),
        "contact_err_m": float(np.nanmean(mean_errs)),
        "contact_coverage": float(np.nanmean(cov_rate)),
        "force_err_N": float(np.nanmean(mean_mag)),
        "dir_err_deg": float(np.nanmean(mean_dir)),
        "friction_viol_pct": float(mean_fvr * 100),
        "wrench_residual_Nm": float(mean_wb),
        "top1_score": float(np.mean(top1)),
        "top3_score": float(np.mean(top3)),
    }

    print(f"\n  Results for {variant}:")
    for k, v in result.items():
        if k not in ("variant", "checkpoint"):
            print(f"    {k}: {v:.4f}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="0",
                        help="GPU device ID or 'cpu'")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Candidates per object (default 10)")
    parser.add_argument("--hand", choices=["allegro", "leap"], default="allegro",
                        help="Which hand's variants + dataset to evaluate")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Override variant list (defaults to hand-specific set)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output YAML path (default: results_table_81obj[_<hand>].yaml)")
    parser.add_argument("--objects", nargs="+", default=None,
                        help="Restrict eval to these objects (e.g. good-tier). "
                             "Default: all in the dataset.")
    parser.add_argument("--drop", nargs="+", default=None,
                        help="Exclude these objects (e.g. phydex).")
    args = parser.parse_args()

    if args.variants is None:
        args.variants = VARIANTS_BY_HAND[args.hand]
    if args.out is None:
        suffix = "" if args.hand == "allegro" else f"_{args.hand}"
        args.out = FROGGER_OUT / f"results_table_81obj{suffix}.yaml"
    test_config = _test_config(args.hand)
    if args.objects:
        names = list(args.objects)
        if args.drop:
            names = [o for o in names if o not in set(args.drop)]
        test_config["dataset"]["object_names"] = names
        print(f"Restricting eval to {len(names)} objects")
    elif args.drop:
        # All objects minus the dropped ones (need the full list to subtract).
        import glob as _glob
        allobj = sorted(Path(p).stem for p in _glob.glob(
            test_config["dataset"]["grasp_db_dir"] + "/*.json"))
        names = [o for o in allobj if o not in set(args.drop)]
        test_config["dataset"]["object_names"] = names
        print(f"Dropping {args.drop} -> {len(names)} objects")

    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    results = []
    for v in args.variants:
        r = evaluate_variant(v, device, args.num_samples, test_config=test_config)
        if r is not None:
            results.append(r)

    payload = {
        "split": "test",
        "num_samples": args.num_samples,
        "metric": "top1_pred_vs_nearest_gt_per_object",
        "test_set": "81 objects, 811 grasps",
        "variants": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"\n{'='*60}")
    print(f"Results written to: {args.out}")
    print(f"{'='*60}")
    print(yaml.safe_dump(payload, sort_keys=False))


if __name__ == "__main__":
    sys.exit(main() or 0)
