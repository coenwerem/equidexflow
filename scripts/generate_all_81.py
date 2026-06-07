#!/usr/bin/env python3
"""Generate grasp candidates for all 81 dataset objects.

Stage 1 of the full pipeline: EquiDexFlow inference (GPU).
Stage 2 (Drake FK re-ranking + rendering) runs separately in frogger venv.

Usage (run from EquiDexFlow dir):
    cd ~/ResearchProjects/MuJoCoDex/third_party/grasp_syn/EquiDexFlow
    ~/ResearchProjects/MuJoCoDex/.venv/bin/python \
        ~/ResearchProjects/frogger/scripts/equidex/generate_all_81.py \
        --checkpoint outputs/training_results/equidexflow_dex_full_flow_film/film_20260523-0908/checkpoint_best.pt \
        --device 0 --K 200 \
        --out ~/ResearchProjects/frogger/outputs/figure_grasps_all81
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_EDF_DIR = os.environ.get("EQUIDEXFLOW_OUTPUTS_DIR", os.getcwd())

import argparse
from omegaconf import OmegaConf

DB_DIR = Path.home() / "ResearchProjects/frogger/outputs/datasets/dexgraspdb/v3/allegro"
MESH_DIR = Path.home() / "ResearchProjects/MuJoCoDex/assets/misc/objects"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K", type=int, default=200)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    torch.manual_seed(args.seed)

    all_objects = sorted([p.stem for p in DB_DIR.glob("*.json")])
    print(f"Found {len(all_objects)} objects in dataset")

    from equidexflow.loaders import get_dataloader
    test_cfg = OmegaConf.create({
        "dataset": {
            "name": "dexgrasp",
            "grasp_db_dir": str(DB_DIR),
            "object_mesh_dir": str(MESH_DIR),
            "n_object_points": 512, "max_contacts": 64,
            "mu": 0.5, "object_mass": 0.2,
            "augment": False, "split": "test", "object_names": None,
        },
        "batch_size": 1, "num_workers": 0, "shuffle": False,
    })
    loader = get_dataloader("test", test_cfg)

    pts_by_obj = {}
    for batch in loader:
        names = batch["object_name"]
        for b in range(batch["object_points"].shape[0]):
            name = names[b]
            if name not in pts_by_obj:
                pts_by_obj[name] = batch["object_points"][b]
        if len(pts_by_obj) >= len(all_objects):
            break
    print(f"Loaded point clouds for {len(pts_by_obj)} objects")

    from equidexflow.models import get_dex_model
    model = get_dex_model(
        p_uncond=0.1, guidance=2.0, num_ode_steps=25,
        hand_q_decoder_type="flow",
        n_coupling_layers=8,
        surface_proj_tau=0.005,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()

    from equidexflow.kinematics.allegro_fk import AllegroRightHandFK
    fk = AllegroRightHandFK().to(device)
    from equidexflow.physics.scorer import GraspScorer
    scorer = GraspScorer(mu=0.5, object_mass=0.2,
                         beta1=1.0, beta2=2.0, beta3=1.0, beta4=0.5,
                         fk_module=fk, fk_collision_weight=5.0)

    from scipy.spatial.transform import Rotation
    import trimesh

    def load_mesh_pts_normals(obj_name, n_pts=4096):
        import re
        _EXPLICIT = {
            "cube": "graspit/cube", "sphere": "graspit/sphere",
            "graspit_box": "graspit/box", "graspit_cylinder": "graspit/cylinder",
            "sns_cup": "graspit/sns_cup",
        }
        mesh = None
        stem = _EXPLICIT.get(obj_name)
        if stem:
            for ext in (".stl", ".obj", ".ply"):
                path = MESH_DIR / (stem + ext)
                if path.exists():
                    mesh = trimesh.load(str(path), force="mesh")
                    break
        if mesh is None:
            ycb_dir = MESH_DIR / "frogger_ycb"
            for p in ycb_dir.glob(f"*_{obj_name}.obj"):
                mesh = trimesh.load(str(p), force="mesh")
                break
        if mesh is None and re.match(r"^[A-Z]\d+$", obj_name):
            for setname in ("egad_eval_set", "egad_train_set"):
                path = Path.home() / f"ResearchProjects/Datasets/{setname}/{obj_name}.obj"
                if path.exists():
                    mesh = trimesh.load(str(path), force="mesh")
                    mesh.apply_scale(0.001)
                    break
        if mesh is None:
            return None, None
        mesh_centered = mesh.copy()
        mesh_centered.apply_translation(-mesh_centered.centroid)
        samples, face_idx = trimesh.sample.sample_surface(mesh_centered, n_pts)
        pts = torch.from_numpy(samples.astype(np.float32))
        nrm = torch.from_numpy(mesh_centered.face_normals[face_idx].astype(np.float32))
        nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return pts, nrm

    args.out.mkdir(parents=True, exist_ok=True)
    total_t0 = time.time()
    done = 0
    skipped = 0

    with torch.no_grad():
        for obj_name in all_objects:
            if obj_name not in pts_by_obj:
                print(f"  {obj_name}: not in dataloader, skipping")
                skipped += 1
                continue

            out_dir = args.out / obj_name
            if args.skip_existing and out_dir.exists():
                existing = list(out_dir.glob(f"{obj_name}__grasp__*.json"))
                if len(existing) >= args.K:
                    print(f"  {obj_name}: {len(existing)} existing, skipping")
                    skipped += 1
                    continue

            obj_pts = pts_by_obj[obj_name].unsqueeze(0).to(device)

            t0 = time.time()
            preds = []
            for chunk_start in range(0, args.K, args.batch):
                chunk_n = min(args.batch, args.K - chunk_start)
                torch.manual_seed(args.seed + chunk_start)
                chunk_preds = model.sample(obj_pts, chunk_n)
                preds.extend(chunk_preds)
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            mp, mn = load_mesh_pts_normals(obj_name)
            if mp is not None:
                obj_nrm = mn.to(device)
                fk_pts = mp.to(device)
                ranked = scorer.rank_candidates(
                    preds, obj_pts.permute(0, 2, 1)[0], obj_nrm, fk_pts)
            else:
                ranked = preds

            gen_time = time.time() - t0

            out_dir.mkdir(parents=True, exist_ok=True)
            for rank, pred in enumerate(ranked):
                wrist_T = pred["wrist_pose"].cpu().numpy()
                hand_q = pred["hand_q"].cpu().numpy()
                R = wrist_T[:3, :3]
                p = wrist_T[:3, 3]
                quat_xyzw = Rotation.from_matrix(R).as_quat()
                q_star = np.concatenate([
                    [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
                    p, hand_q,
                ])
                score = pred.get("score", 0.0)
                if isinstance(score, torch.Tensor):
                    score = float(score.cpu())
                data = {
                    "object_name": obj_name,
                    "variant": "pool_K200",
                    "rank": rank,
                    "score": score,
                    "wrist_pose": wrist_T.tolist(),
                    "hand_q": hand_q.tolist(),
                    "q_star": q_star.tolist(),
                    "contacts": pred["contacts"].cpu().numpy().tolist(),
                    "forces": pred["forces"].cpu().numpy().tolist(),
                }
                path = out_dir / f"{obj_name}__grasp__{rank:02d}.json"
                path.write_text(json.dumps(data, indent=2))

            done += 1
            print(f"  [{done:3d}/{len(all_objects)}] {obj_name}: {len(ranked)} candidates in {gen_time:.1f}s")

    total_time = time.time() - total_t0
    print(f"\nDone. {done} objects generated, {skipped} skipped. Total: {total_time:.0f}s")


if __name__ == "__main__":
    main()
