"""``equidexflow-demo`` - one-shot grasp synthesis from a mesh.

Loads an object mesh, samples a point cloud, runs the model, and writes
per-grasp ``.npz`` files plus a single PNG preview. Pure-inference: needs
``[demo]`` extras (``trimesh``, ``matplotlib``); ``--viz`` additionally
needs ``open3d``.

Example::

    equidexflow-demo \\
        --mesh assets/objects/graspit/sphere.stl \\
        --checkpoint allegro_full \\
        --num-samples 8 --out out/sphere
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from equidexflow.api import load_checkpoint
from equidexflow.pipeline import generate_seated_grasps


def _load_mesh_points(mesh_path: Path, n_points: int, rng: np.random.Generator):
    try:
        import trimesh
    except ImportError as e:
        raise SystemExit(
            "`trimesh` not installed. Install the demo extra: "
            "`pip install -e \".[demo]\"`."
        ) from e
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if mesh.is_empty:
        raise SystemExit(f"empty mesh at {mesh_path}")
    pts, _ = trimesh.sample.sample_surface(mesh, n_points, seed=int(rng.integers(2**31)))
    return mesh, np.asarray(pts, dtype=np.float32)


def _save_grasp_npz(out_dir: Path, idx: int, g: dict, spheres_xyz, sphere_r):
    np.savez(
        out_dir / f"grasp_{idx:02d}.npz",
        wrist_pose=g["wrist_pose"].cpu().numpy(),
        hand_q=g["hand_q"].cpu().numpy(),
        contacts=g["contacts"].cpu().numpy(),
        forces=g["forces"].cpu().numpy(),
        contact_logits=g["contact_logits"].cpu().numpy(),
        hand_sphere_xyz=spheres_xyz,
        hand_sphere_radii=sphere_r,
    )


def _render_png(
    out_path: Path,
    obj_pts: np.ndarray,
    contacts: np.ndarray,
    forces: np.ndarray,
    spheres_xyz: np.ndarray,
    sphere_r: np.ndarray,
) -> None:
    """Two 2D orthographic projections (XY, XZ) - headless-safe; no 3D backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    all_pts = np.concatenate([obj_pts, spheres_xyz, contacts], axis=0)
    ctr = all_pts.mean(0)
    rng = float(np.max(np.ptp(all_pts, axis=0)) * 0.6) or 0.1

    for ax, (i, j), title in zip(axes, [(0, 1), (0, 2)], ["XY (top)", "XZ (side)"]):
        ax.scatter(obj_pts[:, i], obj_pts[:, j], s=4, c="#6B8FB8", alpha=0.5, label="object")
        sz = 60 * (sphere_r / sphere_r.max()) ** 2
        ax.scatter(spheres_xyz[:, i], spheres_xyz[:, j], s=sz, c="#2C2C2C", alpha=0.8, label="hand")
        ax.scatter(contacts[:, i], contacts[:, j], s=80, c="#E07A5F", marker="x", label="contacts")
        for c, f in zip(contacts, forces):
            ax.arrow(c[i], c[j], 0.03 * f[i], 0.03 * f[j],
                     head_width=0.004, color="#E07A5F", length_includes_head=True)
        ax.set_xlim(ctr[i] - rng, ctr[i] + rng)
        ax.set_ylim(ctr[j] - rng, ctr[j] + rng)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    axes[1].legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="equidexflow-demo", description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path, help="object mesh file (.stl/.obj/.ply)")
    parser.add_argument("--checkpoint", default="allegro_full",
                        help="checkpoint key under checkpoints/ or path to checkpoint_best.pt")
    parser.add_argument("--num-samples", type=int, default=32,
                        help="candidate pool size (cheap); the best few are seated. "
                             "Grasp quality improves with a larger pool.")
    parser.add_argument("--seat-top-k", type=int, default=4,
                        help="how many top-ranked candidates to seat (the expensive step)")
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("out") / "demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seat-steps", type=int, default=250,
                        help="task-space optimization steps for seating the hand")
    parser.add_argument("--viz", action="store_true",
                        help="open an interactive Open3D viewer with the hand visual mesh (top grasp)")
    parser.add_argument("--render-mesh", action="store_true",
                        help="also write preview_mesh.png: an offscreen visual-mesh render of the top grasp")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"[demo] mesh        : {args.mesh}")
    print(f"[demo] checkpoint  : {args.checkpoint}")
    print(f"[demo] device      : {args.device}")
    print(f"[demo] num_samples : {args.num_samples}")

    mesh, _ = _load_mesh_points(args.mesh, args.num_points, rng)
    print(f"[demo] mesh extents (m): {mesh.extents}")

    model = load_checkpoint(args.checkpoint, device=args.device)
    hand = getattr(model, "hand", None) or "allegro"
    if str(hand).lower() != "allegro":
        print(f"[demo] WARN: checkpoint hand='{hand}' - only allegro FK is wired for the PNG render.",
              file=sys.stderr)

    # Shared pipeline: sample pool -> force-closure rank -> seat best few -> rerank.
    print(f"[demo] selecting: best {args.seat_top_k} of {args.num_samples} by GraspScorer, "
          f"then penetration-aware seating ({args.seat_steps} steps)")
    res = generate_seated_grasps(
        model, mesh, num_samples=args.num_samples, seat_top_k=args.seat_top_k,
        num_points=args.num_points, seat_steps=args.seat_steps,
        seed=args.seed, device=args.device,
    )
    grasps, order, fk = res["grasps"], res["order"], res["fk"]
    pts = res["pc"].T.cpu().numpy()   # (N, 3) for the 2D preview
    top = grasps[order[0]]

    args.out.mkdir(parents=True, exist_ok=True)

    # Per-grasp .npz with FK spheres
    for rank, idx in enumerate(order):
        g = grasps[idx]
        with torch.no_grad():
            spheres, radii = fk.forward_all_spheres(g["hand_q"].to(args.device),
                                                    g["wrist_pose"].to(args.device))
        s_np = spheres.squeeze(0).cpu().numpy()
        r_np = radii.cpu().numpy()
        _save_grasp_npz(args.out, rank, g, s_np, r_np)

    # PNG preview of the top grasp
    with torch.no_grad():
        s, r = fk.forward_all_spheres(top["hand_q"].to(args.device),
                                      top["wrist_pose"].to(args.device))
    s_np, r_np = s.squeeze(0).cpu().numpy(), r.cpu().numpy()
    png = args.out / "preview.png"
    _render_png(png, pts,
                top["contacts"].cpu().numpy(),
                top["forces"].cpu().numpy(),
                s_np, r_np)
    print(f"[demo] wrote {len(grasps)} grasps + {png}")

    # Optional offscreen visual-mesh render (real Allegro link meshes).
    if args.render_mesh:
        from equidexflow.render import render_hand_offscreen
        mesh_png = args.out / "preview_mesh.png"
        render_hand_offscreen(
            mesh_png,
            top["hand_q"].cpu().numpy(),
            top["wrist_pose"].cpu().numpy(),
            obj_mesh=mesh,
            contacts=top["contacts"].cpu().numpy(),
            forces=top["forces"].cpu().numpy(),
        )
        print(f"[demo] wrote {mesh_png}")

    if args.viz:
        from equidexflow.render import view_hand
        view_hand(
            top["hand_q"].cpu().numpy(),
            top["wrist_pose"].cpu().numpy(),
            obj_mesh=mesh,
            contacts=top["contacts"].cpu().numpy(),
            forces=top["forces"].cpu().numpy(),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
