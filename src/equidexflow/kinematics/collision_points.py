"""Full-hand collision points for penetration-aware seating.

The sparse collision spheres from ``forward_all_spheres`` only cover the finger
phalanges (links >= 30 mm) and the 4 fingertips -- not the palm or the short
proximal links. For a faithful penetration term during seating we instead sample
points off **every** hand visual mesh (palm included) and rigidly attach them to
their link frames, so they move with the differentiable FK
(``forward_link_frames``) and let the optimizer push the *whole* hand out of the
object.

``build_allegro_collision_fn`` returns a callable ``(hand_q, wrist) -> (B, K, 3)``
of world-frame points, differentiable in both arguments.
"""

from __future__ import annotations

import numpy as np
import torch


def build_allegro_collision_fn(
    fk,
    n_per_link: int = 48,
    seed: int = 0,
    device="cpu",
    dtype: torch.dtype = torch.float32,
    sdf_path=None,
):
    """Build a differentiable full-hand collision-point function for Allegro.

    Samples ``n_per_link`` surface points off each visual mesh (in its link
    frame, via the SDF visual pose), grouped by link. The returned function
    transforms them to the world frame with ``fk.forward_link_frames`` at the
    given ``hand_q`` / ``wrist`` -- a rigid (hence differentiable) map.
    """
    import trimesh

    from equidexflow.render.allegro_assets import load_allegro_visuals

    # Valid FK link names (so we skip e.g. the *_FROGGERSAMPLE frame).
    valid = set(
        fk.forward_link_frames(
            torch.zeros(fk.HAND_DOF), torch.eye(4)
        ).keys()
    )

    rng = np.random.default_rng(seed)
    per_link: dict[str, list[torch.Tensor]] = {}
    for vm in load_allegro_visuals(sdf_path):
        if vm.link_name not in valid or not vm.mesh_path.exists():
            continue
        mesh = trimesh.load(str(vm.mesh_path), force="mesh")
        if mesh.is_empty:
            continue
        pts, _ = trimesh.sample.sample_surface(
            mesh, n_per_link, seed=int(rng.integers(2**31))
        )
        if not np.allclose(vm.scale, 1.0):
            pts = np.asarray(pts) * vm.scale
        homog = np.c_[np.asarray(pts), np.ones(len(pts))] @ np.asarray(vm.X_BG).T
        per_link.setdefault(vm.link_name, []).append(
            torch.as_tensor(homog, dtype=dtype, device=device)
        )

    packed = {k: torch.cat(v, dim=0) for k, v in per_link.items()}  # name -> (k, 4)
    names = list(packed.keys())

    def collision_points(hand_q: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
        frames = fk.forward_link_frames(hand_q, wrist)  # name -> (B, 4, 4)
        outs = []
        for nm in names:
            X = frames[nm]                               # (B, 4, 4)
            P = packed[nm].to(X.dtype)                   # (k, 4)
            Pw = torch.einsum("bij,kj->bki", X, P)[..., :3]  # (B, k, 3)
            outs.append(Pw)
        return torch.cat(outs, dim=1)                    # (B, K, 3)

    return collision_points
