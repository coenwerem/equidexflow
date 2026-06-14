"""Visual-mesh rendering of a posed Allegro hand grasping an object.

Pure-torch FK (:class:`AllegroRightHandFK`) supplies per-link world transforms;
the SDF (:mod:`equidexflow.render.allegro_assets`) supplies each link's visual
meshes and their body-frame poses. Open3D draws the real hand meshes, the object
mesh, and 3D contact markers/force arrows -- the FRoGGeR/DexGraspNet-style
visualization, Drake-free.

Open3D is an optional (``[demo]`` / ``[viz]``) dependency and is imported lazily
so ``import equidexflow.render`` stays cheap for pure-inference users.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from equidexflow.kinematics.allegro_fk import AllegroRightHandFK
from equidexflow.render.allegro_assets import VisualMesh, load_allegro_visuals

# Colors: dark aluminum body, white silicone fingertips, blue object, coral contacts.
_BODY_COLOR = (0.12, 0.12, 0.13)
_TIP_COLOR = (0.85, 0.85, 0.87)
_OBJECT_COLOR = (0.42, 0.56, 0.72)
_CONTACT_COLOR = (0.88, 0.48, 0.37)


def _require_open3d():
    try:
        import open3d as o3d  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "`open3d` not installed. Install the demo extra: "
            'pip install -e ".[demo]" (or ".[viz]").'
        ) from e
    return o3d


def _as_tensor(x, device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device).float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def _visual_color(vm: VisualMesh) -> tuple[float, float, float]:
    name = f"{vm.visual_name} {vm.mesh_path.name}".lower()
    return _TIP_COLOR if "tip" in name else _BODY_COLOR


def build_hand_meshes(
    hand_q,
    X_WP,
    sdf_path: Path | str | None = None,
    fk: AllegroRightHandFK | None = None,
):
    """Return a list of world-posed, colored Open3D meshes for the hand.

    Parameters
    ----------
    hand_q : (16,) array-like of joint angles.
    X_WP   : (4, 4) wrist pose in the world frame (base/palm frame).
    """
    o3d = _require_open3d()
    fk = fk or AllegroRightHandFK()
    device = "cpu"
    fk = fk.to(device).eval()

    hand_q_t = _as_tensor(hand_q, device).reshape(-1)
    X_WP_t = _as_tensor(X_WP, device).reshape(4, 4)

    with torch.no_grad():
        frames = fk.forward_link_frames(hand_q_t, X_WP_t)  # name -> (1,4,4)

    visuals = load_allegro_visuals(sdf_path)
    meshes = []
    for vm in visuals:
        X_WB = frames.get(vm.link_name)
        if X_WB is None:
            continue  # body not in the FK chain (e.g. *_FROGGERSAMPLE frame)
        X_WB = X_WB.squeeze(0).cpu().numpy()
        if not vm.mesh_path.exists():
            continue
        mesh = o3d.io.read_triangle_mesh(str(vm.mesh_path))
        if mesh.is_empty():
            continue
        if not np.allclose(vm.scale, 1.0):
            mesh.scale(float(vm.scale[0]), center=(0.0, 0.0, 0.0))
        mesh.transform(X_WB @ vm.X_BG)
        mesh.paint_uniform_color(_visual_color(vm))
        mesh.compute_vertex_normals()
        meshes.append(mesh)
    return meshes


def trimesh_to_o3d(mesh, color=_OBJECT_COLOR):
    """Convert a trimesh mesh to a colored Open3D TriangleMesh."""
    o3d = _require_open3d()
    o = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
    )
    o.compute_vertex_normals()
    o.paint_uniform_color(color)
    return o


def _arrow(origin, direction, length, color, radius_scale=1.0):
    """A small Open3D arrow from ``origin`` along ``direction``."""
    o3d = _require_open3d()
    n = float(np.linalg.norm(direction))
    if n < 1e-9 or length < 1e-9:
        return None
    d = np.asarray(direction) / n
    cyl_r = 0.0018 * radius_scale
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cyl_r,
        cone_radius=cyl_r * 2.0,
        cylinder_height=length * 0.75,
        cone_height=length * 0.25,
    )
    # create_arrow points along +z; rotate +z -> d.
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d)
    s = np.linalg.norm(v)
    if s < 1e-9:
        R = np.eye(3) if d[2] > 0 else o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.array([np.pi, 0.0, 0.0])
        )
    else:
        c = float(np.dot(z, d))
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    arrow.rotate(R, center=(0, 0, 0))
    arrow.translate(np.asarray(origin))
    arrow.paint_uniform_color(color)
    arrow.compute_vertex_normals()
    return arrow


def contact_geometries(contacts, forces=None, force_scale=0.03):
    """3D contact markers (spheres) + optional force arrows."""
    o3d = _require_open3d()
    geoms = []
    contacts = np.asarray(contacts)
    for i, c in enumerate(contacts):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.006, resolution=12)
        s.translate(c)
        s.paint_uniform_color(_CONTACT_COLOR)
        s.compute_vertex_normals()
        geoms.append(s)
        if forces is not None:
            f = np.asarray(forces)[i]
            a = _arrow(c, f, force_scale * float(np.linalg.norm(f)), _CONTACT_COLOR)
            if a is not None:
                geoms.append(a)
    return geoms


def view_hand(
    hand_q,
    X_WP,
    obj_mesh=None,
    contacts=None,
    forces=None,
    sdf_path: Path | str | None = None,
    window_name: str = "equidexflow-demo",
):  # pragma: no cover - GUI
    """Open an interactive Open3D window with the posed hand mesh + object."""
    o3d = _require_open3d()
    geoms = build_hand_meshes(hand_q, X_WP, sdf_path=sdf_path)
    if obj_mesh is not None:
        geoms.append(trimesh_to_o3d(obj_mesh))
    if contacts is not None:
        geoms.extend(contact_geometries(contacts, forces))
    o3d.visualization.draw_geometries(geoms, window_name=window_name)


def render_hand_offscreen(
    out_path,
    hand_q,
    X_WP,
    obj_mesh=None,
    contacts=None,
    forces=None,
    sdf_path: Path | str | None = None,
    width: int = 1280,
    height: int = 960,
    azimuth_deg: float = 45.0,
    elevation_deg: float = 20.0,
):
    """Render the posed hand mesh + object to a PNG via Open3D OffscreenRenderer.

    Works headlessly where an EGL/OSMesa GL context is available.
    """
    o3d = _require_open3d()
    geoms = build_hand_meshes(hand_q, X_WP, sdf_path=sdf_path)
    if obj_mesh is not None:
        geoms.append(trimesh_to_o3d(obj_mesh))
    if contacts is not None:
        geoms.extend(contact_geometries(contacts, forces))

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.scene.set_sun_light([0.4, -0.5, -1.0], [1.0, 1.0, 1.0], 70000)
    renderer.scene.scene.enable_sun_light(True)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    for i, g in enumerate(geoms):
        renderer.scene.add_geometry(f"g{i}", g, mat)

    bb = renderer.scene.bounding_box
    center = (bb.get_max_bound() + bb.get_min_bound()) / 2.0
    radius = float(np.linalg.norm(bb.get_max_bound() - bb.get_min_bound())) / 2.0 or 0.1
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    eye = center + radius * 2.6 * np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
    )
    renderer.setup_camera(45.0, center, eye, np.array([0.0, 0.0, 1.0]))
    img = renderer.render_to_image()
    o3d.io.write_image(str(out_path), img)
    return out_path
