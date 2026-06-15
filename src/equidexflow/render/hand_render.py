"""Visual-mesh rendering of a posed Allegro hand grasping an object.

Pure-torch FK (:class:`AllegroRightHandFK`) supplies per-link world transforms;
the SDF (:mod:`equidexflow.render.allegro_assets`) supplies each link's visual
meshes and their body-frame poses. Open3D draws the real hand meshes, the object
mesh, and 3D contact markers/force arrows -- the FRoGGeR/DexGraspNet-style
visualization, Drake-free.

This module is the **shared, commodity core**: FK link frames -> placed visual
meshes -> Open3D scene -> offscreen/interactive render. Its public defaults
(:data:`DEFAULT_STYLE`) are deliberately neutral. Publication-grade look
(camera/lighting/material/palette/composition) is supplied by a separate, private
figure kit that passes its own :class:`RenderStyle` into these same functions, so
there is one rendering code path, not two.

Open3D is an optional (``[demo]`` / ``[viz]``) dependency and is imported lazily
so ``import equidexflow.render`` stays cheap for pure-inference users.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from equidexflow.kinematics.allegro_fk import AllegroRightHandFK
from equidexflow.render.allegro_assets import VisualMesh, load_allegro_visuals


@dataclass
class RenderStyle:
    """All look-and-feel knobs for a render. Defaults are the neutral *public*
    style: a clean visual-mesh viewer that is intentionally NOT the paper look.

    The private figure kit defines its own ``RenderStyle`` (palette, lighting,
    camera, supersampling, materials) and threads it through the same functions.
    """

    # Palette. Public default: a muted, presentable "lite" look -- a slightly
    # dark hand body with softly lighter fingertips (so tip vs link is legible),
    # over a soft-blue object. Intentionally tamer than the figure kit's
    # high-contrast near-black-body / bright-white-tip / steel-blue signature.
    body_color: tuple = (0.34, 0.34, 0.36)
    tip_color: tuple = (0.74, 0.74, 0.76)
    object_color: tuple = (0.55, 0.64, 0.82)
    contact_color: tuple = (0.85, 0.47, 0.36)
    distinguish_tips: bool = True           # tips render in tip_color (vs body)

    # Lighting / background.
    background: tuple = (1.0, 1.0, 1.0, 1.0)
    sun_dir: tuple = (0.3, -0.4, -1.0)
    sun_color: tuple = (1.0, 1.0, 1.0)
    sun_intensity: float = 45000.0

    # Material (Open3D MaterialRecord, defaultLit).
    base_roughness: float = 0.7
    base_metallic: float = 0.0

    # Camera.
    fov: float = 50.0
    azimuth_deg: float = 40.0
    elevation_deg: float = 22.0
    dist_scale: float = 2.7

    # Output.
    width: int = 1024
    height: int = 768
    supersample: int = 1                    # >1 => render big + box-downsample (AA)

    # Contact markers.
    contact_radius: float = 0.006
    force_scale: float = 0.03


DEFAULT_STYLE = RenderStyle()


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


def _visual_color(vm: VisualMesh, style: RenderStyle) -> tuple[float, float, float]:
    if style.distinguish_tips:
        name = f"{vm.visual_name} {vm.mesh_path.name}".lower()
        if "tip" in name:
            return style.tip_color
    return style.body_color


def build_hand_meshes(
    hand_q,
    X_WP,
    sdf_path: Path | str | None = None,
    fk: AllegroRightHandFK | None = None,
    style: RenderStyle = DEFAULT_STYLE,
):
    """Return a list of world-posed, colored Open3D meshes for the hand.

    Parameters
    ----------
    hand_q : (16,) array-like of joint angles.
    X_WP   : (4, 4) wrist pose in the world frame (base/palm frame).
    style  : palette source (body/tip colors, tip distinction).
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
        mesh.paint_uniform_color(_visual_color(vm, style))
        mesh.compute_vertex_normals()
        meshes.append(mesh)
    return meshes


def trimesh_to_o3d(mesh, color=None, style: RenderStyle = DEFAULT_STYLE):
    """Convert a trimesh mesh to a colored Open3D TriangleMesh."""
    o3d = _require_open3d()
    o = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
    )
    o.compute_vertex_normals()
    o.paint_uniform_color(color if color is not None else style.object_color)
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


def contact_geometries(contacts, forces=None, style: RenderStyle = DEFAULT_STYLE):
    """3D contact markers (spheres) + optional force arrows."""
    o3d = _require_open3d()
    geoms = []
    contacts = np.asarray(contacts)
    for i, c in enumerate(contacts):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=style.contact_radius, resolution=12)
        s.translate(c)
        s.paint_uniform_color(style.contact_color)
        s.compute_vertex_normals()
        geoms.append(s)
        if forces is not None:
            f = np.asarray(forces)[i]
            a = _arrow(c, f, style.force_scale * float(np.linalg.norm(f)), style.contact_color)
            if a is not None:
                geoms.append(a)
    return geoms


def build_scene_geometries(hand_q, X_WP, obj_mesh=None, contacts=None, forces=None,
                           sdf_path=None, fk=None, style: RenderStyle = DEFAULT_STYLE):
    """Hand meshes + (optional) object mesh + (optional) contact markers."""
    geoms = build_hand_meshes(hand_q, X_WP, sdf_path=sdf_path, fk=fk, style=style)
    if obj_mesh is not None:
        geoms.append(trimesh_to_o3d(obj_mesh, style=style))
    if contacts is not None:
        geoms.extend(contact_geometries(contacts, forces, style=style))
    return geoms


def view_hand(
    hand_q,
    X_WP,
    obj_mesh=None,
    contacts=None,
    forces=None,
    sdf_path: Path | str | None = None,
    style: RenderStyle = DEFAULT_STYLE,
    window_name: str = "equidexflow-demo",
):  # pragma: no cover - GUI
    """Open an interactive Open3D window with the posed hand mesh + object."""
    o3d = _require_open3d()
    geoms = build_scene_geometries(hand_q, X_WP, obj_mesh, contacts, forces,
                                   sdf_path=sdf_path, style=style)
    o3d.visualization.draw_geometries(geoms, window_name=window_name)


def _box_downsample(img: np.ndarray, ss: int) -> np.ndarray:
    """Average-pool an (H, W, C) image by integer factor ss (anti-aliasing)."""
    if ss <= 1:
        return img
    h, w = (img.shape[0] // ss) * ss, (img.shape[1] // ss) * ss
    img = img[:h, :w]
    return img.reshape(h // ss, ss, w // ss, ss, img.shape[2]).mean(axis=(1, 3)).astype(img.dtype)


def render_scene_to_array(geoms, style: RenderStyle = DEFAULT_STYLE) -> np.ndarray:
    """Rasterize Open3D geometries to an (H, W, 3) uint8 array with the style.

    Headless where an EGL/OSMesa GL context is available. Supersampling
    (``style.supersample`` > 1) renders large then box-downsamples for AA.
    """
    o3d = _require_open3d()
    ss = max(1, int(style.supersample))
    W, H = style.width * ss, style.height * ss

    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background(list(style.background))
    renderer.scene.scene.set_sun_light(
        list(style.sun_dir), list(style.sun_color), style.sun_intensity
    )
    renderer.scene.scene.enable_sun_light(True)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_roughness = style.base_roughness
    mat.base_metallic = style.base_metallic
    for i, g in enumerate(geoms):
        renderer.scene.add_geometry(f"g{i}", g, mat)

    bb = renderer.scene.bounding_box
    center = (bb.get_max_bound() + bb.get_min_bound()) / 2.0
    radius = float(np.linalg.norm(bb.get_max_bound() - bb.get_min_bound())) / 2.0 or 0.1
    az, el = np.radians(style.azimuth_deg), np.radians(style.elevation_deg)
    eye = center + radius * style.dist_scale * np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
    )
    renderer.setup_camera(style.fov, center, eye, np.array([0.0, 0.0, 1.0]))
    img = np.asarray(renderer.render_to_image())
    return _box_downsample(img, ss)


def render_scene_offscreen(out_path, geoms, style: RenderStyle = DEFAULT_STYLE):
    """Rasterize geometries to a PNG with the given style (see
    :func:`render_scene_to_array`)."""
    o3d = _require_open3d()
    img = render_scene_to_array(geoms, style)
    o3d.io.write_image(str(out_path), o3d.geometry.Image(np.ascontiguousarray(img)))
    return out_path


def render_hand_offscreen(
    out_path,
    hand_q,
    X_WP,
    obj_mesh=None,
    contacts=None,
    forces=None,
    sdf_path: Path | str | None = None,
    style: RenderStyle = DEFAULT_STYLE,
    # Back-compat overrides (None => take from style):
    width: int | None = None,
    height: int | None = None,
    azimuth_deg: float | None = None,
    elevation_deg: float | None = None,
):
    """Render the posed hand mesh + object to a PNG (neutral style by default)."""
    if any(v is not None for v in (width, height, azimuth_deg, elevation_deg)):
        from dataclasses import replace
        style = replace(
            style,
            width=width if width is not None else style.width,
            height=height if height is not None else style.height,
            azimuth_deg=azimuth_deg if azimuth_deg is not None else style.azimuth_deg,
            elevation_deg=elevation_deg if elevation_deg is not None else style.elevation_deg,
        )
    geoms = build_scene_geometries(hand_q, X_WP, obj_mesh, contacts, forces,
                                   sdf_path=sdf_path, style=style)
    return render_scene_offscreen(out_path, geoms, style=style)
