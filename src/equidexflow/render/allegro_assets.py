"""Read the Allegro hand's *visual* geometry out of its SDF.

The shipped renderer is Drake-free: instead of querying a Drake ``SceneGraph``
(as FRoGGeR's ``renderer._extract_visual_geometries`` does), we parse the
visual blocks of ``assets/hands/allegro/allegro_rh.sdf`` directly and pair them
with per-link world transforms from the pure-torch FK
(:meth:`AllegroRightHandFK.forward_link_frames`).

For each ``<link>`` we collect every ``<visual>``'s mesh path, scale, and the
mesh-in-body transform ``X_BG`` (from the visual ``<pose>``). A renderer then
places each mesh at ``X_WB @ X_BG`` where ``X_WB`` is the body's world pose.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _default_allegro_sdf() -> Path:
    """Locate the bundled Allegro SDF.

    Prefer the packaged copy (``equidexflow._allegro_hand``, present in wheel
    installs); fall back to the repo-root ``assets/hands/allegro`` used by
    editable installs and source checkouts.
    """
    try:
        from importlib.resources import files

        cand = Path(str(files("equidexflow._allegro_hand"))) / "allegro_rh.sdf"
        if cand.exists():
            return cand
    except (ImportError, ModuleNotFoundError, TypeError):
        pass
    return (
        Path(__file__).resolve().parents[2].parent
        / "assets" / "hands" / "allegro" / "allegro_rh.sdf"
    )


@dataclass
class VisualMesh:
    """One renderable visual mesh attached to a hand body."""

    link_name: str
    visual_name: str
    mesh_path: Path          # absolute path to the .obj
    X_BG: np.ndarray         # (4, 4) mesh pose in the body frame
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))


def _pose_to_matrix(pose_text: str | None) -> np.ndarray:
    """Parse an SDFormat ``<pose>x y z r p y</pose>`` into a 4x4 matrix.

    SDFormat uses extrinsic roll-pitch-yaw (fixed-axis x, y, z).
    """
    T = np.eye(4)
    if pose_text is None:
        return T
    vals = [float(v) for v in pose_text.split()]
    if len(vals) != 6:
        return T
    x, y, z, r, p, yw = vals
    T[:3, :3] = Rotation.from_euler("xyz", [r, p, yw]).as_matrix()
    T[:3, 3] = (x, y, z)
    return T


def _scale_to_vec(scale_text: str | None) -> np.ndarray:
    if scale_text is None:
        return np.ones(3)
    vals = [float(v) for v in scale_text.split()]
    if len(vals) == 1:
        return np.full(3, vals[0])
    if len(vals) == 3:
        return np.asarray(vals)
    return np.ones(3)


def _load_sdf_root(sdf_path: Path) -> ET.Element:
    """Parse the SDF, tolerating ``drake:``-prefixed tags in <collision> blocks.

    ``xml.etree`` raises "unbound prefix" on ``drake:`` elements because the SDF
    never declares that namespace. We only need the (clean) ``<visual>`` blocks,
    so we declare the namespace on the root tag before parsing -- no new
    dependency, and the collision blocks parse harmlessly.
    """
    text = sdf_path.read_text()
    if "xmlns:drake" not in text:
        text = re.sub(
            r"<sdf\b",
            '<sdf xmlns:drake="http://drake.mit.edu/schema/sdf"',
            text,
            count=1,
        )
    return ET.fromstring(text)


def load_allegro_visuals(sdf_path: Path | str | None = None) -> list[VisualMesh]:
    """Return every visual mesh in the Allegro SDF, with body-frame poses.

    Mesh URIs (``meshes/foo.obj``) are resolved relative to the SDF directory,
    so the result is independent of the current working directory.
    """
    sdf_path = Path(sdf_path) if sdf_path is not None else _default_allegro_sdf()
    sdf_path = sdf_path.resolve()
    if not sdf_path.exists():
        raise FileNotFoundError(f"Allegro SDF not found: {sdf_path}")
    sdf_dir = sdf_path.parent

    root = _load_sdf_root(sdf_path)
    out: list[VisualMesh] = []
    for link in root.iter("link"):
        link_name = link.get("name", "")
        for visual in link.findall("visual"):
            mesh_el = visual.find("./geometry/mesh")
            if mesh_el is None:
                continue
            uri_el = mesh_el.find("uri")
            if uri_el is None or not uri_el.text:
                continue
            mesh_path = (sdf_dir / uri_el.text.strip()).resolve()
            pose_el = visual.find("pose")
            scale_el = mesh_el.find("scale")
            out.append(
                VisualMesh(
                    link_name=link_name,
                    visual_name=visual.get("name", ""),
                    mesh_path=mesh_path,
                    X_BG=_pose_to_matrix(pose_el.text if pose_el is not None else None),
                    scale=_scale_to_vec(scale_el.text if scale_el is not None else None),
                )
            )
    return out
