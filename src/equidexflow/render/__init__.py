"""Visual-mesh rendering for posed dexterous hands (Allegro).

Drake-free: per-link world transforms come from the pure-torch FK, mesh poses
from the hand SDF, and rasterization from Open3D (optional ``[demo]``/``[viz]``
dependency, imported lazily).
"""

from equidexflow.render.allegro_assets import VisualMesh, load_allegro_visuals
from equidexflow.render.hand_render import (
    build_hand_meshes,
    contact_geometries,
    render_hand_offscreen,
    trimesh_to_o3d,
    view_hand,
)

__all__ = [
    "VisualMesh",
    "load_allegro_visuals",
    "build_hand_meshes",
    "contact_geometries",
    "render_hand_offscreen",
    "trimesh_to_o3d",
    "view_hand",
]
