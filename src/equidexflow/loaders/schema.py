"""
Canonical schema definitions for the EquiDexFlow dexterous grasp dataset.

Each public symbol documents the exact tensor shapes and dtypes expected.
"""

from typing import List
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict  # type: ignore[no-redef]

import torch

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

N_FINGERS: int = 4
"""Number of fingers on the LEAP hand (index, middle, ring, thumb)."""

MAX_CONTACTS: int = 64
"""Maximum number of contact points per grasp (used for fixed-size batching)."""

HAND_DOF: int = 16
"""Hand degrees of freedom: 4 fingers x 4 joints (LEAP)."""


# ---------------------------------------------------------------------------
# Per-example schema
# ---------------------------------------------------------------------------

class GraspExample(TypedDict):
    """
    Canonical per-example dict returned by ``DexGraspDBDataset.__getitem__``.

    All tensors are on CPU unless otherwise noted.

    Fields
    ------
    object_points : FloatTensor, shape (3, N_pts)
        Object surface point cloud in object frame, ordered (xyz, points).
        Convention matches EquiGraspFlow (channels-first).
    object_point_normals : FloatTensor, shape (3, N_pts)
        Per-point outward-pointing unit normals, same channels-first layout
        as ``object_points``. Zero where the source mesh is unavailable
        (fallback proxy path); the collision loss treats zero normals as
        "no surface signal" and skips the corresponding points.
    hand_q : FloatTensor, shape (HAND_DOF,) = (11,)
        Hand joint angles in radians.
        Order: [thumb_0..2, index_0..1, middle_0..1, ring_0..1, pinky_0..1].
    wrist_pose : FloatTensor, shape (4, 4)
        Wrist SE(3) transform in object frame (homogeneous matrix).
    contacts : FloatTensor, shape (MAX_CONTACTS, 3)
        Contact positions in meters, object frame. Zero-padded beyond n_valid.
    normals : FloatTensor, shape (MAX_CONTACTS, 3)
        Inward-pointing unit contact normals (pointing INTO the object).
        Zero-padded beyond n_valid.
    forces : FloatTensor, shape (MAX_CONTACTS, 3)
        Per-contact forces in Newtons satisfying quasistatic equilibrium.
        Zero-padded beyond n_valid.
    finger_ids : LongTensor, shape (MAX_CONTACTS,)
        Finger assignment: 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky.
        Padded entries are -1.
    valid_mask : BoolTensor, shape (MAX_CONTACTS,)
        True for the first n_valid entries; False for padding.
    object_name : str
        Human-readable object identifier.
    grasp_quality : FloatTensor, shape (2,)
        [epsilon_quality, volume_quality] from GWS metrics.
    """

    object_points: torch.Tensor          # (3, N_pts)        float32
    object_point_normals: torch.Tensor   # (3, N_pts)        float32
    hand_q: torch.Tensor                 # (11,)             float32
    wrist_pose: torch.Tensor      # (4, 4)             float32
    contacts: torch.Tensor        # (MAX_CONTACTS, 3)  float32
    normals: torch.Tensor         # (MAX_CONTACTS, 3)  float32
    forces: torch.Tensor          # (MAX_CONTACTS, 3)  float32
    finger_ids: torch.Tensor      # (MAX_CONTACTS,)    int64
    valid_mask: torch.Tensor      # (MAX_CONTACTS,)    bool
    object_name: str
    grasp_quality: torch.Tensor   # (2,)               float32


# ---------------------------------------------------------------------------
# Batched schema
# ---------------------------------------------------------------------------

class GraspBatch(TypedDict):
    """
    Batched version of ``GraspExample`` with a leading batch dimension B.

    Produced by ``torch.utils.data.DataLoader`` with the custom collate
    function from ``loaders.dexgrasp_db``.

    Fields (same semantics as ``GraspExample`` with shape prefix [B, ...])
    -----------------------------------------------------------------------
    object_points        : FloatTensor, shape (B, 3, N_pts)
    object_point_normals : FloatTensor, shape (B, 3, N_pts)
    hand_q               : FloatTensor, shape (B, 11)
    wrist_pose    : FloatTensor, shape (B, 4, 4)
    contacts      : FloatTensor, shape (B, MAX_CONTACTS, 3)
    normals       : FloatTensor, shape (B, MAX_CONTACTS, 3)
    forces        : FloatTensor, shape (B, MAX_CONTACTS, 3)
    finger_ids    : LongTensor,  shape (B, MAX_CONTACTS)
    valid_mask    : BoolTensor,  shape (B, MAX_CONTACTS)
    object_name   : List[str],   length B
    grasp_quality : FloatTensor, shape (B, 2)
    """

    object_points: torch.Tensor          # (B, 3, N_pts)
    object_point_normals: torch.Tensor   # (B, 3, N_pts)
    hand_q: torch.Tensor                 # (B, 11)
    wrist_pose: torch.Tensor      # (B, 4, 4)
    contacts: torch.Tensor        # (B, MAX_CONTACTS, 3)
    normals: torch.Tensor         # (B, MAX_CONTACTS, 3)
    forces: torch.Tensor          # (B, MAX_CONTACTS, 3)
    finger_ids: torch.Tensor      # (B, MAX_CONTACTS)
    valid_mask: torch.Tensor      # (B, MAX_CONTACTS)
    object_name: List[str]
    grasp_quality: torch.Tensor   # (B, 2)
