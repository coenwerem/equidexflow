"""
Pure-PyTorch forward kinematics for the LEAP hand (right hand).

Kinematic parameters are hard-coded from the URDF:
  frogger/models/leap_hand/leap_hand_rh.urdf

Joint ordering follows Drake's `MultibodyPlant.GetPositionNames()` for that
URDF (verified against the q_star meta JSONs produced by FRoGGeR):

  [0]   index_1   palm_lower      -> mcp_joint        axis=(0, 0, -1)
  [1]   index_0   mcp_joint       -> pip              axis=(0, 0, -1)
  [2]   index_2   pip             -> dip              axis=(0, 0, -1)
  [3]   index_3   dip             -> fingertip        axis=(0, 0, -1)
  [4]   middle_1  palm_lower      -> mcp_joint_2      axis=(0, 0, -1)
  [5]   middle_0  mcp_joint_2     -> pip_2            axis=(0, 0, -1)
  [6]   middle_2  pip_2           -> dip_2            axis=(0, 0, -1)
  [7]   middle_3  dip_2           -> fingertip_2      axis=(0, 0, -1)
  [8]   ring_1    palm_lower      -> mcp_joint_3      axis=(0, 0, -1)
  [9]   ring_0    mcp_joint_3     -> pip_3            axis=(0, 0, -1)
  [10]  ring_2    pip_3           -> dip_3            axis=(0, 0, -1)
  [11]  ring_3    dip_3           -> fingertip_3      axis=(0, 0, -1)
  [12]  thumb_0   palm_lower      -> thumb_temp_base  axis=(0, 0, -1)
  [13]  thumb_1   thumb_temp_base -> thumb_pip        axis=(0, 0, -1)
  [14]  thumb_2   thumb_pip       -> thumb_dip        axis=(0, 0, -1)
  [15]  thumb_3   thumb_dip       -> thumb_fingertip  axis=(0, 0, -1)

`wrist_pose` is the SE(3) pose of the URDF root `leap_hand_base` in the world
frame. The fixed transform `T_base_palm` (from `leap_hand_base_joint`) is
composed first so that all downstream chains start in the palm frame.
Fingertip world positions are returned in canonical order:

  [0] index   (fingertip + index_tip site)
  [1] middle  (fingertip_2 + middle_tip site)
  [2] ring    (fingertip_3 + ring_tip site)
  [3] thumb   (thumb_fingertip + thumb_tip site)

This module is SE(3)-equivariant by construction: rotating `wrist_pose` by R
rotates all output fingertip positions by R (the chain composes linearly).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Finger-centric wrist reparameterization
#
# leap_hand_base sits ~139 mm behind the fingertip centroid, so a small
# wrist-orientation error is amplified into a large fingertip miss. Training
# the SE(3) flow to predict a frame located at the grasp center (fingertip
# centroid) instead of the base shrinks that lever to ~53 mm (the fingertip
# spread), which is what keeps the predicted fingers on the object. The offset
# is a FIXED rigid translation in the base frame: the dataset-mean
# fingertip-centroid expressed in leap_hand_base (measured over all 8110
# v3/leap grasps). Pure relabel — orientation unchanged, only the origin moves.
# ---------------------------------------------------------------------------
LEAP_GRASP_CENTER_OFFSET = (0.1144, 0.0129, 0.0759)  # metres, base frame


def shift_wrist_frame(wrist_pose, to_base: bool):
    """Convert a LEAP wrist pose between base and grasp-center frame.

    wrist_pose : (...,4,4) torch.Tensor or np.ndarray.
      to_base=False : base -> grasp_center   (t += R @ c)
      to_base=True  : grasp_center -> base   (t -= R @ c)
    Orientation untouched. Returns a new array/tensor (input not mutated).
    """
    sign = -1.0 if to_base else 1.0
    if isinstance(wrist_pose, torch.Tensor):
        c = torch.as_tensor(LEAP_GRASP_CENTER_OFFSET,
                             dtype=wrist_pose.dtype, device=wrist_pose.device)
        out = wrist_pose.clone()
        out[..., :3, 3] = out[..., :3, 3] + sign * (out[..., :3, :3] @ c)
        return out
    c = np.asarray(LEAP_GRASP_CENTER_OFFSET, dtype=wrist_pose.dtype)
    out = np.array(wrist_pose, copy=True)
    out[..., :3, 3] = out[..., :3, 3] + sign * (out[..., :3, :3] @ c)
    return out


# ---------------------------------------------------------------------------
# Numpy helpers for building constant SE(3) matrices at import time
# ---------------------------------------------------------------------------

def _rpy_to_rot_np(rpy: list[float]) -> np.ndarray:
    """URDF roll-pitch-yaw (xyz extrinsic) -> 3x3 rotation matrix (float64)."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _T_urdf(pos: list[float], rpy: list[float] | None = None) -> np.ndarray:
    """Build a (4, 4) homogeneous transform from URDF origin tags."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = pos
    if rpy is not None:
        T[:3, :3] = _rpy_to_rot_np(rpy)
    return T


# ---------------------------------------------------------------------------
# Hard-coded constant transforms (extracted from URDF — do NOT parse at runtime)
# ---------------------------------------------------------------------------

# leap_hand_base -> palm_lower (fixed)
_T_base_palm = _T_urdf(pos=[0.0, 0.038, 0.098], rpy=[0.0, -1.57079, 0.0])

# ## Index ###################################################################
# palm_lower      --(index_1)--> mcp_joint
_T_palm_idx_mcp     = _T_urdf(pos=[-0.0070, 0.0230, -0.0187], rpy=[1.57079, 1.57079, 0.0])
# mcp_joint       --(index_0)--> pip
_T_idx_mcp_pip      = _T_urdf(pos=[-0.0122, 0.03810, 0.01450], rpy=[-1.57079, 0.0, 1.57079])
# pip             --(index_2)--> dip
_T_idx_pip_dip      = _T_urdf(pos=[0.015, 0.0143, -0.013], rpy=[1.57079, -1.57079, 0.0])
# dip             --(index_3)--> fingertip
_T_idx_dip_tip      = _T_urdf(pos=[0.0, -0.0361, 0.0002], rpy=[0.0, 0.0, 0.0])
# fingertip       --(index_tip fixed)--> index_tip_head
_T_idx_tip_head     = _T_urdf(pos=[0.0, -0.035, 0.015])

# ## Middle ##################################################################
_T_palm_mid_mcp     = _T_urdf(pos=[-0.0071, -0.0224, -0.0187], rpy=[1.57079, 1.57079, 0.0])
_T_mid_mcp_pip      = _T_urdf(pos=[-0.0122, 0.0381, 0.0145], rpy=[-1.57079, 0.0, 1.57079])
_T_mid_pip_dip      = _T_urdf(pos=[0.015, 0.0143, -0.013], rpy=[1.57079, -1.57079, 0.0])
_T_mid_dip_tip      = _T_urdf(pos=[0.0, -0.0361, 0.0002], rpy=[0.0, 0.0, 0.0])
_T_mid_tip_head     = _T_urdf(pos=[0.0, -0.035, 0.015])

# ## Ring ####################################################################
_T_palm_ring_mcp    = _T_urdf(pos=[-0.00709, -0.0678, -0.0187], rpy=[1.57079, 1.57079, 0.0])
_T_ring_mcp_pip     = _T_urdf(pos=[-0.0122, 0.0381, 0.0145], rpy=[-1.57079, 0.0, 1.57079])
_T_ring_pip_dip     = _T_urdf(pos=[0.015, 0.0143, -0.013], rpy=[1.57079, -1.57079, 0.0])
_T_ring_dip_tip     = _T_urdf(pos=[0.0, -0.03609, 0.0002], rpy=[0.0, 0.0, 0.0])
_T_ring_tip_head    = _T_urdf(pos=[0.0, -0.035, 0.015])

# ## Thumb ###################################################################
# palm_lower            --(thumb_0)--> thumb_temp_base
_T_palm_thumb_base  = _T_urdf(pos=[-0.0693, -0.0012, -0.0216], rpy=[0.0, 1.57079, 0.0])
# thumb_temp_base       --(thumb_1)--> thumb_pip
_T_thumb_base_pip   = _T_urdf(pos=[0.0, 0.0143, -0.013], rpy=[1.57079, -1.57079, 0.0])
# thumb_pip             --(thumb_2)--> thumb_dip
_T_thumb_pip_dip    = _T_urdf(pos=[0.0, 0.0145, -0.017], rpy=[-1.57079, 0.0, 0.0])
# thumb_dip             --(thumb_3)--> thumb_fingertip
_T_thumb_dip_tip    = _T_urdf(pos=[0.0, 0.0466, 0.0002], rpy=[0.0, 0.0, 3.14159])
# thumb_fingertip       --(thumb_tip fixed)--> thumb_tip_head
_T_thumb_tip_head   = _T_urdf(pos=[0.0, -0.040, -0.014])


# ---------------------------------------------------------------------------
# Pure-torch rotation utility (axis-angle, all 16 LEAP joints share axis (0,0,-1))
# ---------------------------------------------------------------------------

def _rot_z_neg(angle: torch.Tensor) -> torch.Tensor:
    """Rotation about axis (0, 0, -1) by `angle` radians.

    Equivalent to Rz(-angle). Differentiable; shape (..., 3, 3).
    """
    c = torch.cos(angle)
    s = torch.sin(angle)
    zero = torch.zeros_like(angle)
    one = torch.ones_like(angle)
    # R about +z by theta:
    #   [ c -s  0 ]
    #   [ s  c  0 ]
    #   [ 0  0  1 ]
    # About -z: replace theta with -theta -> s -> -s.
    R = torch.stack(
        [
             c,  s, zero,
            -s,  c, zero,
            zero, zero, one,
        ],
        dim=-1,
    )
    return R.reshape(*angle.shape, 3, 3)


def _compose(T_a: torch.Tensor, T_b: torch.Tensor) -> torch.Tensor:
    """Right-multiply two SE(3) batches: T_a @ T_b. Broadcasting in batch dims."""
    return T_a @ T_b


def _T_joint(T_fixed: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Compose a fixed link transform with an axis-(0,0,-1) joint rotation.

    Returns T_fixed @ Rz_neg(theta) lifted to SE(3).
    """
    R = _rot_z_neg(theta)  # (..., 3, 3)
    # Build (..., 4, 4) from R, identity translation.
    shape = R.shape[:-2] + (4, 4)
    Tj = torch.zeros(shape, device=R.device, dtype=R.dtype)
    Tj[..., :3, :3] = R
    Tj[..., 3, 3] = 1.0
    return T_fixed @ Tj


# ---------------------------------------------------------------------------
# LeapFK
# ---------------------------------------------------------------------------

class LeapFK(nn.Module):
    """Pure-torch forward kinematics for the LEAP hand (right).

    All 16 LEAP joints share axis (0, 0, -1) in their child frames, so the
    per-joint rotation is `_rot_z_neg(theta)`. The joint ordering matches
    Drake's position order and the `hand_dof_values` field emitted by the
    dexgraspdb adapter.

    `wrist_pose` is the SE(3) pose of `leap_hand_base` in the world frame
    (the first 7 entries of the FRoGGeR `q_star`). The fixed transform
    `leap_hand_base_joint` (T_base_palm) is composed inside `forward()`.
    """

    # Joint axes (all (0, 0, -1) by URDF inspection)
    JOINT_AXES: np.ndarray = np.tile(
        np.array([0.0, 0.0, -1.0], dtype=np.float64),
        (16, 1),
    )

    # Joint limits [lo, hi] in radians (from URDF <limit ... lower upper>)
    JOINT_RANGES: np.ndarray = np.array(
        [
            [-0.314, 2.23],   # [0]  index_1
            [-1.047, 1.047],  # [1]  index_0
            [-0.506, 1.885],  # [2]  index_2
            [-0.366, 2.042],  # [3]  index_3
            [-0.314, 2.23],   # [4]  middle_1
            [-1.047, 1.047],  # [5]  middle_0
            [-0.506, 1.885],  # [6]  middle_2
            [-0.366, 2.042],  # [7]  middle_3
            [-0.314, 2.23],   # [8]  ring_1
            [-1.047, 1.047],  # [9]  ring_0
            [-0.506, 1.885],  # [10] ring_2
            [-0.366, 2.042],  # [11] ring_3
            [-0.314, 2.23],   # [12] thumb_0
            [-1.047, 1.047],  # [13] thumb_1
            [-0.506, 1.885],  # [14] thumb_2
            [-0.366, 2.042],  # [15] thumb_3
        ],
        dtype=np.float64,
    )

    N_FINGERS: int = 4
    HAND_DOF: int = 16

    def __init__(self) -> None:
        super().__init__()
        # Register all constant SE(3) transforms as float32 buffers so that
        # .to(device) / .cuda() propagates them automatically.
        self._reg("T_base_palm",         _T_base_palm)
        # Index chain
        self._reg("T_palm_idx_mcp",      _T_palm_idx_mcp)
        self._reg("T_idx_mcp_pip",       _T_idx_mcp_pip)
        self._reg("T_idx_pip_dip",       _T_idx_pip_dip)
        self._reg("T_idx_dip_tip",       _T_idx_dip_tip)
        self._reg("T_idx_tip_head",      _T_idx_tip_head)
        # Middle chain
        self._reg("T_palm_mid_mcp",      _T_palm_mid_mcp)
        self._reg("T_mid_mcp_pip",       _T_mid_mcp_pip)
        self._reg("T_mid_pip_dip",       _T_mid_pip_dip)
        self._reg("T_mid_dip_tip",       _T_mid_dip_tip)
        self._reg("T_mid_tip_head",      _T_mid_tip_head)
        # Ring chain
        self._reg("T_palm_ring_mcp",     _T_palm_ring_mcp)
        self._reg("T_ring_mcp_pip",      _T_ring_mcp_pip)
        self._reg("T_ring_pip_dip",      _T_ring_pip_dip)
        self._reg("T_ring_dip_tip",      _T_ring_dip_tip)
        self._reg("T_ring_tip_head",     _T_ring_tip_head)
        # Thumb chain
        self._reg("T_palm_thumb_base",   _T_palm_thumb_base)
        self._reg("T_thumb_base_pip",    _T_thumb_base_pip)
        self._reg("T_thumb_pip_dip",     _T_thumb_pip_dip)
        self._reg("T_thumb_dip_tip",     _T_thumb_dip_tip)
        self._reg("T_thumb_tip_head",    _T_thumb_tip_head)

        # Body collision sphere config (URDF-derived; see
        # leap_collision_spheres.json). Each sphere is parameterized by
        # (chain, step) into the cumulative-link-transform table built in
        # forward_all_spheres, plus a local offset and radius.
        import json as _json
        from pathlib import Path as _Path
        sphere_cfg_path = _Path(__file__).resolve().parent / "leap_collision_spheres.json"
        cfg = _json.loads(sphere_cfg_path.read_text())
        body_spheres = cfg["body_spheres"]
        n_body = len(body_spheres)
        self._n_body_spheres = n_body
        # Keep chain as a Python list (string), step/offset/radius as tensors.
        self._sphere_chain: list[str] = [s["chain"] for s in body_spheres]
        self.register_buffer("_sphere_step",
                              torch.tensor([s["step"] for s in body_spheres],
                                           dtype=torch.long))
        self.register_buffer("_sphere_offsets",
                              torch.tensor([s["offset_m"] for s in body_spheres],
                                           dtype=torch.float32))
        self.register_buffer("_sphere_radii",
                              torch.tensor([s["radius_m"] for s in body_spheres],
                                           dtype=torch.float32))
        # Tip-sphere radius — registered as a buffer so `.to(device)` moves
        # it alongside the other tensors used in forward_all_spheres.
        self.register_buffer("_fingertip_radius",
                              torch.tensor(0.008, dtype=torch.float32))

    def _reg(self, name: str, T_np: np.ndarray) -> None:
        T = torch.from_numpy(T_np).to(dtype=torch.float32)
        self.register_buffer(name, T)

    def _chain(
        self,
        T_palm: torch.Tensor,     # (B, 4, 4)
        T_link_seq: list,         # list of registered buffers (4, 4)
        thetas: torch.Tensor,     # (B, 4) joint angles for the 4 dofs of this chain
        T_tip_head: torch.Tensor, # (4, 4) fixed fingertip site transform
    ) -> torch.Tensor:
        """Compose one finger chain: T_palm -> joint -> link -> ... -> tip_head.

        Returns (B, 3) world-frame tip-head position.
        """
        T = T_palm  # (B, 4, 4)
        for i, T_fixed in enumerate(T_link_seq):
            # Each step: T = T @ (T_fixed @ Rz_neg(theta_i))
            T_step = _T_joint(T_fixed.unsqueeze(0), thetas[:, i])  # (B, 4, 4)
            T = T @ T_step
        # Final fixed tip-head transform (no joint here)
        T = T @ T_tip_head.unsqueeze(0)
        return T[:, :3, 3]  # (B, 3)

    def forward(
        self,
        q: torch.Tensor,           # (B, 16) joint angles in Drake position order
        wrist_pose: torch.Tensor,  # (B, 4, 4) leap_hand_base pose in world frame
    ) -> torch.Tensor:
        """Compute the 4 fingertip world-frame positions.

        Returns
        -------
        tips : (B, 4, 3) in canonical order [index, middle, ring, thumb].
        """
        if q.shape[-1] != self.HAND_DOF:
            raise ValueError(f"q last-dim {q.shape[-1]}, expected {self.HAND_DOF}")
        B = q.shape[0]

        T_palm = wrist_pose @ self.T_base_palm.unsqueeze(0)  # (B, 4, 4)

        # Index chain joints 0..3 (q[:, 0:4])
        tip_idx = self._chain(
            T_palm,
            [self.T_palm_idx_mcp, self.T_idx_mcp_pip,
             self.T_idx_pip_dip, self.T_idx_dip_tip],
            q[:, 0:4],
            self.T_idx_tip_head,
        )
        # Middle chain joints 4..7
        tip_mid = self._chain(
            T_palm,
            [self.T_palm_mid_mcp, self.T_mid_mcp_pip,
             self.T_mid_pip_dip, self.T_mid_dip_tip],
            q[:, 4:8],
            self.T_mid_tip_head,
        )
        # Ring chain joints 8..11
        tip_ring = self._chain(
            T_palm,
            [self.T_palm_ring_mcp, self.T_ring_mcp_pip,
             self.T_ring_pip_dip, self.T_ring_dip_tip],
            q[:, 8:12],
            self.T_ring_tip_head,
        )
        # Thumb chain joints 12..15
        tip_thumb = self._chain(
            T_palm,
            [self.T_palm_thumb_base, self.T_thumb_base_pip,
             self.T_thumb_pip_dip, self.T_thumb_dip_tip],
            q[:, 12:16],
            self.T_thumb_tip_head,
        )

        return torch.stack([tip_idx, tip_mid, tip_ring, tip_thumb], dim=1)

    # ------------------------------------------------------------------
    # All collision spheres (link bodies + fingertips) — for collision loss
    # ------------------------------------------------------------------

    @property
    def fingertip_radius(self) -> torch.Tensor:
        """Scalar tip-sphere radius (m). Matches project_wrist.py's
        _TIP_RADIUS_BY_HAND['leap'] = 0.008 (LEAP fingertip 8mm).
        Backed by a registered buffer so .to(device) moves it correctly."""
        return self._fingertip_radius

    @property
    def n_collision_spheres(self) -> int:
        """Total spheres returned by forward_all_spheres (body + tips).
        Matches Allegro's contract: link spheres + N_FINGERS fingertip spheres."""
        return self._sphere_offsets.shape[0] + self.N_FINGERS

    def _chain_with_link_transforms(
        self,
        T_palm: torch.Tensor,   # (B, 4, 4)
        T_link_seq: list,       # 4 fixed link transforms
        thetas: torch.Tensor,   # (B, 4) joint angles
    ) -> list:
        """Same chain composition as `_chain`, but return the per-step
        cumulative world-frame transform after each (fixed × joint) compose.
        list length = 4 (one per link in the chain)."""
        T = T_palm
        out = []
        for i, T_fixed in enumerate(T_link_seq):
            T_step = _T_joint(T_fixed.unsqueeze(0), thetas[:, i])
            T = T @ T_step
            out.append(T)
        return out

    def forward_all_spheres(
        self,
        hand_q: torch.Tensor,    # (B, 16)
        X_WP: torch.Tensor,      # (B, 4, 4) leap_hand_base pose in world
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FK to ALL collision sphere positions (link bodies + fingertips).

        Mirrors allegro_fk.py:283 (`AllegroRightHandFK.forward_all_spheres`)
        so that physics_loss / scorer / model wiring is hand-agnostic. The
        sphere set is URDF-derived (37 body spheres tiled inside the LEAP
        URDF's per-link <collision><box> entries + 4 tip spheres at the
        *_tip_head sites). See `leap_collision_spheres.json`.

        Returns
        -------
        positions : (B, N_total, 3) world positions of all spheres. The
            last `N_FINGERS` entries are the fingertip contact spheres.
        radii     : (N_total,) per-sphere radius in metres.
        """
        if hand_q.dim() == 1:
            hand_q = hand_q.unsqueeze(0)
        if X_WP.dim() == 2:
            X_WP = X_WP.unsqueeze(0)
        if hand_q.shape[-1] != self.HAND_DOF:
            raise ValueError(
                f"hand_q last-dim {hand_q.shape[-1]}, expected {self.HAND_DOF}")

        B = hand_q.shape[0]
        device = hand_q.device
        dtype = hand_q.dtype

        T_palm = X_WP @ self.T_base_palm.unsqueeze(0)  # (B, 4, 4)

        # Per-chain per-link cumulative transforms
        idx_links = self._chain_with_link_transforms(
            T_palm,
            [self.T_palm_idx_mcp, self.T_idx_mcp_pip,
             self.T_idx_pip_dip, self.T_idx_dip_tip],
            hand_q[:, 0:4])
        mid_links = self._chain_with_link_transforms(
            T_palm,
            [self.T_palm_mid_mcp, self.T_mid_mcp_pip,
             self.T_mid_pip_dip, self.T_mid_dip_tip],
            hand_q[:, 4:8])
        ring_links = self._chain_with_link_transforms(
            T_palm,
            [self.T_palm_ring_mcp, self.T_ring_mcp_pip,
             self.T_ring_pip_dip, self.T_ring_dip_tip],
            hand_q[:, 8:12])
        thumb_links = self._chain_with_link_transforms(
            T_palm,
            [self.T_palm_thumb_base, self.T_thumb_base_pip,
             self.T_thumb_pip_dip, self.T_thumb_dip_tip],
            hand_q[:, 12:16])

        # Tip-head transforms (final fixed transform per chain)
        T_idx_tip   = idx_links[-1]   @ self.T_idx_tip_head.unsqueeze(0)
        T_mid_tip   = mid_links[-1]   @ self.T_mid_tip_head.unsqueeze(0)
        T_ring_tip  = ring_links[-1]  @ self.T_ring_tip_head.unsqueeze(0)
        T_thumb_tip = thumb_links[-1] @ self.T_thumb_tip_head.unsqueeze(0)

        # Stack body transforms in the order body_spheres were registered
        # (see __init__ for the chain-step → buffer-row mapping).
        chain_tables = {
            ("palm",   0): T_palm,
            ("index",  0): idx_links[0], ("index",  1): idx_links[1],
            ("index",  2): idx_links[2], ("index",  3): idx_links[3],
            ("middle", 0): mid_links[0], ("middle", 1): mid_links[1],
            ("middle", 2): mid_links[2], ("middle", 3): mid_links[3],
            ("ring",   0): ring_links[0], ("ring",  1): ring_links[1],
            ("ring",   2): ring_links[2], ("ring",  3): ring_links[3],
            ("thumb",  0): thumb_links[0], ("thumb", 1): thumb_links[1],
            ("thumb",  2): thumb_links[2], ("thumb", 3): thumb_links[3],
        }
        # Body sphere positions
        body_pos = torch.zeros(B, self._n_body_spheres, 3, device=device, dtype=dtype)
        for i in range(self._n_body_spheres):
            chain = self._sphere_chain[i]
            step = int(self._sphere_step[i].item())
            T = chain_tables[(chain, step)]                       # (B, 4, 4)
            off4 = torch.zeros(B, 4, device=device, dtype=dtype)
            off4[:, :3] = self._sphere_offsets[i]
            off4[:, 3]  = 1.0
            world = (T @ off4.unsqueeze(-1)).squeeze(-1)
            body_pos[:, i] = world[:, :3]

        # Tip sphere positions (offset=0 at tip head; just take translation)
        tip_pos = torch.stack([
            T_idx_tip[:, :3, 3],
            T_mid_tip[:, :3, 3],
            T_ring_tip[:, :3, 3],
            T_thumb_tip[:, :3, 3],
        ], dim=1)  # (B, 4, 3)

        all_pos = torch.cat([body_pos, tip_pos], dim=1)
        all_radii = torch.cat([
            self._sphere_radii,
            self.fingertip_radius.expand(self.N_FINGERS).to(dtype),
        ])
        return all_pos, all_radii
