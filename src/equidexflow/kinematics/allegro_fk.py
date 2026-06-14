"""
Differentiable forward kinematics for the Allegro right hand (16 DOF).

Reads the kinematic chain extracted from frogger's Drake-parsed SDF
(``allegro_rh_kinematics.json``) and applies the joint transforms in pure
torch so gradients flow from fingertip world positions back through joint
angles and wrist pose.

Usage
-----
>>> fk = AllegroRightHandFK().to(device)
>>> # hand_q: (B, 16) joint angles; X_WP: (B, 4, 4) wrist pose in world frame
>>> tips_W = fk(hand_q, X_WP)               # (B, 4, 3) fingertip centres
>>> tips_W_with_r, radii = fk.with_radii(hand_q, X_WP)
>>> # radii: (4,) fingertip sphere radii in metres

Joint order in ``hand_q`` (must match the dataset emitter)::

    [index_axl, index_mcp, index_pip, index_dip,      # 0..3
     middle_axl, middle_mcp, middle_pip, middle_dip,  # 4..7
     ring_axl, ring_mcp, ring_pip, ring_dip,          # 8..11
     thumb_cmc, thumb_axl, thumb_mcp, thumb_ipl]      # 12..15

Drake convention (verified by extraction):
    For each revolute joint with parent frame X_PJ on the parent body and
    rotation axis ``axis`` (in the joint frame), the child body frame is
    related to the parent body frame by

        X_PC(q) = X_PJ * R_axis(q)

    where R_axis(q) is the rotation matrix about ``axis`` by angle ``q``.
    The ``frame_on_child`` is identity (verified during extraction).

This module is autograd-compatible and adds <1ms per batch on GPU at B=8.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_KINEMATICS_JSON = Path(__file__).parent / "allegro_rh_kinematics.json"

_FINGER_NAMES = ("index", "middle", "ring", "thumb")

# ---------------------------------------------------------------------------
# Finger-centric wrist reparameterization (mirrors kinematics.leap)
#
# The Allegro palm frame sits ~141 mm behind the contact centroid, so a small
# wrist-orientation error is amplified into a large fingertip miss. Training
# the SE(3) flow to predict a frame at the grasp center (dataset-mean contact
# centroid) instead of the palm shrinks that lever arm. The offset is a FIXED
# rigid translation in the palm frame: the dataset-mean contact centroid
# expressed in the Allegro palm frame (measured over 8099 v3/allegro grasps).
# Pure relabel: orientation unchanged, only the origin moves.
# ---------------------------------------------------------------------------
ALLEGRO_GRASP_CENTER_OFFSET = (0.0621, 0.0140, 0.1255)  # metres, palm frame


def shift_wrist_frame(wrist_pose, to_base: bool):
    """Convert an Allegro wrist pose between palm (base) and grasp-center frame.

    wrist_pose : (...,4,4) torch.Tensor or np.ndarray.
      to_base=False : base -> grasp_center   (t += R @ c)
      to_base=True  : grasp_center -> base   (t -= R @ c)
    Orientation untouched. Returns a new array/tensor (input not mutated).
    """
    sign = -1.0 if to_base else 1.0
    if isinstance(wrist_pose, torch.Tensor):
        c = torch.as_tensor(ALLEGRO_GRASP_CENTER_OFFSET,
                            dtype=wrist_pose.dtype, device=wrist_pose.device)
        out = wrist_pose.clone()
        out[..., :3, 3] = out[..., :3, 3] + sign * (out[..., :3, :3] @ c)
        return out
    c = np.asarray(ALLEGRO_GRASP_CENTER_OFFSET, dtype=wrist_pose.dtype)
    out = np.array(wrist_pose, copy=True)
    out[..., :3, 3] = out[..., :3, 3] + sign * (out[..., :3, :3] @ c)
    return out


def _snap_axis(axis_list: list[float]) -> list[float]:
    """Snap a near-cardinal axis to the closest unit vector along ±x/y/z.

    The Drake-extracted axes carry float noise (e.g. ``[0, -3e-7, 1]``).
    For numerical stability and to keep the rotation formula clean we
    snap when an axis is unambiguously cardinal.
    """
    a = torch.tensor(axis_list, dtype=torch.float64)
    a = a / a.norm().clamp(min=1e-12)
    snap_thresh = 1e-3
    out = a.clone()
    for i in range(3):
        if abs(abs(out[i]) - 1.0) < snap_thresh:
            sign = 1.0 if out[i] > 0 else -1.0
            out[:] = 0.0
            out[i] = sign
            return out.tolist()
    return out.tolist()


def _axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula, batched.

    Args:
        axis:  (3,) unit vector (single fixed axis, broadcast across batch)
        angle: (B,) rotation angle in radians

    Returns:
        R: (B, 3, 3) rotation matrices
    """
    B = angle.shape[0]
    ax = axis.to(angle.device).to(angle.dtype)              # (3,)
    K = torch.zeros(3, 3, device=ax.device, dtype=ax.dtype)
    K[0, 1] = -ax[2]; K[0, 2] =  ax[1]
    K[1, 0] =  ax[2]; K[1, 2] = -ax[0]
    K[2, 0] = -ax[1]; K[2, 1] =  ax[0]
    K2 = K @ K
    I = torch.eye(3, device=ax.device, dtype=ax.dtype)
    s = torch.sin(angle).view(B, 1, 1)
    c = (1.0 - torch.cos(angle)).view(B, 1, 1)
    R = I.unsqueeze(0) + s * K.unsqueeze(0) + c * K2.unsqueeze(0)
    return R  # (B, 3, 3)


def _compose(T_a: torch.Tensor, T_b: torch.Tensor) -> torch.Tensor:
    """Batched SE(3) composition: T_a @ T_b, both shape (..., 4, 4)."""
    return torch.matmul(T_a, T_b)


def _se3_from_R_p(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Stack a rotation (B, 3, 3) and translation (B, 3) into SE(3) (B, 4, 4)."""
    B = R.shape[0]
    T = torch.zeros(B, 4, 4, device=R.device, dtype=R.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = p
    T[:, 3, 3] = 1.0
    return T


_MIN_LINK_LENGTH = 0.030  # 30mm - skip base links (~16mm)


class AllegroRightHandFK(nn.Module):
    """Forward kinematics for the 16-DOF Allegro right hand.

    All static kinematic parameters (joint axes, frame-on-parent transforms,
    fingertip offsets) are registered as buffers so they move with ``.to(device)``
    and are saved/loaded with the model state.
    """

    HAND_DOF: int = 16
    N_FINGERS: int = 4

    def __init__(self, config_path: Path = _KINEMATICS_JSON):
        super().__init__()
        with open(config_path) as fh:
            cfg = json.load(fh)

        # Build (4 fingers × 4 joints) packed tensors.
        # joint_axes_local : (16, 3) - joint axis in joint frame
        # X_PJ_R           : (16, 3, 3) - rotation of joint frame in parent body
        # X_PJ_p           : (16, 3) - translation of joint frame in parent body
        joint_axes_local = torch.zeros(self.HAND_DOF, 3)
        X_PJ_R = torch.zeros(self.HAND_DOF, 3, 3)
        X_PJ_p = torch.zeros(self.HAND_DOF, 3)
        # Body (link) names per joint, in hand_q order, plus the wrist/root link.
        # These match the <link name=...> entries in allegro_rh.sdf and let
        # forward_link_frames() key world transforms by body for mesh rendering.
        link_names: list[str] = [""] * self.HAND_DOF
        root_link_name = cfg["fingers"][_FINGER_NAMES[0]][0]["parent"]
        for fi, fname in enumerate(_FINGER_NAMES):
            chain = cfg["fingers"][fname]
            for ji, jcfg in enumerate(chain):
                idx = fi * 4 + ji
                joint_axes_local[idx] = torch.tensor(_snap_axis(jcfg["axis"]))
                X_PJ_R[idx] = torch.tensor(jcfg["R_PJ"])
                X_PJ_p[idx] = torch.tensor(jcfg["p_PJ"])
                link_names[idx] = jcfg["child"]
        self.link_names = link_names              # 16 child-body names (hand_q order)
        self.root_link_name = root_link_name      # wrist/palm body (== X_WP frame)

        # fingertip offset in distal-link body frame (4, 3)
        ftip = torch.zeros(self.N_FINGERS, 3)
        for fi, fname in enumerate(_FINGER_NAMES):
            ftip[fi] = torch.tensor(cfg["fingertip_offsets"][fname])

        self.register_buffer("joint_axes_local", joint_axes_local)  # (16, 3)
        self.register_buffer("X_PJ_R", X_PJ_R)                       # (16, 3, 3)
        self.register_buffer("X_PJ_p", X_PJ_p)                       # (16, 3)
        self.register_buffer("fingertip_offsets", ftip)              # (4, 3)
        self.register_buffer(
            "fingertip_radius",
            torch.tensor(float(cfg["fingertip_radius"])),
        )

        # --- Link collision spheres for intermediate phalanges ---------------
        # For each joint j in {0..2}, the child body extends to the next joint
        # at p_PJ[j+1].  For joint 3, it extends to the fingertip offset.
        # We place a sphere at the midpoint of each link longer than
        # _MIN_LINK_LENGTH, with radius proportional to finger cross-section.
        _sphere_finger: list[int] = []
        _sphere_depth: list[int] = []
        _sphere_offset: list[list[float]] = []
        _sphere_radius: list[float] = []
        _sphere_is_tip: list[bool] = []

        for fi, fname in enumerate(_FINGER_NAMES):
            chain = cfg["fingers"][fname]
            for ji in range(4):
                if ji < 3:
                    extent = torch.tensor(chain[ji + 1]["p_PJ"])
                else:
                    extent = torch.tensor(cfg["fingertip_offsets"][fname])
                link_len = extent.norm().item()
                if link_len < _MIN_LINK_LENGTH:
                    continue
                _sphere_finger.append(fi)
                _sphere_depth.append(ji)
                _sphere_offset.append((0.5 * extent).tolist())
                _sphere_radius.append(min(0.014, max(0.010, link_len * 0.25)))
                _sphere_is_tip.append(False)

        n_link = len(_sphere_finger)
        self.register_buffer(
            "_link_sphere_finger",
            torch.tensor(_sphere_finger, dtype=torch.long),
        )
        self.register_buffer(
            "_link_sphere_depth",
            torch.tensor(_sphere_depth, dtype=torch.long),
        )
        self.register_buffer(
            "_link_sphere_offsets",
            torch.tensor(_sphere_offset),
        )
        self.register_buffer(
            "_link_sphere_radii",
            torch.tensor(_sphere_radius),
        )
        self._n_link_spheres = n_link

    # ------------------------------------------------------------------
    # Shared: batched joint transforms
    # ------------------------------------------------------------------

    def _compute_X_PC(
        self,
        hand_q: torch.Tensor,
    ) -> torch.Tensor:
        """(B, 16, 4, 4) per-joint parent→child transforms in the local frame."""
        B = hand_q.shape[0]
        device = hand_q.device
        dtype = hand_q.dtype

        q_flat = hand_q.reshape(B * self.HAND_DOF)
        axes_expand = (
            self.joint_axes_local.unsqueeze(0)
            .expand(B, -1, -1)
            .reshape(B * self.HAND_DOF, 3)
        )
        K = torch.zeros(B * self.HAND_DOF, 3, 3, device=device, dtype=dtype)
        K[:, 0, 1] = -axes_expand[:, 2]; K[:, 0, 2] =  axes_expand[:, 1]
        K[:, 1, 0] =  axes_expand[:, 2]; K[:, 1, 2] = -axes_expand[:, 0]
        K[:, 2, 0] = -axes_expand[:, 1]; K[:, 2, 1] =  axes_expand[:, 0]
        K2 = torch.bmm(K, K)
        I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        s = torch.sin(q_flat).view(-1, 1, 1)
        c = (1.0 - torch.cos(q_flat)).view(-1, 1, 1)
        R_joint = (I + s * K + c * K2).view(B, self.HAND_DOF, 3, 3)

        X_PJ = torch.zeros(self.HAND_DOF, 4, 4, device=device, dtype=dtype)
        X_PJ[:, :3, :3] = self.X_PJ_R
        X_PJ[:, :3, 3] = self.X_PJ_p
        X_PJ[:, 3, 3] = 1.0

        X_J = torch.zeros(B, self.HAND_DOF, 4, 4, device=device, dtype=dtype)
        X_J[:, :, :3, :3] = R_joint
        X_J[:, :, 3, 3] = 1.0
        return X_PJ.unsqueeze(0) @ X_J  # (B, 16, 4, 4)

    # ------------------------------------------------------------------
    # Fingertip-only FK (backward compatible)
    # ------------------------------------------------------------------

    def forward(
        self,
        hand_q: torch.Tensor,     # (B, 16)
        X_WP: torch.Tensor,       # (B, 4, 4) wrist pose in world frame
    ) -> torch.Tensor:            # (B, 4, 3) fingertip positions in world frame
        """FK from joint angles + wrist pose to fingertip world positions."""
        if hand_q.dim() == 1:
            hand_q = hand_q.unsqueeze(0)
        if X_WP.dim() == 2:
            X_WP = X_WP.unsqueeze(0)
        if hand_q.shape[-1] != self.HAND_DOF:
            raise ValueError(
                f"hand_q last-dim {hand_q.shape[-1]}, expected {self.HAND_DOF}"
            )

        B = hand_q.shape[0]
        device = hand_q.device
        dtype = hand_q.dtype
        X_PC = self._compute_X_PC(hand_q)

        tips_world = torch.zeros(B, self.N_FINGERS, 3, device=device, dtype=dtype)
        for fi in range(self.N_FINGERS):
            X = X_WP
            for ji in range(4):
                X = X @ X_PC[:, fi * 4 + ji]
            offset = torch.zeros(B, 4, device=device, dtype=dtype)
            offset[:, :3] = self.fingertip_offsets[fi]
            offset[:, 3] = 1.0
            tip = (X @ offset.unsqueeze(-1)).squeeze(-1)
            tips_world[:, fi] = tip[:, :3]

        return tips_world

    # ------------------------------------------------------------------
    # Per-link body frames (for visual-mesh rendering)
    # ------------------------------------------------------------------

    def forward_link_frames(
        self,
        hand_q: torch.Tensor,     # (B, 16) or (16,)
        X_WP: torch.Tensor,       # (B, 4, 4) or (4, 4) wrist pose in world frame
    ) -> dict[str, torch.Tensor]:
        """World pose (B, 4, 4) of every hand body, keyed by SDF link name.

        The root/wrist body (``self.root_link_name``) is ``X_WP`` itself; each
        finger link is the accumulated chain ``X_WP @ X_PC[0] @ ... @ X_PC[ji]``
        down its joint chain. Body names match ``allegro_rh.sdf`` so a renderer
        can place each link's visual mesh by ``X_WB @ X_BG`` (mesh-in-body pose
        read from the SDF). Autograd-compatible.
        """
        if hand_q.dim() == 1:
            hand_q = hand_q.unsqueeze(0)
        if X_WP.dim() == 2:
            X_WP = X_WP.unsqueeze(0)
        if hand_q.shape[-1] != self.HAND_DOF:
            raise ValueError(
                f"hand_q last-dim {hand_q.shape[-1]}, expected {self.HAND_DOF}"
            )

        X_PC = self._compute_X_PC(hand_q)  # (B, 16, 4, 4)
        frames: dict[str, torch.Tensor] = {self.root_link_name: X_WP}
        for fi in range(self.N_FINGERS):
            X = X_WP
            for ji in range(4):
                X = X @ X_PC[:, fi * 4 + ji]
                frames[self.link_names[fi * 4 + ji]] = X
        return frames

    # ------------------------------------------------------------------
    # All collision spheres (link midpoints + fingertips)
    # ------------------------------------------------------------------

    def forward_all_spheres(
        self,
        hand_q: torch.Tensor,     # (B, 16)
        X_WP: torch.Tensor,       # (B, 4, 4) wrist pose in world frame
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FK to ALL collision sphere positions (link midpoints + fingertips).

        Returns
        -------
        positions : (B, N_total, 3) world positions of all collision spheres.
            The last ``N_FINGERS`` entries are the fingertip spheres.
        radii     : (N_total,) sphere radius per sphere in metres.
        """
        if hand_q.dim() == 1:
            hand_q = hand_q.unsqueeze(0)
        if X_WP.dim() == 2:
            X_WP = X_WP.unsqueeze(0)
        if hand_q.shape[-1] != self.HAND_DOF:
            raise ValueError(
                f"hand_q last-dim {hand_q.shape[-1]}, expected {self.HAND_DOF}"
            )

        B = hand_q.shape[0]
        device = hand_q.device
        dtype = hand_q.dtype
        X_PC = self._compute_X_PC(hand_q)

        n_link = self._n_link_spheres
        link_pos = torch.zeros(B, n_link, 3, device=device, dtype=dtype)
        tips_world = torch.zeros(B, self.N_FINGERS, 3, device=device, dtype=dtype)

        for fi in range(self.N_FINGERS):
            X = X_WP  # (B, 4, 4)
            for ji in range(4):
                X = X @ X_PC[:, fi * 4 + ji]
                # Place link spheres that belong to (fi, ji).
                mask = (self._link_sphere_finger == fi) & (self._link_sphere_depth == ji)
                idxs = mask.nonzero(as_tuple=False).squeeze(-1)
                for si in idxs:
                    off4 = torch.zeros(B, 4, device=device, dtype=dtype)
                    off4[:, :3] = self._link_sphere_offsets[si]
                    off4[:, 3] = 1.0
                    pos = (X @ off4.unsqueeze(-1)).squeeze(-1)
                    link_pos[:, si] = pos[:, :3]

            # Fingertip sphere
            offset = torch.zeros(B, 4, device=device, dtype=dtype)
            offset[:, :3] = self.fingertip_offsets[fi]
            offset[:, 3] = 1.0
            tip = (X @ offset.unsqueeze(-1)).squeeze(-1)
            tips_world[:, fi] = tip[:, :3]

        all_pos = torch.cat([link_pos, tips_world], dim=1)
        all_radii = torch.cat([
            self._link_sphere_radii.to(dtype),
            self.fingertip_radius.expand(self.N_FINGERS),
        ])
        return all_pos, all_radii

    @property
    def n_collision_spheres(self) -> int:
        """Total number of collision spheres (link + fingertip)."""
        return self._n_link_spheres + self.N_FINGERS

    def fingertip_radii(self) -> torch.Tensor:
        """(N_FINGERS,) fingertip sphere radius in metres."""
        return self.fingertip_radius.expand(self.N_FINGERS)
