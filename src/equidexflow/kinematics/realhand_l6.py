"""
Pure-PyTorch forward kinematics for the RealHand L6 (right hand).

Kinematic parameters are hard-coded from the MJCF:
  assets/end_effectors/realhand_l6_right/realhand_l6_right.xml

Body tree (all finger roots are direct children of the 'root' body):

  root  [wrist / hand_base_link - externally provided SE(3) pose]
  ├## thumb_metacarpals_base2  pos=[0.011508, 0.022975, 0.032794]
  │     joint: thumb_cmc_yaw   axis=[0, 0, -1]   range=[0.00, 1.54]
  │   └## thumb_metacarpals  pos=[0.0061649, 0.010678, -0.004891]
  │                          quat=[0.4192, -0.472029, -0.272524, -0.726079] (w,x,y,z)
  │         joint: thumb_cmc_pitch  axis=[0, 1, 0]  range=[0.00, 0.52]
  │       └## thumb_distal  pos=[0.0037776, 0, 0.045368]
  │             joint: thumb_ip  axis=[0, 1, 0]  range=[0.00, 0.96]
  │             site: thumb_tip_site  pos=[0, 0, 0.04]
  ├## index_proximal  pos=[0.0024758, 0.02419, 0.098779]
  │                   quat=[0.999657, -0.026177, 0, 0]
  │     joint: index_mcp_pitch  axis=[0, 1, 0]  range=[0.00, 1.57]
  │   └## index_distal  pos=[-0.0052516, 0, 0.036625]
  │         joint: index_dip  axis=[0, 1, 0]  range=[0.00, 1.40]
  │         site: index_tip_site  pos=[0, 0, 0.04]
  ├## middle_proximal  pos=[0.00052576, 0.00634, 0.1027]
  │     joint: middle_mcp_pitch  axis=[0, 1, 0]  range=[0.00, 1.57]
  │   └## middle_distal  pos=[-0.0052516, 0, 0.036625]
  │         joint: middle_dip  axis=[0, 1, 0]  range=[0.00, 1.40]
  │         site: middle_tip_site  pos=[0, 0, 0.04]
  ├## ring_proximal  pos=[0.0010258, -0.011135, 0.098767]
  │                  quat=[0.999657, 0.026177, 0, 0]
  │     joint: ring_mcp_pitch  axis=[0, 1, 0]  range=[0.00, 1.57]
  │   └## ring_distal  pos=[-0.0052516, 0, 0.036625]
  │         joint: ring_dip  axis=[0, 1, 0]  range=[0.00, 1.40]
  │         site: ring_tip_site  pos=[0, 0, 0.04]
  └## pinky_proximal  pos=[0.0024758, -0.028372, 0.092741]
                      quat=[0.999048, 0.0436192, 0, 0]
        joint: pinky_mcp_pitch  axis=[0, 1, 0]  range=[0.00, 1.57]
      └## pinky_distal  pos=[-0.0052516, 0, 0.036625]
            joint: pinky_dip  axis=[0, 1, 0]  range=[0.00, 1.40]
            site: pinky_tip_site  pos=[0, 0, 0.04]
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Numpy helpers for building constant SE(3) matrices at import time
# ---------------------------------------------------------------------------

def _quat_to_rot_np(q: list[float]) -> np.ndarray:
    """MuJoCo quaternion (w, x, y, z)  ->  3x3 rotation matrix (float64)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _make_T_np(pos: list[float], quat: list[float] | None = None) -> np.ndarray:
    """Build a (4, 4) homogeneous transform.  *quat* is (w, x, y, z)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = pos
    if quat is not None:
        T[:3, :3] = _quat_to_rot_np(quat)
    return T


# ---------------------------------------------------------------------------
# Hard-coded constant transforms (extracted from MJCF - do NOT parse at runtime)
# ---------------------------------------------------------------------------
# Convention: T_A_to_B  = static SE(3) from body A's origin to body B's origin,
#             expressed in body A's local frame.

# ## Thumb ##################################################################
# root  ->  thumb_metacarpals_base2
_T_root_to_tmb_base2 = _make_T_np(
    pos=[0.011508, 0.022975, 0.032794]
)  # no orientation change
# thumb_metacarpals_base2  ->  thumb_metacarpals  (after yaw joint)
_T_tmb_base2_to_meta = _make_T_np(
    pos=[0.0061649, 0.010678, -0.004891],
    quat=[0.4192, -0.472029, -0.272524, -0.726079],
)
# thumb_metacarpals  ->  thumb_distal  (after pitch joint)
_T_meta_to_distal = _make_T_np(
    pos=[0.0037776, 0.0, 0.045368]
)  # no orientation change
# thumb_distal  ->  fingertip site  (after ip joint)
_T_tmb_distal_to_tip = _make_T_np(pos=[0.0, 0.0, 0.04])

# ## Index ###################################################################
# root  ->  index_proximal
_T_root_to_idx_prox = _make_T_np(
    pos=[0.0024758, 0.02419, 0.098779],
    quat=[0.999657, -0.026177, 0.0, 0.0],
)
# index_proximal  ->  index_distal  (after mcp joint)
_T_idx_prox_to_dist = _make_T_np(pos=[-0.0052516, 0.0, 0.036625])
# index_distal  ->  fingertip site  (after dip joint)
_T_idx_dist_to_tip = _make_T_np(pos=[0.0, 0.0, 0.04])

# ## Middle ##################################################################
# root  ->  middle_proximal
_T_root_to_mid_prox = _make_T_np(pos=[0.00052576, 0.00634, 0.1027])
# middle_proximal  ->  middle_distal  (after mcp joint)
_T_mid_prox_to_dist = _make_T_np(pos=[-0.0052516, 0.0, 0.036625])
# middle_distal  ->  fingertip site  (after dip joint)
_T_mid_dist_to_tip = _make_T_np(pos=[0.0, 0.0, 0.04])

# ## Ring ####################################################################
# root  ->  ring_proximal
_T_root_to_ring_prox = _make_T_np(
    pos=[0.0010258, -0.011135, 0.098767],
    quat=[0.999657, 0.026177, 0.0, 0.0],
)
# ring_proximal  ->  ring_distal  (after mcp joint)
_T_ring_prox_to_dist = _make_T_np(pos=[-0.0052516, 0.0, 0.036625])
# ring_distal  ->  fingertip site  (after dip joint)
_T_ring_dist_to_tip = _make_T_np(pos=[0.0, 0.0, 0.04])

# ## Pinky ###################################################################
# root  ->  pinky_proximal
_T_root_to_pnk_prox = _make_T_np(
    pos=[0.0024758, -0.028372, 0.092741],
    quat=[0.999048, 0.0436192, 0.0, 0.0],
)
# pinky_proximal  ->  pinky_distal  (after mcp joint)
_T_pnk_prox_to_dist = _make_T_np(pos=[-0.0052516, 0.0, 0.036625])
# pinky_distal  ->  fingertip site  (after dip joint)
_T_pnk_dist_to_tip = _make_T_np(pos=[0.0, 0.0, 0.04])


# ---------------------------------------------------------------------------
# Pure-torch rotation utilities
# ---------------------------------------------------------------------------

def _rot_from_axis_angle(
    axis: torch.Tensor,   # (..., 3)  unit vector
    angle: torch.Tensor,  # (...,)    radians
) -> torch.Tensor:        # (..., 3, 3)
    """Rodrigues' rotation formula - differentiable."""
    c = torch.cos(angle)   # (...)
    s = torch.sin(angle)   # (...)
    t = 1.0 - c            # (...)
    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]
    # fmt: off
    R = torch.stack(
        [
            t*x*x + c,   t*x*y - s*z, t*x*z + s*y,
            t*x*y + s*z, t*y*y + c,   t*y*z - s*x,
            t*x*z - s*y, t*y*z + s*x, t*z*z + c,
        ],
        dim=-1,
    )  # (..., 9)
    # fmt: on
    return R.reshape(*angle.shape, 3, 3)


# ---------------------------------------------------------------------------
# RealHandL6FK
# ---------------------------------------------------------------------------

class RealHandL6FK(nn.Module):
    """Pure-torch forward kinematics for RealHand L6 (right hand).

    Joint ordering (11 DOF):
      [0]   thumb_cmc_yaw - thumb_metacarpals_base2, axis=[0, 0, -1]
      [1]   thumb_cmc_pitch - thumb_metacarpals,       axis=[0, 1,  0]
      [2]   thumb_ip - thumb_distal,             axis=[0, 1,  0]
      [3]   index_mcp_pitch - index_proximal,           axis=[0, 1,  0]
      [4]   index_dip - index_distal,             axis=[0, 1,  0]
      [5]   middle_mcp_pitch - middle_proximal,          axis=[0, 1,  0]
      [6]   middle_dip - middle_distal,            axis=[0, 1,  0]
      [7]   ring_mcp_pitch - ring_proximal,            axis=[0, 1,  0]
      [8]   ring_dip - ring_distal,              axis=[0, 1,  0]
      [9]   pinky_mcp_pitch - pinky_proximal,           axis=[0, 1,  0]
      [10]  pinky_dip - pinky_distal,             axis=[0, 1,  0]

    *wrist_pose* is the SE(3) pose of the 'root' / hand_base_link body
    in the world frame.  All fingertip poses are expressed in the world frame
    as  T_world = wrist_pose @ T_local.
    """

    # Joint axes (from MJCF)
    JOINT_AXES: np.ndarray = np.array(
        [
            [0.0, 0.0, -1.0],  # [0]  thumb_cmc_yaw
            [0.0, 1.0,  0.0],  # [1]  thumb_cmc_pitch
            [0.0, 1.0,  0.0],  # [2]  thumb_ip
            [0.0, 1.0,  0.0],  # [3]  index_mcp_pitch
            [0.0, 1.0,  0.0],  # [4]  index_dip
            [0.0, 1.0,  0.0],  # [5]  middle_mcp_pitch
            [0.0, 1.0,  0.0],  # [6]  middle_dip
            [0.0, 1.0,  0.0],  # [7]  ring_mcp_pitch
            [0.0, 1.0,  0.0],  # [8]  ring_dip
            [0.0, 1.0,  0.0],  # [9]  pinky_mcp_pitch
            [0.0, 1.0,  0.0],  # [10] pinky_dip
        ],
        dtype=np.float64,
    )

    # Joint limits [lo, hi] in radians (from MJCF range="lo hi")
    JOINT_RANGES: np.ndarray = np.array(
        [
            [0.00, 1.54],  # thumb_cmc_yaw
            [0.00, 0.52],  # thumb_cmc_pitch
            [0.00, 0.96],  # thumb_ip
            [0.00, 1.57],  # index_mcp_pitch
            [0.00, 1.40],  # index_dip
            [0.00, 1.57],  # middle_mcp_pitch
            [0.00, 1.40],  # middle_dip
            [0.00, 1.57],  # ring_mcp_pitch
            [0.00, 1.40],  # ring_dip
            [0.00, 1.57],  # pinky_mcp_pitch
            [0.00, 1.40],  # pinky_dip
        ],
        dtype=np.float64,
    )

    def __init__(self) -> None:
        super().__init__()
        # Register all constant SE(3) transforms as float32 buffers so that
        # .to(device) / .cuda() propagate them automatically.
        self._reg("T_root_to_tmb_base2",  _T_root_to_tmb_base2)
        self._reg("T_tmb_base2_to_meta",  _T_tmb_base2_to_meta)
        self._reg("T_meta_to_distal",     _T_meta_to_distal)
        self._reg("T_tmb_distal_to_tip",  _T_tmb_distal_to_tip)

        self._reg("T_root_to_idx_prox",   _T_root_to_idx_prox)
        self._reg("T_idx_prox_to_dist",   _T_idx_prox_to_dist)
        self._reg("T_idx_dist_to_tip",    _T_idx_dist_to_tip)

        self._reg("T_root_to_mid_prox",   _T_root_to_mid_prox)
        self._reg("T_mid_prox_to_dist",   _T_mid_prox_to_dist)
        self._reg("T_mid_dist_to_tip",    _T_mid_dist_to_tip)

        self._reg("T_root_to_ring_prox",  _T_root_to_ring_prox)
        self._reg("T_ring_prox_to_dist",  _T_ring_prox_to_dist)
        self._reg("T_ring_dist_to_tip",   _T_ring_dist_to_tip)

        self._reg("T_root_to_pnk_prox",   _T_root_to_pnk_prox)
        self._reg("T_pnk_prox_to_dist",   _T_pnk_prox_to_dist)
        self._reg("T_pnk_dist_to_tip",    _T_pnk_dist_to_tip)

    def _reg(self, name: str, T_np: np.ndarray) -> None:
        self.register_buffer(name, torch.tensor(T_np, dtype=torch.float32))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_batch(
        q: torch.Tensor,
        wrist_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Promote (11,) and (4,4) inputs to batched (B,11) / (B,4,4)."""
        if q.dim() == 1:
            q = q.unsqueeze(0)
        if wrist_pose.dim() == 2:
            wrist_pose = wrist_pose.unsqueeze(0)
        B = q.shape[0]
        assert wrist_pose.shape[0] == B or wrist_pose.shape[0] == 1, (
            f"Batch size mismatch: q {B} vs wrist_pose {wrist_pose.shape[0]}"
        )
        if wrist_pose.shape[0] == 1 and B > 1:
            wrist_pose = wrist_pose.expand(B, 4, 4)
        return q, wrist_pose, B

    def _const(
        self, buf: torch.Tensor, B: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Broadcast a (4,4) buffer to (B,4,4) with correct dtype/device."""
        return buf.to(dtype=dtype, device=device).unsqueeze(0).expand(B, 4, 4)

    def _joint_T(
        self, axis_np: np.ndarray, angle: torch.Tensor
    ) -> torch.Tensor:
        """Build (B, 4, 4) SE(3) for a revolute joint: pure rotation about *axis*."""
        B = angle.shape[0]
        dtype, device = angle.dtype, angle.device
        axis = (
            torch.tensor(axis_np, dtype=dtype, device=device)
            .unsqueeze(0)
            .expand(B, 3)
        )
        R = _rot_from_axis_angle(axis, angle)  # (B, 3, 3)
        T = torch.zeros(B, 4, 4, dtype=dtype, device=device)
        T[:, :3, :3] = R
        T[:, 3, 3] = 1.0
        return T

    # ------------------------------------------------------------------
    # Per-finger FK
    # ------------------------------------------------------------------

    def _thumb_tip(
        self, q: torch.Tensor, wp: torch.Tensor, B: int
    ) -> torch.Tensor:
        """(B, 4, 4) SE(3) of the thumb fingertip in world frame."""
        dt, dv = q.dtype, q.device
        c = lambda buf: self._const(buf, B, dt, dv)
        T = wp @ c(self.T_root_to_tmb_base2)
        T = T @ self._joint_T(self.JOINT_AXES[0], q[:, 0])   # thumb_cmc_yaw
        T = T @ c(self.T_tmb_base2_to_meta)
        T = T @ self._joint_T(self.JOINT_AXES[1], q[:, 1])   # thumb_cmc_pitch
        T = T @ c(self.T_meta_to_distal)
        T = T @ self._joint_T(self.JOINT_AXES[2], q[:, 2])   # thumb_ip
        T = T @ c(self.T_tmb_distal_to_tip)
        return T

    def _two_joint_finger_tip(
        self,
        q_prox: torch.Tensor,
        q_dist: torch.Tensor,
        j_prox: int,
        j_dist: int,
        T_root_to_prox: torch.Tensor,
        T_prox_to_dist: torch.Tensor,
        T_dist_to_tip: torch.Tensor,
        wp: torch.Tensor,
        B: int,
    ) -> torch.Tensor:
        """Generic 2-joint finger FK  ->  (B, 4, 4)."""
        dt, dv = q_prox.dtype, q_prox.device
        c = lambda buf: self._const(buf, B, dt, dv)
        T = wp @ c(T_root_to_prox)
        T = T @ self._joint_T(self.JOINT_AXES[j_prox], q_prox)
        T = T @ c(T_prox_to_dist)
        T = T @ self._joint_T(self.JOINT_AXES[j_dist], q_dist)
        T = T @ c(T_dist_to_tip)
        return T

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fingertip_poses(
        self,
        q: torch.Tensor,           # (B, 11) or (11,)
        wrist_pose: torch.Tensor,  # (B, 4, 4) or (4, 4)
    ) -> torch.Tensor:             # (B, 5, 4, 4)
        """5 fingertip SE(3) frames in world frame.

        Finger order: [thumb, index, middle, ring, pinky].
        """
        q, wp, B = self._ensure_batch(q, wrist_pose)

        T_thumb  = self._thumb_tip(q, wp, B)
        T_index  = self._two_joint_finger_tip(
            q[:, 3], q[:, 4], 3, 4,
            self.T_root_to_idx_prox, self.T_idx_prox_to_dist, self.T_idx_dist_to_tip,
            wp, B,
        )
        T_middle = self._two_joint_finger_tip(
            q[:, 5], q[:, 6], 5, 6,
            self.T_root_to_mid_prox, self.T_mid_prox_to_dist, self.T_mid_dist_to_tip,
            wp, B,
        )
        T_ring   = self._two_joint_finger_tip(
            q[:, 7], q[:, 8], 7, 8,
            self.T_root_to_ring_prox, self.T_ring_prox_to_dist, self.T_ring_dist_to_tip,
            wp, B,
        )
        T_pinky  = self._two_joint_finger_tip(
            q[:, 9], q[:, 10], 9, 10,
            self.T_root_to_pnk_prox, self.T_pnk_prox_to_dist, self.T_pnk_dist_to_tip,
            wp, B,
        )
        return torch.stack([T_thumb, T_index, T_middle, T_ring, T_pinky], dim=1)

    def fingertip_positions(
        self,
        q: torch.Tensor,           # (B, 11) or (11,)
        wrist_pose: torch.Tensor,  # (B, 4, 4) or (4, 4)
    ) -> torch.Tensor:             # (B, 5, 3)
        """Fingertip XYZ positions in world frame (faster than full poses)."""
        return self.fingertip_poses(q, wrist_pose)[..., :3, 3]

    def jacobian(
        self,
        q: torch.Tensor,           # (B, 11)
        wrist_pose: torch.Tensor,  # (B, 4, 4)
    ) -> torch.Tensor:             # (B, 5, 3, 11)
        """Positional Jacobian ∂p/∂q per fingertip, computed via autograd."""
        q, wp, B = self._ensure_batch(q, wrist_pose)
        results = []
        for b in range(B):
            q_b = q[b]          # (11,)
            wp_b = wp[b:b+1]    # (1, 4, 4)

            def fk_b(q_single: torch.Tensor) -> torch.Tensor:
                # q_single: (11,)  ->  (5, 3)
                return self.fingertip_positions(q_single.unsqueeze(0), wp_b).squeeze(0)

            # jac: (5, 3, 11)
            jac = torch.autograd.functional.jacobian(fk_b, q_b, vectorize=True)
            results.append(jac)
        return torch.stack(results, dim=0)  # (B, 5, 3, 11)

    def forward(
        self,
        q: torch.Tensor,
        wrist_pose: torch.Tensor,
    ) -> torch.Tensor:
        return self.fingertip_positions(q, wrist_pose)
