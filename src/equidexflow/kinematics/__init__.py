"""
Kinematics module for the LEAP hand (right).

Exports
-------
LeapFK
    Pure-PyTorch forward kinematics (16 DOF + SE(3) wrist pose).
compute_grasp_map
    Build the 6x3M grasp map G from contact positions.
friction_cone_penalty
    Differentiable friction-cone constraint penalty.
collision_penalty
    Soft fingertip-to-object proximity penalty.
"""

from equidexflow.kinematics.leap import LeapFK
from equidexflow.kinematics.grasp_map import compute_grasp_map, wrench_balance_residual
from equidexflow.kinematics.friction_cone import friction_cone_penalty, friction_cone_violation_rate
from equidexflow.kinematics.collision import collision_penalty, self_collision_penalty

__all__ = [
    "LeapFK",
    "compute_grasp_map",
    "wrench_balance_residual",
    "friction_cone_penalty",
    "friction_cone_violation_rate",
    "collision_penalty",
    "self_collision_penalty",
]
