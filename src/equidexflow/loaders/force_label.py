"""
Quasistatic force-closure force labeling for dexterous grasps.

Given N contact points and inward normals, solves for a minimal-norm set of
contact forces that satisfies:
  1. Wrench balance:  G(C) @ f + w_ext = 0,  w_ext = [0,0,-mg, 0,0,0]
  2. Friction cone:   f_i = α_i n_i + τ_i,   |τ_i| ≤ μ α_i,  α_i ≥ f_min

The 6x(3N) grasp map G is assembled as:
    G[:, 3i:3i+3] = [[I_3],  [skew(p_i)]]     (stacked 6x3 block for contact i)

so the wrench contribution from contact i is  G_i @ f_i = [f_i;  p_i x f_i].
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _skew(v: np.ndarray) -> np.ndarray:
    """
    Return the 3x3 skew-symmetric cross-product matrix for vector v.

    Parameters
    ----------
    v : (3,)

    Returns
    -------
    S : (3, 3)  such that  S @ w = v x w
    """
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ], dtype=float)


def _build_grasp_map(contact_points: np.ndarray) -> np.ndarray:
    """
    Build the 6x(3N) grasp map G from N contact positions.

    Parameters
    ----------
    contact_points : (N, 3)  positions in metres

    Returns
    -------
    G : (6, 3N)
    """
    N = len(contact_points)
    G = np.zeros((6, 3 * N), dtype=float)
    for i, p in enumerate(contact_points):
        G[:3, 3 * i : 3 * i + 3] = np.eye(3)
        G[3:, 3 * i : 3 * i + 3] = _skew(p)
    return G


def _pseudoinverse_forces(G: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Minimum-norm unconstrained solution  f = G^+ b  via least squares.

    Parameters
    ----------
    G : (6, 3N)
    b : (6,)

    Returns
    -------
    f : (3N,)
    """
    f, _, _, _ = np.linalg.lstsq(G, b, rcond=None)
    return f


def _normal_scale_forces(
    contact_normals: np.ndarray,
    G: np.ndarray,
    b: np.ndarray,
    f_min: float,
) -> np.ndarray:
    """
    Fallback: distribute forces purely along contact normals.

    Solves  G_n @ α = b  where  G_n[:, i] = G[:, 3i:3i+3] @ n_i,
    then clamps α ≥ f_min and returns  f_i = α_i * n_i.

    Parameters
    ----------
    contact_normals : (N, 3)  unit normals
    G               : (6, 3N)
    b               : (6,)

    Returns
    -------
    forces : (3N,)
    """
    N = len(contact_normals)
    norms = np.linalg.norm(contact_normals, axis=1, keepdims=True)
    n_unit = contact_normals / np.where(norms > 1e-8, norms, 1.0)

    G_n = np.zeros((6, N), dtype=float)
    for i in range(N):
        G_n[:, i] = G[:, 3 * i : 3 * i + 3] @ n_unit[i]

    alpha, _, _, _ = np.linalg.lstsq(G_n, b, rcond=None)
    alpha = np.maximum(alpha, f_min)
    return (alpha[:, None] * n_unit).ravel()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_contact_forces(
    contact_points: np.ndarray,
    contact_normals: np.ndarray,
    object_mass: float = 0.2,
    mu: float = 0.5,
    g: float = 9.81,
) -> np.ndarray:
    """
    Compute per-contact forces satisfying quasistatic equilibrium.

    Solves the quadratic program::

        min  ½ ‖f‖²
        s.t. G f = b                                (wrench balance)
             f_i · n_i ≥ f_min  ∀i               (compressive normal)
             ‖f_i - (f_i·n_i) n_i‖ ≤ μ (f_i·n_i)  (friction cone)

    where  b = [0, 0, m g, 0, 0, 0]ᵀ  and  G  is the 6x3N grasp map.
    Falls back to a pure-normal least-squares solution if the optimizer fails.

    Parameters
    ----------
    contact_points  : (N, 3)  contact positions in **metres**, object frame
    contact_normals : (N, 3)  unit normals pointing **into** the object
    object_mass     : object mass [kg]
    mu              : Coulomb friction coefficient
    g               : gravitational acceleration [m s⁻²]

    Returns
    -------
    forces : (N, 3)  per-contact forces in Newtons
    """
    N = len(contact_points)
    if N == 0:
        return np.zeros((0, 3), dtype=float)

    # Target wrench: contact forces must balance gravity (upward resultant)
    b = np.array([0.0, 0.0, object_mass * g, 0.0, 0.0, 0.0])
    G = _build_grasp_map(contact_points)          # (6, 3N)

    f_min = 1e-3   # minimum normal force [N] to avoid numerical issues

    # ------------------------------------------------------------------
    # Build constraints for scipy.optimize.minimize (SLSQP)
    # ------------------------------------------------------------------
    constraints: list[dict] = [
        {   # wrench balance (equality)
            "type": "eq",
            "fun": lambda f: G @ f - b,
            "jac": lambda f: G,
        }
    ]

    for i in range(N):
        n_i = contact_normals[i].copy()

        def _normal_ineq(f: np.ndarray, i: int = i, n: np.ndarray = n_i) -> float:
            """f_i · n_i ≥ f_min  (compressive contact)."""
            return float(np.dot(f[3 * i : 3 * i + 3], n) - f_min)

        def _friction_ineq(f: np.ndarray, i: int = i, n: np.ndarray = n_i) -> float:
            """μ (f_i · n_i) - ‖tangential component‖ ≥ 0  (inside friction cone)."""
            fi = f[3 * i : 3 * i + 3]
            fn = float(np.dot(fi, n))
            ft = fi - fn * n
            return float(mu * fn - np.linalg.norm(ft))

        constraints.append({"type": "ineq", "fun": _normal_ineq})
        constraints.append({"type": "ineq", "fun": _friction_ineq})

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    f0 = _pseudoinverse_forces(G, b)   # warm start

    result = minimize(
        fun=lambda f: 0.5 * float(np.dot(f, f)),
        x0=f0,
        jac=lambda f: f,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-7, "maxiter": 500, "disp": False},
    )

    # Accept result if converged OR wrench residual is small enough
    wrench_residual = np.linalg.norm(G @ result.x - b)
    if result.success or wrench_residual < 0.05 * object_mass * g:
        return result.x.reshape(N, 3)

    # Fallback: distribute forces purely along normals
    f_fallback = _normal_scale_forces(contact_normals, G, b, f_min)
    return f_fallback.reshape(N, 3)


def assign_finger_ids(
    contact_points: np.ndarray,
    fingertip_positions: np.ndarray,
) -> np.ndarray:
    """
    Assign each contact to the nearest fingertip by Euclidean distance.

    Parameters
    ----------
    contact_points    : (N, 3)  contact positions (any consistent unit/frame)
    fingertip_positions : (5, 3)  one position per finger,
                          ordered 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky

    Returns
    -------
    ids : (N,)  int64 array with values in {0, 1, 2, 3, 4}
    """
    if len(contact_points) == 0:
        return np.empty(0, dtype=np.int64)

    # (N, 5) pairwise squared distances
    diff = contact_points[:, None, :] - fingertip_positions[None, :, :]   # (N,5,3)
    dist2 = (diff ** 2).sum(axis=2)                                        # (N, 5)
    return dist2.argmin(axis=1).astype(np.int64)
