"""EquiDexFlow: SE(3)-equivariant 6-DoF dexterous grasp generative flows.

Public API
----------
    from equidexflow import load_checkpoint
    model   = load_checkpoint("checkpoints/allegro_full/checkpoint_best.pt", device="cuda")
    grasps  = model.sample(point_cloud, num_samples=10)   # (3,N) or (B,3,N) tensor

The model package is pure torch/numpy/scipy/omegaconf/roma. ``loaders`` and
``trainers`` (which pull in ``trimesh``) are intentionally NOT imported here so
that pure inference works without the data-prep extras.
"""

from equidexflow.models import get_dex_model
from equidexflow.models.equi_dex_flow import EquiDexFlow as EquiDexFlowModel
from equidexflow.api import load_checkpoint

__all__ = ["EquiDexFlowModel", "get_dex_model", "load_checkpoint"]
__version__ = "0.1.0"
