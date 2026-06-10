"""High-level inference API: rebuild a trained model from its config + weights.

The checkpoint is only a ``state_dict`` - the architecture (deterministic vs
flow joint decoder, wrist frame, hand) lives in the run's config and MUST be
supplied to rebuild the network. ``load_checkpoint`` reads that config from a
sibling ``*.yml`` next to the checkpoint (or one passed explicitly) and warns
loudly on any missing/unexpected weights rather than silently random-initing a
mismatched head.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
from omegaconf import OmegaConf

from equidexflow.models import get_dex_model

# repo_root/checkpoints  (src/equidexflow/api.py -> parents[2] == repo root)
_CKPT_ROOT = Path(__file__).resolve().parents[2] / "checkpoints"


def _resolve_checkpoint(path_or_key) -> Path:
    """Accept a direct path OR a manifest variant key (checkpoints/<key>/checkpoint_best.pt)."""
    p = Path(path_or_key)
    if p.is_file():
        return p
    if p.is_dir() and (p / "checkpoint_best.pt").is_file():
        return p / "checkpoint_best.pt"
    cand = _CKPT_ROOT / str(path_or_key) / "checkpoint_best.pt"
    if cand.is_file():
        return cand
    raise FileNotFoundError(
        f"no checkpoint found at {p} nor at {cand}. "
        f"Pass a path to checkpoint_best.pt or a frozen variant key under {_CKPT_ROOT}."
    )


def _find_config(ckpt_path: Path):
    cfgs = sorted(list(ckpt_path.parent.glob("*.yml")) + list(ckpt_path.parent.glob("*.yaml")))
    if not cfgs:
        return None
    # prefer a file literally named config.*, else first alphabetical
    cfgs.sort(key=lambda q: (q.stem != "config", q.name))
    return cfgs[0]


def _extract_state(ckpt):
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            return ckpt["model_state"]
        if "model" in ckpt:
            return ckpt["model"]
    return ckpt


def load_checkpoint(path, config=None, device="cpu", strict=False):
    """Load a trained EquiDexFlow model.

    Parameters
    ----------
    path    : path to ``checkpoint_best.pt`` (or a directory / variant key holding one)
    config  : optional path to the run config yml; if None, a sibling *.yml is used
    device  : torch device string
    strict  : passed to ``load_state_dict``; default False but mismatches are warned

    Returns
    -------
    EquiDexFlow model in ``.eval()`` mode.
    """
    ckpt_path = _resolve_checkpoint(path)
    cfg_path = Path(config) if config is not None else _find_config(ckpt_path)
    cfg = OmegaConf.load(str(cfg_path)) if cfg_path is not None else OmegaConf.create({})
    m = cfg.get("model", OmegaConf.create({})) or OmegaConf.create({})

    model = get_dex_model(
        p_uncond=float(m.get("p_uncond", 0.1)),
        guidance=float(m.get("guidance", 2.0)),
        num_ode_steps=int(m.get("num_ode_steps", 10)),
        hand_q_decoder_type=str(m.get("hand_q_decoder", "deterministic")),
        n_coupling_layers=int(m.get("n_coupling_layers", 8)),
        surface_proj_tau=float(m.get("surface_proj_tau", 0.005)),
        wrist_frame=str(m.get("wrist_frame", "base")),
        hand=str(m.get("hand", "allegro")),
        cond_norm=bool(m.get("cond_norm", False)),
    ).to(device)

    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only kwarg
        ckpt = torch.load(str(ckpt_path), map_location="cpu")

    missing, unexpected = model.load_state_dict(_extract_state(ckpt), strict=strict)
    if missing or unexpected:
        warnings.warn(
            f"load_checkpoint: {len(missing)} missing / {len(unexpected)} unexpected weights "
            f"(arch/config mismatch - wrong decoder/hand/frame?). "
            f"missing[:3]={list(missing)[:3]} unexpected[:3]={list(unexpected)[:3]}",
            stacklevel=2,
        )
    return model.eval()
