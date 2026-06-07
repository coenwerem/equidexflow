"""CPU smoke tests — no dataset, no GPU. Run: pytest tests/test_smoke.py"""
import glob

import torch

from equidexflow import get_dex_model


def test_build_and_sample():
    """Random-init build + sample produces the documented Allegro shapes."""
    m = get_dex_model(num_ode_steps=2).eval()
    out = m.sample(torch.randn(1, 3, 256), num_samples=3)
    assert len(out) == 3
    g = out[0]
    assert tuple(g["wrist_pose"].shape) == (4, 4)
    assert tuple(g["hand_q"].shape) == (16,)          # HAND_DOF=16
    assert g["contacts"].shape[0] == 4 and g["contacts"].shape[-1] == 3   # N_FINGERS=4
    assert g["forces"].shape == g["contacts"].shape
    for k in ("wrist_pose", "hand_q", "contacts", "forces"):
        assert torch.isfinite(g[k]).all()


def test_load_checkpoint_roundtrip():
    """If a frozen checkpoint is present, it loads clean and samples."""
    import pytest

    cps = glob.glob("checkpoints/**/checkpoint_best.pt", recursive=True)
    if not cps:
        pytest.skip("no checkpoint present")
    from equidexflow import load_checkpoint

    m = load_checkpoint(sorted(cps)[0], device="cpu")
    out = m.sample(torch.randn(3, 256), num_samples=2)
    assert len(out) == 2
