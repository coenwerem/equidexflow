"""CPU smoke tests - no dataset, no GPU. Run: pytest tests/test_smoke.py"""
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


def test_link_frames_match_fingertips():
    """forward_link_frames returns one frame per hand body, and each distal
    frame reproduces the fingertip from forward() (same chain)."""
    from equidexflow.kinematics.allegro_fk import AllegroRightHandFK

    fk = AllegroRightHandFK()
    q = torch.zeros(16)
    X = torch.eye(4)
    frames = fk.forward_link_frames(q, X)
    assert fk.root_link_name in frames
    assert len(frames) == fk.HAND_DOF + 1          # 16 child links + root
    tips = fk.forward(q, X)[0]                      # (4, 3)
    for fi in range(fk.N_FINGERS):
        distal = frames[fk.link_names[fi * 4 + 3]][0]   # (4, 4)
        off = torch.cat([fk.fingertip_offsets[fi], torch.ones(1)])
        tip = (distal @ off)[:3]
        assert torch.allclose(tip, tips[fi], atol=1e-5)


def test_sdf_visuals_resolve():
    """Every Allegro SDF visual mesh resolves to a file on disk, and each
    visual's link is in the FK chain (so the renderer can place it)."""
    import pytest

    from equidexflow.kinematics.allegro_fk import AllegroRightHandFK
    from equidexflow.render.allegro_assets import load_allegro_visuals

    try:
        vis = load_allegro_visuals()
    except FileNotFoundError:
        pytest.skip("Allegro SDF/assets not present")
    assert vis, "no visual meshes parsed from the SDF"
    assert all(v.mesh_path.exists() for v in vis)
    frames = AllegroRightHandFK().forward_link_frames(torch.zeros(16), torch.eye(4))
    assert all(v.link_name in frames for v in vis)


def test_seating_reduces_contact_gap():
    """sample_seated brings the FK fingertips substantially closer to the
    predicted contacts than the raw decoder output."""
    from equidexflow.kinematics.allegro_fk import AllegroRightHandFK

    fk = AllegroRightHandFK()
    m = get_dex_model(num_ode_steps=2).eval()
    pc = torch.randn(1, 3, 256)
    torch.manual_seed(0)
    out = m.sample_seated(pc, num_samples=2, n_steps=120, return_raw=True)

    def gap(hq, wp, c):
        t = fk.forward(hq.unsqueeze(0), wp.unsqueeze(0))[0]
        return (t - c).norm(dim=-1).mean()

    for g in out:
        raw = gap(g["hand_q_raw"], g["wrist_pose_raw"], g["contacts"])
        seated = gap(g["hand_q"], g["wrist_pose"], g["contacts"])
        assert seated < raw
        assert torch.isfinite(g["hand_q"]).all()


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
