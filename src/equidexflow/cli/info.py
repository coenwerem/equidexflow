"""``equidexflow-info`` — smoke test. Prints version, torch/CUDA, and which
frozen checkpoint variants are present locally."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

import equidexflow

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CKPT_ROOT = _REPO_ROOT / "checkpoints"


def _have(mod: str) -> str:
    return "yes" if importlib.util.find_spec(mod) is not None else "no"


def _list_checkpoints() -> list[tuple[str, bool]]:
    out: list[tuple[str, bool]] = []
    if not _CKPT_ROOT.is_dir():
        return out
    for d in sorted(p for p in _CKPT_ROOT.iterdir() if p.is_dir()):
        out.append((d.name, (d / "checkpoint_best.pt").is_file()))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="equidexflow-info", description=__doc__)
    parser.parse_args(argv)

    print(f"equidexflow      : v{equidexflow.__version__}")
    print(f"torch            : {torch.__version__}")
    print(f"cuda available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device      : {torch.cuda.get_device_name(0)}")
    print()
    print("optional extras  :")
    for mod, extra in [
        ("trimesh", "data"),
        ("h5py", "data/train"),
        ("open3d", "viz/demo"),
        ("plotly", "viz"),
        ("matplotlib", "viz"),
        ("gdown", "demo"),
    ]:
        print(f"  {mod:12s} [{extra:10s}]  {_have(mod)}")

    print()
    print(f"checkpoints      : {_CKPT_ROOT}")
    ckpts = _list_checkpoints()
    if not ckpts:
        print("  (none — run `python checkpoints/download_checkpoints.py --all`)")
    else:
        for name, ready in ckpts:
            mark = "OK   " if ready else "miss "
            print(f"  [{mark}] {name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
