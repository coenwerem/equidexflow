#!/usr/bin/env python3
"""Fetch EquiDexFlow checkpoints listed in MANIFEST.yaml from Google Drive.

    python checkpoints/download_checkpoints.py allegro_full
    python checkpoints/download_checkpoints.py --all

Each checkpoint is stored as  checkpoints/<key>/{checkpoint_best.pt, config.yml}.
If the file is already present and its sha256 matches the manifest, it is left
as-is. Drive ids are filled in by the maintainer after upload; until then this
script reports which ids are missing.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "MANIFEST.yaml"
_PLACEHOLDER = "REPLACE_WITH_GDRIVE_FILE_ID"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(key: str, entry: dict) -> bool:
    dest_dir = HERE / key
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / entry.get("file", "checkpoint_best.pt")
    want = entry.get("sha256")

    if dest.is_file() and want and want != "REPLACE_AFTER_FREEZE" and sha256(dest) == want:
        print(f"[{key}] present and verified - skipping")
        return True

    drive_id = entry.get("drive_id")
    if not drive_id or drive_id == _PLACEHOLDER:
        print(f"[{key}] MISSING drive_id in MANIFEST.yaml - "
              f"upload {dest} and paste its id, or copy the file in manually.")
        return False

    try:
        import gdown
    except ImportError:
        sys.exit("gdown not installed.  pip install 'equidexflow[demo]'  (or pip install gdown)")

    print(f"[{key}] downloading {drive_id} -> {dest}")
    gdown.download(id=drive_id, output=str(dest), quiet=False)
    if want and want != "REPLACE_AFTER_FREEZE":
        got = sha256(dest)
        if got != want:
            sys.exit(f"[{key}] sha256 mismatch: got {got[:16]}… want {want[:16]}…")
        print(f"[{key}] sha256 verified")
    return True


def main():
    manifest = yaml.safe_load(open(MANIFEST))
    cks = manifest["checkpoints"]
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", help="checkpoint keys (e.g. allegro_full)")
    ap.add_argument("--all", action="store_true", help="fetch every checkpoint")
    ap.add_argument("--list", action="store_true", help="list available keys and exit")
    args = ap.parse_args()

    if args.list:
        for k, e in cks.items():
            tag = "preprint" if e.get("reproduces_preprint_table") else "alt"
            print(f"  {k:24s} {e.get('size_mb','?')} MB  [{tag}]")
        return

    keys = list(cks) if args.all else args.keys
    if not keys:
        ap.error("give one or more keys, or --all (use --list to see them)")
    ok = all(fetch_one(k, cks[k]) for k in keys if k in cks)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
