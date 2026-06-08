#!/usr/bin/env python3
"""Fetch EquiDexFlow dataset tarballs listed in checkpoints/MANIFEST.yaml's
`datasets:` section, verify sha256, extract into data/dexgraspdb/v3/<hand>/.

    python scripts/download_assets.py dexgraspdb_v3_allegro
    python scripts/download_assets.py --all

Mirrors the checkpoints/download_checkpoints.py contract: drive_ids are
filled in by the maintainer after upload; until then this script reports
which ids are missing.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "checkpoints" / "MANIFEST.yaml"
DATA_DIR = REPO_ROOT / "data" / "dexgraspdb" / "v3"
_PLACEHOLDER = "REPLACE_WITH_GDRIVE_FILE_ID"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hand_from_key(key: str) -> str:
    # key like 'dexgraspdb_v3_allegro' -> 'allegro'
    return key.rsplit("_", 1)[-1]


def fetch_one(key: str, entry: dict) -> bool:
    hand = _hand_from_key(key)
    extracted_dir = DATA_DIR / hand
    tarball = DATA_DIR / f"{key}.tar.gz"
    want = entry.get("sha256")

    if tarball.is_file() and want and want != "REPLACE_AFTER_FREEZE" and sha256(tarball) == want:
        print(f"[{key}] tarball present and verified")
    else:
        drive_id = entry.get("drive_id")
        if not drive_id or drive_id == _PLACEHOLDER:
            print(f"[{key}] MISSING drive_id in MANIFEST.yaml — "
                  f"upload {tarball.name} and paste its id, or copy the tarball in manually.")
            return False
        try:
            import gdown
        except ImportError:
            sys.exit("gdown not installed.  pip install 'equidexflow[demo]'  (or pip install gdown)")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[{key}] downloading {drive_id} -> {tarball}")
        gdown.download(id=drive_id, output=str(tarball), quiet=False)
        if want and want != "REPLACE_AFTER_FREEZE":
            got = sha256(tarball)
            if got != want:
                sys.exit(f"[{key}] sha256 mismatch: got {got[:16]}… want {want[:16]}…")
            print(f"[{key}] sha256 verified")

    print(f"[{key}] extracting -> {extracted_dir}")
    extracted_dir.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(DATA_DIR)
    n = len(list(extracted_dir.glob("*.json")))
    print(f"[{key}] {n} object files now under {extracted_dir}")
    return True


def main():
    manifest = yaml.safe_load(open(MANIFEST))
    ds = manifest.get("datasets", {})
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", help="dataset keys (e.g. dexgraspdb_v3_allegro)")
    ap.add_argument("--all", action="store_true", help="fetch every dataset")
    ap.add_argument("--list", action="store_true", help="list available keys and exit")
    args = ap.parse_args()

    if args.list:
        for k, e in ds.items():
            n = e.get("n_objects", "?")
            sz = e.get("size_mb", "?")
            print(f"  {k:30s} {sz} MB  {n} objects")
        return

    keys = list(ds) if args.all else args.keys
    if not keys:
        ap.error("give one or more keys, or --all (use --list to see them)")
    ok = all(fetch_one(k, ds[k]) for k in keys if k in ds)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
