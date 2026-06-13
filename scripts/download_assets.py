#!/usr/bin/env python3
"""Fetch EquiDexFlow asset tarballs listed in checkpoints/MANIFEST.yaml.

    python scripts/download_assets.py dexgraspdb_v3_allegro   # one dataset key
    python scripts/download_assets.py objects_ycb             # one object-mesh key
    python scripts/download_assets.py --all                   # everything

Two asset families are managed here, both keyed off MANIFEST.yaml:
  - `datasets:` entries -> test-split grasp tarballs extracted to
    data/dexgraspdb/v3/<hand>/.
  - `objects:` entries  -> object-mesh tarballs (YCB, EGAD) extracted to
    `extract_to` (resolved relative to the repo root, or against $HOME
    for paths starting with `~`).

drive_ids are placeholders until the maintainer uploads each tarball.
The script reports which ids are missing so they can be filled in
without a code change.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "checkpoints" / "MANIFEST.yaml"
DATA_DIR = REPO_ROOT / "data" / "dexgraspdb" / "v3"
OBJECTS_CACHE = REPO_ROOT / "build" / "_object_tarballs"
_PLACEHOLDER = "REPLACE_WITH_GDRIVE_FILE_ID"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_extract_to(raw: str) -> Path:
    p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _fetch_verified(key: str, entry: dict, tarball: Path) -> bool:
    """Download `tarball` (if missing/stale) and verify sha256. Returns False
    when the manifest still has a placeholder drive_id."""
    want = entry.get("sha256")
    if tarball.is_file() and want and want != "REPLACE_AFTER_FREEZE" and sha256(tarball) == want:
        print(f"[{key}] tarball present and verified")
        return True

    drive_id = entry.get("drive_id")
    if not drive_id or drive_id == _PLACEHOLDER:
        print(f"[{key}] MISSING drive_id in MANIFEST.yaml - "
              f"upload {tarball.name} and paste its id, or copy the tarball in manually.")
        return False
    try:
        import gdown
    except ImportError:
        sys.exit("gdown not installed.  pip install 'equidexflow[demo]'  (or pip install gdown)")
    tarball.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{key}] downloading {drive_id} -> {tarball}")
    gdown.download(id=drive_id, output=str(tarball), quiet=False)
    if want and want != "REPLACE_AFTER_FREEZE":
        got = sha256(tarball)
        if got != want:
            sys.exit(f"[{key}] sha256 mismatch: got {got[:16]}… want {want[:16]}…")
        print(f"[{key}] sha256 verified")
    return True


def fetch_dataset(key: str, entry: dict) -> bool:
    hand = key.rsplit("_", 1)[-1]
    extracted_dir = DATA_DIR / hand
    tarball = DATA_DIR / f"{key}.tar.gz"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not _fetch_verified(key, entry, tarball):
        return False

    print(f"[{key}] extracting -> {extracted_dir}")
    extracted_dir.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(DATA_DIR)
    n = len(list(extracted_dir.glob("*.json")))
    print(f"[{key}] {n} object files now under {extracted_dir}")
    return True


def fetch_objects(key: str, entry: dict) -> bool:
    extract_to = _resolve_extract_to(entry["extract_to"])
    tarball = OBJECTS_CACHE / f"{key}.tar.gz"
    OBJECTS_CACHE.mkdir(parents=True, exist_ok=True)
    if not _fetch_verified(key, entry, tarball):
        return False

    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"[{key}] extracting -> {extract_to}")
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(extract_to)
    print(f"[{key}] {entry.get('n_objects', '?')} meshes now under {extract_to}")
    return True


def main():
    manifest = yaml.safe_load(open(MANIFEST))
    ds = manifest.get("datasets", {})
    obj = manifest.get("objects", {})
    all_entries = {**{k: ("dataset", v) for k, v in ds.items()},
                   **{k: ("objects", v) for k, v in obj.items()}}

    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", help="asset keys (datasets or objects)")
    ap.add_argument("--all", action="store_true", help="fetch every asset")
    ap.add_argument("--list", action="store_true", help="list available keys and exit")
    args = ap.parse_args()

    if args.list:
        for k, (kind, e) in all_entries.items():
            n = e.get("n_objects", "?")
            sz = e.get("size_mb", "?")
            print(f"  {k:30s} {kind:8s} {sz} MB  {n} objects")
        return

    keys = list(all_entries) if args.all else args.keys
    if not keys:
        ap.error("give one or more keys, or --all (use --list to see them)")

    ok = True
    for k in keys:
        if k not in all_entries:
            print(f"[{k}] unknown key; skipping")
            ok = False
            continue
        kind, entry = all_entries[k]
        if kind == "dataset":
            ok = fetch_dataset(k, entry) and ok
        else:
            ok = fetch_objects(k, entry) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
