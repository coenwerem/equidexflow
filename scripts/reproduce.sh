#!/usr/bin/env bash
# Reproduce the paper's Allegro test-split numbers in one command.
#
#   ./scripts/reproduce.sh                 # CPU/GPU autodetect
#   ./scripts/reproduce.sh --device 0      # pin a GPU
#
# Steps:
#   1. fetch all 4 checkpoints from Drive (sha256-verified against MANIFEST.yaml)
#   2. fetch the 2 test-split grasp tarballs + YCB + EGAD meshes (sha256-verified)
#   3. run scripts/run_full_eval.py --pre-split on the 811-grasp test set
#
# Reproduces the model-side numbers in REPRODUCE.md. The Drake-side artifacts
# (paper-quality renders, shake validation, hardware) live in the FRoGGeR fork
# (see REPRODUCE.md > "Requires FRoGGeR / Drake").

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

echo "[reproduce] (1/3) downloading checkpoints..."
python checkpoints/download_checkpoints.py --all

echo "[reproduce] (2/3) downloading datasets and object meshes..."
python scripts/download_assets.py --all

echo "[reproduce] (3/3) running full eval on 811-grasp test split..."
python scripts/run_full_eval.py --hand allegro --pre-split "$@"

echo "[reproduce] done. Numbers should match REPRODUCE.md > Model-side artifacts."
