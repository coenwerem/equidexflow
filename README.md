<div align="center">

# EquiDexFlow

**SE(3)-equivariant 6-DoF dexterous grasp generative flows**

[Project page](https://equidexflow.github.io) &nbsp;·&nbsp;
[Paper (arXiv, coming soon)](#) &nbsp;·&nbsp;
[Data](#pretrained-checkpoints--datasets) &nbsp;·&nbsp;
License: MIT &nbsp;·&nbsp;
Python ≥ 3.10 &nbsp;·&nbsp;
PyTorch ≥ 2.0

<img src="assets/teaser/allegro_gallery_2x8.png" width="100%" alt="Allegro grasp gallery — sixteen EquiDexFlow grasps on YCB / EGAD / GraspIt primitives." />

</div>

EquiDexFlow takes an object point cloud and produces, in a single forward
pass: a wrist SE(3) pose, $n_h$ joint angles from a conditional normalizing
flow, a set of $n_c$ contact points projected onto the object surface, and
per-contact forces projected into the friction cone — all jointly consistent
with the learned distribution. The released Allegro checkpoints use
$n_h{=}16$ and $n_c{=}5$. Both are set per-hand in the model config.

---

## Quickstart

Three lines from a fresh checkout to a posed-hand preview PNG:

```bash
pip install -e ".[demo]"
python checkpoints/download_checkpoints.py allegro_full
equidexflow-demo --mesh assets/objects/graspit/sphere.stl --checkpoint allegro_full
# -> out/demo/preview.png  +  out/demo/grasp_{00..07}.npz
```

Or use the model from Python (pure inference, no extras required):

```python
import torch, trimesh
from equidexflow import load_checkpoint

mesh = trimesh.load("assets/objects/graspit/sphere.stl", force="mesh")
pts, _ = trimesh.sample.sample_surface(mesh, 512)
pc    = torch.from_numpy(pts.T).float().cuda()           # (3, N)

model  = load_checkpoint("allegro_full", device="cuda")
grasps = model.sample(pc, num_samples=10)                # list[dict] of length 10

g = grasps[0]
g["wrist_pose"]    # (4, 4)  SE(3) wrist pose
g["hand_q"]        # (16,)   joint angles
g["contacts"]      # (5, 3)  surface-projected fingertip contacts
g["forces"]        # (5, 3)  friction-cone-projected contact forces
g["contact_logits"]# (5,)    per-finger confidence
```

## Installation

Tested on Linux, Python 3.10–3.12, PyTorch 2.0+, CUDA 11.8+ (CPU also
works for inference).

```bash
pip install -e .              # pure inference (torch, numpy, scipy, omegaconf, roma)
pip install -e ".[demo]"      # + trimesh / open3d / matplotlib / gdown  (recommended)
pip install -e ".[all]"       # + training / dataset loaders / plotting

equidexflow-info              # smoke test: print version, CUDA, present checkpoints
```

CUDA is auto-detected. CPU works for inference but is slow for the ODE
solver at full sample counts. The extras (`data`, `train`, `viz`, `demo`)
are defined in `pyproject.toml`.

## Pretrained checkpoints & datasets

Both are released on Google Drive and pinned by sha256 in
[`checkpoints/MANIFEST.yaml`](checkpoints/MANIFEST.yaml). A file already on
disk with the correct hash is left untouched on re-download.

```bash
# 5 model variants (allegro_full + 3 ablations + 1 alt flow), ~5 .pt files
python checkpoints/download_checkpoints.py --all

# 2 test-split tarballs (811 grasps per hand) -> data/dexgraspdb/v3/<hand>/
python scripts/download_assets.py --all
```

The released dataset is the **10% test split** (811 grasps per hand) used
to produce the paper's results table. The remaining 90% (training and
validation) was generated with an internal grasp-synthesis pipeline that
is not part of this release. The published checkpoints are the artifacts
of that training run.

A few example object meshes are bundled under `assets/objects/` — enough
to drive the demo CLI and the Python quickstart. Running the full
evaluation against the released test split additionally requires the
corresponding object meshes (YCB, EGAD, and GraspIt primitives) on disk.
Point `EQUIDEXFLOW_OBJECTS_DIR` at a directory holding those source
meshes:

```bash
export EQUIDEXFLOW_OBJECTS_DIR=/path/to/objects
```

## Demo and visualization

`equidexflow-demo` is the front door: mesh in, grasps + preview out.

```bash
# Default: 8 grasps, headless 2-pane preview PNG
equidexflow-demo --mesh assets/objects/frogger_ycb/006_mustard_bottle.obj \
                 --checkpoint allegro_full --num-samples 8 --out out/mustard

# Interactive viewer (Open3D): object mesh + hand collision spheres + contacts
equidexflow-demo --mesh assets/objects/graspit/cylinder.stl --viz
```

Each run writes one `preview.png` plus a `grasp_NN.npz` per sample containing
the wrist pose, joint angles, contacts, forces, contact logits, and the
forward-kinematics-evaluated hand sphere positions. Decoding is one line:

```python
import numpy as np
g = np.load("out/demo/grasp_00.npz")
g.files  # ['wrist_pose', 'hand_q', 'contacts', 'forces', 'contact_logits',
         #  'hand_sphere_xyz', 'hand_sphere_radii']
```

## Hardware results

Allegro grasps produced by this codebase were retargeted via inverse
kinematics to a physical [LEAP Hand](https://leaphand.com/) mounted on a
6-DoF [FAIR Innovation FR3 cobot](https://www.frtech.fr/FR/5.html), and
executed on two objects (a box primitive and the YCB potted-meat can) at
two object rotations (0°, 120°):

<p align="center">
  <img src="assets/teaser/hardware_2x2.gif" width="75%" alt="2x2 hardware execution panel: box primitive and potted-meat can, each at 0 and 120 deg, on a LEAP Hand." />
  <br/>
  <sub><b>Top row:</b> box primitive at 0° / 120°. &nbsp; <b>Bottom row:</b> potted-meat can at 0° / 120°.</sub>
</p>

Higher-resolution MP4 versions are on the
[project page](https://equidexflow.github.io). The retargeting and
controller stack used to drive the physical hand is platform-specific and
is not part of this release. This repository ships the grasp-generation
model whose outputs were executed in those clips.

## Reproducing the paper's table

This release reproduces the **model-side** numbers in the paper: the
grasp-quality table over the four ablations on the 81-object test split,
the per-metric contact / force / rollout / equivariance / diversity
breakdowns, and the inference-time ablations.

One command after `download_checkpoints` + `download_assets`:

```bash
./scripts/reproduce.sh                 # CPU/GPU autodetect
./scripts/reproduce.sh --device 0      # pin a GPU
```

For per-metric breakdowns and individual evaluation commands, see
[**REPRODUCE.md**](REPRODUCE.md). Caveat: `model.sample()` is stochastic
and the eval sets no seed by default. REPRODUCE.md documents the expected
spread on composite scores.

The paper additionally reports physics-based validation (Drake/MuJoCo
shake tests) and hardware execution. Those analyses use external
simulators and a platform-specific controller stack that are out of scope
for this release. The released checkpoints are the same ones that
produced those results.

## Repo Layout

```
equidexflow/
├── src/equidexflow/        # model + API (pure torch/numpy/scipy)
│   ├── api.py              # load_checkpoint(...)
│   ├── models/             # equi_dex_flow + VN-DGCNN + decoders
│   ├── kinematics/         # Allegro / LEAP / RealHand-L6 FK (differentiable)
│   ├── losses/  trainers/  metrics/  loaders/  physics/
│   └── cli/                # equidexflow-demo, equidexflow-info
├── scripts/                # train.py, run_full_eval.py, eval_*, plot/, reproduce.sh
├── checkpoints/            # MANIFEST.yaml and downloader. <variant>/{best.pt, config.yml}
├── data/dexgraspdb/v3/     # downloaded test-split tarballs will be saved here
├── assets/                 # hand URDFs + mesh primitives + teaser images
└── tests/                  # pytest
```

## Citation

```bibtex
@article{enwerem2026equidexflow,
  author  = {Enwerem, Clinton and Baras, John S. and Belta, Calin},
  title   = {{EquiDexFlow}: Contact-Grounded {SE(3)}-Equivariant Dexterous Grasp Generative Flows},
  journal = {arXiv},
  year    = {2026},
}
```

## Acknowledgments

This codebase is a dexterous extension of
[**EquiGraspFlow**](https://github.com/bdlim99/EquiGraspFlow) (Lim et al.,
CoRL 2024), used under the MIT License. The SE(3)-equivariant
flow-matching backbone, VN-DGCNN encoder, Lie-group utilities, ODE solvers,
and SE(3) base distributions originate upstream. See [`NOTICE`](NOTICE) for
a per-file breakdown. The encoders themselves build on Vector Neurons
(Deng et al., 2021) and DGCNN (Wang et al., 2019). The ground-truth grasps
used to train the released checkpoints were synthesized with FRoGGeR
(Li et al., 2023). Hardware results in the paper were executed on the
LEAP Hand (Shaw et al., 2023). We thank the maintainers of PyTorch,
Open3D, trimesh, and Drake.
