# Reproduce

Model-side (no-Drake) reproduction. One environment:

```bash
pip install -e .[train,viz]
python checkpoints/download_checkpoints.py --all     # 4 .pt files into checkpoints/<key>/
python scripts/download_assets.py --all              # 2 grasp tarballs + YCB + EGAD meshes
```

The `[train,viz]` extras are required on this path (eval imports `h5py`
transitively); `.[demo]` alone is not enough.

`download_assets.py --all` fetches everything the 81-object test split
needs: the two test-split grasp tarballs into `data/dexgraspdb/v3/<hand>/`,
the 28 YCB clean meshes referenced by the test split into
`assets/objects/frogger_ycb/`, and the 49-mesh EGAD eval set into
`~/.cache/equidexflow/egad/egad_eval_set/`. The four GraspIt primitives
ship in-tree under `assets/objects/graspit/`. No environment variables
are required for a stock reproduction. Set `EQUIDEXFLOW_STRICT_MESH=1`
if you want the loader to raise on a missing mesh instead of warning.

Mesh sources, redistributed under their upstream licenses:
[YCB](https://www.ycbbenchmarks.com/object-models/) (CC BY 4.0; watertight
"clean" variants produced by [FRoGGeR](https://github.com/alberthli/frogger),
MIT) · [EGAD](https://dougsm.github.io/egad/) (CC BY-NC 4.0).

The released dataset contains **only the 10% test split** (811 grasps per
hand) used to produce the paper's results table. Pass `--pre-split` to the
eval scripts so they treat the on-disk data as the test set directly
instead of re-partitioning it (which would otherwise split 811 grasps
80/10/10 and eval on ~82). The training/validation splits were produced
by an internal grasp-synthesis pipeline that is not part of this release.
The released checkpoints are the artifacts of training on that pipeline's
output.

## Model-side artifacts (reproduced in this repo)

| Artifact | Command |
|----------|---------|
| Grasp-quality table (4 variants, 81-obj test) | `python scripts/run_full_eval.py --hand allegro --device 0 --pre-split` |
| Per-metric contacts/forces/rollout | `python scripts/{eval_contacts,eval_forces,eval_rollout}.py …` |
| Equivariance (binned residuals) | `python scripts/compute_equivariance_binned.py --checkpoint allegro_full` |
| Diversity / coverage | `python scripts/compute_diversity.py …` |
| Inference ablations (cone-off, det vs stoch) | `python scripts/run_inference_ablations.py …` |
| Grasp galleries (JSON) | `python scripts/generate_figure_grasps.py --checkpoint allegro_full …` |
| Quick visual demo (posed hand render) | `equidexflow-demo --mesh assets/objects/graspit/sphere.stl --checkpoint allegro_full --viz` |

## Important: composite scores are stochastic

`model.sample()` draws from the wrist SE(3) flow's stochastic source (and,
for a flow joint decoder, the joint latent z). **The eval sets no seed**,
so the composite **Top-1 / Top-3** scores vary run-to-run by a non-trivial
margin (observed spread on the order of ~0.3–0.5 for the Allegro Full).
The **contact error is stable** (~0.039 m) across runs. Treat the
published point estimates as single draws. For a tight number, fix a seed
and report a mean ± interval over several runs. The paper's
score-distribution figure already reports BCa intervals and
Holm-corrected Wilcoxon tests over the 81 per-object values. The headline
table value is one draw.

## Paper artifacts not included in this release

The paper additionally reports physics-based validation and hardware
execution. The supporting code is platform- and simulator-specific and is
not part of this release:

- Drake-driven posed-hand renders used for paper figures.
- Physics / shake validation in Drake and MuJoCo.
- Arm-side IK and reachability for the physical LEAP Hand + 6-DoF arm.
- Hardware execution loop.

The standalone `equidexflow-demo` CLI (defined in
`src/equidexflow/cli/demo.py`) is a self-contained substitute for the
posed-hand visualization: collision-sphere forward kinematics from the
bundled kinematics manifest, a 2D preview via matplotlib (headless-safe),
and an interactive viewer via Open3D (`--viz`).
