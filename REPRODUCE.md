# Reproduce

Model-side (no-Drake) reproduction. One environment:

```bash
pip install -e .[train,viz]
python checkpoints/download_checkpoints.py --all     # 5 .pt files into checkpoints/<key>/
python scripts/download_assets.py --all              # 2 test-split tarballs into data/dexgraspdb/v3/<hand>/
export EQUIDEXFLOW_OBJECTS_DIR=/path/to/objects      # object meshes (YCB + EGAD + GraspIt primitives)
```

The released datasets contain **only the 10% test split** (811 grasps per hand) used to
produce the paper's `tab:results`. Pass `--pre-split` to the eval scripts so they treat
the on-disk data as the test set directly instead of re-partitioning it (which would
otherwise split 811 grasps 80/10/10 and eval on ~82). Full 80/10/10 data is regenerable
via the FRoGGeR fork (`frogger.ablation_runner` + `export_dexgraspdb.py`).

## Model-side artifacts (reproduced in this repo)

| Artifact | Command |
|----------|---------|
| Grasp-quality table (4 variants, 81-obj test) | `python scripts/run_full_eval.py --hand allegro --device 0 --pre-split` |
| Per-metric contacts/forces/rollout | `python scripts/{eval_contacts,eval_forces,eval_rollout}.py …` |
| Equivariance (binned residuals) | `python scripts/compute_equivariance_binned.py --checkpoint allegro_full` |
| Diversity / coverage | `python scripts/compute_diversity.py …` |
| Inference ablations (cone-off, det vs stoch) | `python scripts/run_inference_ablations.py …` |
| Grasp galleries (JSON) | `python scripts/generate_figure_grasps.py --checkpoint allegro_full …` |
| Quick visual demo (posed hand render) | `python demo.py --mesh assets/objects/... --checkpoint allegro_full --viz` |

## Important: composite scores are stochastic

`model.sample()` draws from the wrist SE(3) flow's stochastic source (and, for a
flow joint decoder, the joint latent z). **The eval sets no seed**, so the
composite **Top-1 / Top-3** scores vary run-to-run by a non-trivial margin
(observed spread on the order of ~0.3–0.5 for the Allegro Full). The
**contact error is stable** (~0.039 m) across runs. Treat the published
point estimates as single draws; for a tight number, fix a seed and report a
mean ± interval over several runs. The paper's score-distribution figure already
reports BCa intervals and Holm-corrected Wilcoxon tests over the 81 per-object
values; the headline table value is one draw.

## Requires FRoGGeR / Drake (linked, not reproduced here)

These live in a modified fork of the [FRoGGeR repo](www.github.com/albertli/frogger) that we will release soon (`frogger/REPRO.md`) and need `pydrake`/`mujoco`:

- Paper-quality posed-hand renders driven by Drake FK (`render_paper_figures.py`).
- Physics / shake validation (`run_gagrasp_drake.py`, `frogger.validation`).
- Arm-side IK / reachability and LEAP arm-aware renders (`leap_qarm_and_render.py`).
- Dataset regeneration from FRoGGeR meta-JSONs (`export_dexgraspdb.py` +
  `frogger/equidex/dexgraspdb_adapter.py`).
- Hardware execution (paper supplementary).

The standalone `demo.py` renderer here reproduces the *look* of the posed-hand
galleries without Drake (kinematics manifest + Open3D); see
`src/equidexflow/render/`.
