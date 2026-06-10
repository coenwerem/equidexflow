# Checkpoints

Each checkpoint is a directory `checkpoints/<key>/` holding `checkpoint_best.pt`,
its `config.yml` (needed to rebuild the architecture), `sha256.txt`, and - for
the published variants - a `metrics.json`. Load by key:

```python
from equidexflow import load_checkpoint
model = load_checkpoint("allegro_full", device="cuda")   # or a direct path
grasps = model.sample(point_cloud, num_samples=10)
```

`load_checkpoint` reads the architecture knobs (joint-decoder type, wrist frame)
from the sibling `config.yml`, so the config **must** travel with the checkpoint.

## Allegro (v1)

The four `allegro_*` checkpoints below are exactly the runs that produced the
preprint's `tab:results` - verified against
`results_table_81obj_xhand.yaml` (`source_run: *_xhand/20260526`). Their
`expected_metrics` in `MANIFEST.yaml` match the paper to 4 decimals.

| key | role | decoder | wrench↓ | Top-1↑ | size |
|-----|------|---------|:------:|:------:|:----:|
| `allegro_full` | **Full (paper)** | deterministic | 0.46 | −0.96 | 6.5 MB |
| `allegro_geom_only` | GeomOnly ablation | deterministic | 1.58 | −3.29 | 6.5 MB |
| `allegro_pose_only` | PoseOnly ablation | deterministic | 1.29 | −2.52 | 6.5 MB |
| `allegro_contact_only` | ContactOnly ablation | deterministic | 1.36 | −2.57 | 6.5 MB |
| `allegro_full_flow_alt` | alt: Real-NVP joint decoder, grasp_center | flow | - | - | 22 MB |

**Note on the joint decoder.** `allegro_full` (the run behind the published
numbers) uses a *deterministic* joint decoder with the `base` wrist frame. The
preprint text describes a *conditional normalizing flow* joint decoder;
`allegro_full_flow_alt` is that architecture, staged here but **not** the source
of the published table. Pick one consistently before the public release.

## LEAP

LEAP checkpoints come from machine B - see `docs/MACHINE_B_HANDOFF.md`. Once the
tarball lands they are frozen here as `leap_full` (+ ablations) and added to
`MANIFEST.yaml`.

## Distribution

For the public release the large `*.pt` files (and the dataset) live on Google
Drive, not in git (`.gitignore` excludes them). After uploading, paste each
file's Drive id into `MANIFEST.yaml`; then `download_checkpoints.py <key>`
fetches and sha256-verifies on demand. The `config.yml`, `sha256.txt`,
`metrics.json`, and `MANIFEST.yaml` stay in git.
