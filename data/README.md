# Data Format — the EquiDexFlow grasp schema

EquiDexFlow does not depend on FRoGGeR. It consumes a documented JSON grasp
schema; FRoGGeR is just the default generator. Any synthesis backbone (BODex,
DexGraspNet, your own optimizer) can emit compatible files and train the model.

## Layout

```
$EQUIDEXFLOW_DATA_DIR/dexgraspdb/v3/<hand>/<object_name>.json   # one file per object
$EQUIDEXFLOW_OBJECTS_DIR/<stem>.obj|.stl                        # object meshes (point clouds sampled at load)
```

Point the loader at these with environment variables (see
`src/equidexflow/configs/paths.example.yaml`):

```bash
export EQUIDEXFLOW_DATA_DIR=/path/to/datasets        # holds dexgraspdb/v3/<hand>/*.json
export EQUIDEXFLOW_OBJECTS_DIR=/path/to/objects      # defaults to $EQUIDEXFLOW_DATA_DIR/objects
```

## Per-object file

```json
{
  "object_name": "mustard_bottle",
  "n_grasps": 100,
  "grasps": [ { <grasp> }, ... ]
}
```

## Per-grasp schema

| field | shape | units / frame | required | notes |
|-------|-------|---------------|:--------:|-------|
| `contact_points_mm`  | (4,3) | **mm**, object frame | yes | loader divides by 1000 to metres |
| `contact_normals`    | (4,3) | unit, **inward**, object frame | yes | renormalized at load |
| `hand_dof_values`    | (16,) | radians | yes | `HAND_DOF=16` (Allegro/LEAP), Drake `GetPositionNames` order |
| `epsilon_quality`    | scalar | none | yes | Ferrari-Canny L1 / min-weight metric (force-closure) |
| `volume_quality`     | scalar | m^6 | yes | wrench-cone polytope volume |
| `wrist_pose_object`  | (4,4) | `T_object_wrist`, m | rec | optional, identity fallback; needed for FK rendering |
| `contact_finger_ids` | (4,)  | 0=thumb..3=ring | rec | optional, falls back to FPS+Lloyd clustering |
| `object_position_mm` | (3,)  | world frame | meta | emitter metadata; not consumed by the loader |
| `object_orientation` | (4,)  | quat (w,x,y,z) | meta | emitter metadata; not consumed by the loader |
| `metadata`           | dict  | none | meta | free-form (seed, solve time, mu, ...) |

`rec` = recommended, `meta` = metadata only (not consumed by the loader).

### Forces are computed, not stored

The loader synthesizes quasistatic-equilibrium contact forces at load time
(`compute_contact_forces(contacts, normals, object_mass, mu)`), so no `forces`
field is needed. Defaults: `mu=0.5`, `object_mass=0.2` (override in the config).

### Frame convention (important for bring-your-own-data)

The loader centers each example at the **mean of the sampled object point
cloud** and subtracts that mean from the object points, contacts, and wrist
translation **together**; `model.sample()` re-adds it so outputs return in the
input cloud's frame. You do **not** pre-center. You only need object points,
contacts, and the wrist pose expressed in **one consistent frame** plus a
discoverable mesh. Without a mesh the loader degrades to a noisy contact-point
proxy with zero normals (usable, lower quality).

### Minimum viable record

`contact_points_mm`, `contact_normals`, `hand_dof_values`, `epsilon_quality`,
`volume_quality`. Strongly recommended: `wrist_pose_object`,
`contact_finger_ids`, and a discoverable object mesh.

## Object meshes

Mesh stems are resolved through a name→path map in
`src/equidexflow/loaders/dexgrasp_db.py` (e.g. `cube → graspit/cube`,
`mustard_bottle → frogger_ycb/006_mustard_bottle`), relative to
`$EQUIDEXFLOW_OBJECTS_DIR`. EGAD objects (`^[A-Z]\d+$`) load from
`$EQUIDEXFLOW_EGAD_ROOT/{egad_eval_set,egad_train_set}/<name>.obj` scaled ×0.001.

## Reference emitter

The FRoGGeR adapter that produces these files from min-weight-metric grasp
synthesis is the reference implementation:
`frogger/frogger/equidex/dexgraspdb_adapter.py` (in the FRoGGeR repo). Mirror its
field mapping to emit from any other backbone.
