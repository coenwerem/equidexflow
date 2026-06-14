"""
PyTorch Dataset wrapping the grasp database JSON files (see data/README.md).

Each item returns a ``GraspExample`` dict (see ``loaders.schema``) with
contact positions converted to metres, quasistatic forces computed via
``loaders.force_label``, and optional SO(3) data augmentation.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from equidexflow.loaders.force_label import assign_finger_ids, compute_contact_forces
from equidexflow.loaders.schema import HAND_DOF, MAX_CONTACTS, N_FINGERS


def _cluster_contact_finger_ids(
    contact_points: np.ndarray,
    n_fingers: int = N_FINGERS,
    n_iter: int = 6,
) -> np.ndarray:
    """Assign stable pseudo-finger IDs by clustering contact points.

    The grasp JSON stores object contacts but not the fingertip/link
    that produced each contact.  The dex model supervises one contact per
    finger, so use deterministic farthest-point seeds followed by a few Lloyd
    updates as a geometry-only fallback.
    """
    n_contacts = len(contact_points)
    if n_contacts == 0:
        return np.empty(0, dtype=np.int64)

    points = np.asarray(contact_points, dtype=np.float32)
    n_clusters = min(n_fingers, n_contacts)

    centroid = points.mean(axis=0, keepdims=True)
    first = int(np.argmax(((points - centroid) ** 2).sum(axis=1)))
    seed_indices = [first]

    while len(seed_indices) < n_clusters:
        seeds = points[seed_indices]
        dist2 = ((points[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)
        seed_indices.append(int(np.argmax(dist2.min(axis=1))))

    centers = points[seed_indices].copy()
    labels = np.zeros(n_contacts, dtype=np.int64)
    for _ in range(n_iter):
        dist2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist2.argmin(axis=1).astype(np.int64)
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if mask.any():
                centers[cluster_id] = points[mask].mean(axis=0)

    return labels


# ---------------------------------------------------------------------------
# Collate function (handles the string ``object_name`` field)
# ---------------------------------------------------------------------------

def collate_fn_dexgrasp(batch: list) -> dict:
    """
    Collate a list of ``GraspExample`` dicts into a ``GraspBatch``.

    Stacks all tensor fields along a new leading batch dimension.
    The ``object_name`` field (``str``) is collected into a plain list.

    Parameters
    ----------
    batch : list of GraspExample dicts, length B

    Returns
    -------
    GraspBatch dict
    """
    collated: dict = {}
    for key in batch[0].keys():
        if key == "object_name":
            collated[key] = [sample[key] for sample in batch]
        else:
            collated[key] = torch.stack([sample[key] for sample in batch])
    return collated


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DexGraspDBDataset(Dataset):
    """
    Dataset over the grasp database JSON files.

    Each ``__getitem__`` call returns one grasp as a ``GraspExample`` dict
    with the following fields and shapes (see ``loaders.schema.GraspExample``):

    * ``object_points``  – (3, n_object_points)  float32
    * ``hand_q``         – (11,)                 float32
    * ``wrist_pose``     – (4, 4)                float32  SE(3) matrix
    * ``contacts``       – (max_contacts, 3)     float32  metres
    * ``normals``        – (max_contacts, 3)     float32
    * ``forces``         – (max_contacts, 3)     float32  Newtons
    * ``finger_ids``     – (max_contacts,)       int64    0-4, -1 for padding
    * ``valid_mask``     – (max_contacts,)       bool
    * ``object_name``    – str
    * ``grasp_quality``  – (2,)                  float32  [epsilon, volume]

    Parameters
    ----------
    grasp_db_dir    : path to directory containing ``*.json`` grasp files
    object_mesh_dir : path to object meshes (gracefully skipped if absent)
    n_object_points : number of points to sample for the object point cloud
    max_contacts    : fixed padded size for contact arrays
    mu              : Coulomb friction coefficient for force labeling
    object_mass     : object mass [kg] for force labeling
    augment         : apply random SO(3) rotation augmentation when True
    split           : ``'train'`` | ``'val'`` | ``'test'``
    train_ratio     : fraction of grasps used for training
    val_ratio       : fraction of grasps used for validation
    """

    # Map from object name (as stored in JSON) to mesh filename stem.
    # Add entries here when new objects are added to the grasp_db.
    _MESH_STEM: Dict[str, str] = {
        # graspit primitives
        "box": "graspit/box",
        "cube": "graspit/cube",
        "cylinder": "graspit/cylinder",
        "graspit_box": "graspit/box",
        "graspit_cylinder": "graspit/cylinder",
        "sphere": "graspit/sphere",
        "phydex": "graspit/phydex",
        "sns_cup": "graspit/sns_cup",
        "insertion_object": "insertion_object",
        # YCB via frogger clean meshes (symlinked into frogger_ycb/)
        "chips_can": "frogger_ycb/001_chips_can",
        "master_chef_can": "frogger_ycb/002_master_chef_can",
        "cracker_box": "frogger_ycb/003_cracker_box",
        "sugar_box": "frogger_ycb/004_sugar_box",
        "tomato_soup_can": "frogger_ycb/005_tomato_soup_can",
        "mustard_bottle": "frogger_ycb/006_mustard_bottle",
        "tuna_fish_can": "frogger_ycb/007_tuna_fish_can",
        "pudding_box": "frogger_ycb/008_pudding_box",
        "gelatin_box": "frogger_ycb/009_gelatin_box",
        "potted_meat_can": "frogger_ycb/010_potted_meat_can",
        "banana": "frogger_ycb/011_banana",
        "strawberry": "frogger_ycb/012_strawberry",
        "apple": "frogger_ycb/013_apple",
        "lemon": "frogger_ycb/014_lemon",
        "peach": "frogger_ycb/015_peach",
        "pear": "frogger_ycb/016_pear",
        "orange": "frogger_ycb/017_orange",
        "plum": "frogger_ycb/018_plum",
        "bleach_cleanser": "frogger_ycb/021_bleach_cleanser",
        "wood_block": "frogger_ycb/036_wood_block",
        "softball": "frogger_ycb/054_softball",
        "baseball": "frogger_ycb/055_baseball",
        "tennis_ball": "frogger_ycb/056_tennis_ball",
        "racquetball": "frogger_ycb/057_racquetball",
        "golf_ball": "frogger_ycb/058_golf_ball",
        "foam_brick": "frogger_ycb/061_foam_brick",
        "rubiks_cube": "frogger_ycb/077_rubiks_cube",
        "sns_cup": "frogger_ycb/sns_cup",
    }

    _EGAD_ROOT: str = os.environ.get("EQUIDEXFLOW_EGAD_ROOT", os.path.expanduser("~/.cache/equidexflow/egad"))

    # FRoGGeR's `_PRIMITIVE_SPECS` from frogger/ablation_runner.py - the exact
    # generator arguments used during grasp synthesis. The on-disk
    # graspit/*.stl meshes mismatch these for `graspit_box` (axis-swapped) and
    # `sphere` (r=35mm vs r=30mm); training from the disk meshes shifts
    # contacts/wrist off the loaded surface. Set `use_frogger_primitive_specs`
    # to True (default in retrain configs) to override disk loading with the
    # canonical trimesh.creation generators.
    _FROGGER_PRIMITIVE_SPECS: Dict[str, tuple] = {
        # name: (kind, args_for_trimesh_creation)
        "cube": ("box", {"extents": (0.05, 0.05, 0.05)}),
        "graspit_box": ("box", {"extents": (0.0625, 0.0625, 0.1625)}),
        "graspit_cylinder": ("cylinder", {"radius": 0.022, "height": 0.180, "sections": 64}),
        "sphere": ("sphere", {"radius": 0.03, "subdivisions": 3}),
    }

    def __init__(
        self,
        grasp_db_dir: str,
        object_mesh_dir: str,
        n_object_points: int = 1024,
        max_contacts: int = MAX_CONTACTS,
        mu: float = 0.5,
        object_mass: float = 0.2,
        augment: bool = False,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        object_names: Optional[Sequence[str]] = None,
        use_frogger_primitive_specs: bool = False,
        wrist_frame: str = "base",
        pre_split: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        object_names : optional list of object names to include.
            If None, all objects in grasp_db_dir are used.
            Example: ``['box', 'cube', 'cylinder']`` for the graspit primitives.
        wrist_frame : ``'base'`` (palm frame, default) or ``'grasp_center'``
            (dataset-mean contact centroid). When ``'grasp_center'``, the wrist
            pose target is relabeled so the SE(3) flow predicts a frame at the
            grasp center instead of the palm, shrinking the rotation-error lever.
        pre_split : if True, treat the loaded grasps as ALREADY belonging to
            ``split`` (i.e., the directory on disk already contains only that
            split's grasps). The internal RandomState(42) split is skipped.
            Use this when shipping a test-only data release: point at the
            released test tarball and set ``pre_split=True, split='test'``
            and the loader uses all loaded grasps as the test set.
        """
        super().__init__()

        assert split in ("train", "val", "test"), \
            f"split must be 'train'|'val'|'test', got '{split}'"
        assert train_ratio + val_ratio < 1.0, \
            "train_ratio + val_ratio must be < 1.0"
        assert wrist_frame in ("base", "grasp_center"), \
            f"wrist_frame must be 'base'|'grasp_center', got '{wrist_frame}'"

        self.grasp_db_dir = grasp_db_dir
        self.object_mesh_dir = object_mesh_dir
        self.n_object_points = n_object_points
        self.max_contacts = max_contacts
        self.mu = mu
        self.object_mass = object_mass
        self.augment = augment
        self.split = split
        self.wrist_frame = wrist_frame
        self.object_names = set(object_names) if object_names is not None else None
        self.use_frogger_primitive_specs = use_frogger_primitive_specs

        # Pre-load and cache trimesh meshes keyed by object name
        self._mesh_cache: Dict[str, object] = {}

        # Track per-object proxy-fallback events so we can warn loudly when
        # a mesh is missing instead of silently substituting a degenerate
        # tiled-contact point cloud. Strict mode (env flag) raises instead.
        self._proxy_fallback_seen: Set[str] = set()
        self._strict_mesh: bool = os.environ.get(
            "EQUIDEXFLOW_STRICT_MESH", "0"
        ).lower() in ("1", "true", "yes")

        # ------------------------------------------------------------------
        # Load all grasps from every *.json file in grasp_db_dir
        # ------------------------------------------------------------------
        self._all_grasps: List[dict] = []
        self._load_db(grasp_db_dir)

        # Load meshes after DB so we know all object names present in data
        self._load_meshes()

        if len(self._all_grasps) == 0:
            raise ValueError(
                f"No grasps found in '{grasp_db_dir}'"
                + (f" for objects {self.object_names}" if self.object_names else "") + "."
            )

        # ------------------------------------------------------------------
        # Deterministic train / val / test split
        # ------------------------------------------------------------------
        n_total = len(self._all_grasps)
        if pre_split:
            # The on-disk directory is the released test (or train/val) split
            # itself; don't re-partition. Use every loaded grasp.
            self.indices = np.arange(n_total)
        else:
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            rng = np.random.RandomState(42)
            shuffled = rng.permutation(n_total)
            if split == "train":
                self.indices = shuffled[:n_train]
            elif split == "val":
                self.indices = shuffled[n_train : n_train + n_val]
            else:  # test
                self.indices = shuffled[n_train + n_val :]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_meshes(self) -> None:
        """Load and cache trimesh meshes for all requested objects."""
        try:
            import trimesh  # type: ignore
        except ImportError:
            trimesh = None

        if not os.path.isdir(self.object_mesh_dir):
            return

        data_names = set(g["object_name"] for g in self._all_grasps) if self._all_grasps else set()
        names_to_load = self.object_names or (data_names | set(self._MESH_STEM.keys()))
        for name in names_to_load:
            if name in self._mesh_cache:
                continue
            # FRoGGeR-spec override: rebuild primitives via trimesh.creation
            # with the exact args used during grasp synthesis. Avoids the
            # axis-swap (graspit_box) and radius mismatch (sphere) between the
            # in-memory FRoGGeR primitives and the on-disk graspit/*.stl files.
            if (self.use_frogger_primitive_specs
                    and trimesh is not None
                    and name in self._FROGGER_PRIMITIVE_SPECS):
                kind, kwargs = self._FROGGER_PRIMITIVE_SPECS[name]
                if kind == "box":
                    mesh = trimesh.creation.box(**kwargs)
                elif kind == "cylinder":
                    mesh = trimesh.creation.cylinder(**kwargs)
                elif kind == "sphere":
                    mesh = trimesh.creation.icosphere(**kwargs)
                else:
                    raise ValueError(f"unknown FRoGGeR primitive kind {kind!r}")
                mesh.process(validate=True)
                self._mesh_cache[name] = mesh
                continue
            stem = self._MESH_STEM.get(name, name)
            found = False
            for ext in (".stl", ".obj", ".ply", ".STL"):
                path = os.path.join(self.object_mesh_dir, stem + ext)
                if os.path.isfile(path):
                    if trimesh is not None:
                        try:
                            mesh = trimesh.load(path, force="mesh")
                            # IMPORTANT: do NOT center the mesh here. Targets
                            # (wrist_pose, contacts) in the JSON live in the
                            # body frame, not the geometric-centroid frame.
                            # __getitem__ performs the mean-subtraction on
                            # BOTH points and targets together so they stay
                            # consistent. See EquiGraspFlow §4.2.
                            self._mesh_cache[name] = mesh
                            found = True
                        except Exception:
                            pass
                    break
            if not found and trimesh is not None:
                import re
                if re.match(r"^[A-Z]\d+$", name):
                    for setname in ("egad_eval_set", "egad_train_set"):
                        egad_path = os.path.join(self._EGAD_ROOT, setname, f"{name}.obj")
                        if os.path.isfile(egad_path):
                            try:
                                mesh = trimesh.load(egad_path, force="mesh")
                                mesh.apply_scale(0.001)
                                # NO centering - see comment above. EGAD
                                # meshes come from the file in mm; we only
                                # apply scale, leaving the body-frame origin
                                # as authored.
                                self._mesh_cache[name] = mesh
                            except Exception:
                                pass
                            break

    def _load_db(self, grasp_db_dir: str) -> None:
        """Read all JSON grasp files, optionally filtering by object name."""
        for fname in sorted(os.listdir(grasp_db_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(grasp_db_dir, fname)
            with open(fpath) as fh:
                db = json.load(fh)
            obj_name = db.get("object_name", fname.replace(".json", ""))
            if self.object_names is not None and obj_name not in self.object_names:
                continue
            self._all_grasps.extend(db["grasps"])

    def _sample_object_points(
        self, object_name: str, contacts_m: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample a point cloud + per-point outward normals for the object.

        Uses the cached trimesh mesh via ``trimesh.sample.sample_surface``
        which returns face indices, enabling per-point face-normal lookup.
        Falls back to a contact-point proxy (with zero normals) otherwise.

        Parameters
        ----------
        object_name : str
        contacts_m  : (N, 3) contact positions in metres (object frame)

        Returns
        -------
        pts     : (3, n_object_points) float32
        normals : (3, n_object_points) float32, outward-pointing unit vectors.
                  Zero where the mesh is unavailable (signals the loss to skip).
        """
        mesh = self._mesh_cache.get(object_name)
        if mesh is not None:
            try:
                samples, face_idx = trimesh.sample.sample_surface(
                    mesh, self.n_object_points
                )
                pts = samples.astype(np.float32)                       # (N_pts, 3)
                face_normals = np.asarray(mesh.face_normals)           # (F, 3)
                pt_normals = face_normals[face_idx].astype(np.float32) # (N_pts, 3)
                # Re-normalise defensively in case of degenerate faces.
                nrm = np.linalg.norm(pt_normals, axis=1, keepdims=True)
                pt_normals = pt_normals / np.where(nrm > 1e-8, nrm, 1.0)
                return pts.T, pt_normals.T.astype(np.float32)
            except Exception:
                pass

        # Proxy fallback: tile contact points + Gaussian noise; zero normals.
        # Loudly surface this — silent substitution made eval runs look valid
        # while producing meaningless point clouds. Strict mode raises so the
        # reproduce path fails fast when meshes are unset.
        if self._strict_mesh:
            raise FileNotFoundError(
                f"No mesh available for object '{object_name}' in "
                f"'{self.object_mesh_dir}' (EQUIDEXFLOW_STRICT_MESH=1). "
                f"Point EQUIDEXFLOW_OBJECTS_DIR at the YCB/EGAD/GraspIt "
                f"meshes; see REPRODUCE.md."
            )
        if object_name not in self._proxy_fallback_seen:
            self._proxy_fallback_seen.add(object_name)
            warnings.warn(
                f"[dexgrasp_db] mesh missing for '{object_name}'; falling back "
                f"to a degenerate contact-proxy point cloud (zero normals). "
                f"Numbers from this object are NOT meaningful. "
                f"({len(self._proxy_fallback_seen)} object(s) on proxy so far.) "
                f"Set EQUIDEXFLOW_STRICT_MESH=1 to make this an error.",
                RuntimeWarning,
                stacklevel=2,
            )
        N = len(contacts_m)
        if N >= 1:
            rng = np.random.default_rng()
            rep = int(np.ceil(self.n_object_points / max(N, 1)))
            pts = np.tile(contacts_m, (rep, 1))[: self.n_object_points]
            pts = pts + rng.standard_normal(pts.shape).astype(np.float32) * 0.005
        else:
            pts = np.random.randn(self.n_object_points, 3).astype(np.float32) * 0.05

        pts = pts.astype(np.float32)
        pt_normals = np.zeros_like(pts, dtype=np.float32)
        return pts.T, pt_normals.T

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        """
        Return one ``GraspExample`` dict.

        Parameters
        ----------
        idx : dataset index in [0, len(self))

        Returns
        -------
        GraspExample dict (see class docstring for shapes)
        """
        grasp = self._all_grasps[int(self.indices[idx])]

        # ------------------------------------------------------------------
        # Raw contact data
        # ------------------------------------------------------------------
        contacts_m: np.ndarray = (
            np.asarray(grasp["contact_points_mm"], dtype=float) / 1000.0
        )  # (N, 3) metres
        normals: np.ndarray = np.asarray(
            grasp["contact_normals"], dtype=float
        )  # (N, 3)

        # Ensure unit normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(norms > 1e-8, norms, 1.0)

        N = len(contacts_m)

        # ------------------------------------------------------------------
        # Joint angles
        # ------------------------------------------------------------------
        hand_q: np.ndarray = np.asarray(
            grasp["hand_dof_values"], dtype=np.float32
        )  # (11,)

        # ------------------------------------------------------------------
        # Wrist pose (object frame). Prefer FRoGGeR-emitted T_object_wrist;
        # fall back to identity for legacy JSONs without the field.
        # ------------------------------------------------------------------
        if "wrist_pose_object" in grasp:
            wrist_pose = np.asarray(
                grasp["wrist_pose_object"], dtype=np.float32
            )  # (4, 4)
        else:
            wrist_pose = np.eye(4, dtype=np.float32)

        # ------------------------------------------------------------------
        # Quasistatic contact forces
        # ------------------------------------------------------------------
        forces: np.ndarray = compute_contact_forces(
            contacts_m, normals,
            object_mass=self.object_mass, mu=self.mu,
        )  # (N, 3)
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        max_force = max(5.0 * self.object_mass * 9.81, 1.0)
        forces = forces * np.minimum(1.0, max_force / np.maximum(force_norms, 1e-8))

        # ------------------------------------------------------------------
        # Object point cloud sampled from mesh (or contact proxy fallback)
        # ------------------------------------------------------------------
        object_name: str = grasp["object_name"]
        object_points, object_point_normals = self._sample_object_points(
            object_name, contacts_m
        )  # both (3, N_pts)

        # ------------------------------------------------------------------
        # Finger-centric wrist relabel: convert base -> grasp_center BEFORE
        # centering so the flow target is the grasp-center frame.
        # ------------------------------------------------------------------
        if self.wrist_frame == "grasp_center":
            from equidexflow.kinematics.allegro_fk import shift_wrist_frame as _shift
            wrist_pose = _shift(wrist_pose, to_base=False)

        # ------------------------------------------------------------------
        # R^3 equivariance via mean-subtraction (EquiGraspFlow §4.2).
        # Center the point cloud at its own mean and shift ALL spatial
        # targets (wrist translation, contacts) by the same mu so they live
        # in the same centered frame. Without this, VN-DGCNN gives correct
        # SO(3) behavior around the point-cloud centroid, but the wrist /
        # contact MLP heads have no consistent translation anchor and the
        # output cloud floats with a per-object offset (the body-frame ->
        # geometric-centroid offset of each mesh). Verified upstream in
        # EquiGraspFlow's acronym.py loader (subtracts pc mean from BOTH pc
        # and Ts_grasp[..., :3, 3]).
        pc_mean = object_points.mean(axis=1)                       # (3,)
        pc_mean_obj = pc_mean.astype(np.float32).copy()            # offset to undo centering
        object_points = object_points - pc_mean[:, None]           # (3, N_pts)
        contacts_m = contacts_m - pc_mean[None, :]                 # (N, 3)
        wrist_pose[:3, 3] = wrist_pose[:3, 3] - pc_mean
        # object_point_normals, contact normals, forces, hand_q are
        # translation-intrinsic; do NOT shift them.

        # ------------------------------------------------------------------
        # SO(3) augmentation (applied consistently to all spatial quantities).
        # Order: centering BEFORE SO(3) rotation. Rotation about the centered
        # origin is the standard convention used by EquiGraspFlow.
        # ------------------------------------------------------------------
        if self.augment:
            R = Rotation.random().as_matrix().astype(np.float32)  # (3, 3)
            object_points = R @ object_points                      # (3, N_pts)
            object_point_normals = R @ object_point_normals        # (3, N_pts)
            contacts_m = (contacts_m @ R.T).astype(np.float32)    # (N, 3)
            normals = (normals @ R.T).astype(np.float32)           # (N, 3)
            forces = (forces @ R.T).astype(np.float32)             # (N, 3)
            wrist_pose[:3, :3] = R @ wrist_pose[:3, :3]
            wrist_pose[:3, 3] = R @ wrist_pose[:3, 3]

            # ------------------------------------------------------------------
            # R^3 translation augmentation: shift centered points + targets by
            # the same random offset. Forces the network to be invariant to
            # the absolute world-frame origin (any leftover from imperfect
            # centering, downstream pose composition, etc.). Disabled when
            # augment=False to keep eval deterministic.
            # ------------------------------------------------------------------
            t_aug = (np.random.randn(3) * 0.05).astype(np.float32)  # 5cm std
            object_points = object_points + t_aug[:, None]
            contacts_m = contacts_m + t_aug
            wrist_pose[:3, 3] = wrist_pose[:3, 3] + t_aug

        # ------------------------------------------------------------------
        # Finger ID assignment. Prefer explicit `contact_finger_ids` from the
        # grasp JSON when present (FRoGGeR knows which contact came from
        # which fingertip); fall back to geometry-only clustering otherwise.
        # ------------------------------------------------------------------
        if "contact_finger_ids" in grasp:
            finger_ids_raw = np.asarray(
                grasp["contact_finger_ids"], dtype=np.int64
            )
        else:
            finger_ids_raw = _cluster_contact_finger_ids(contacts_m, N_FINGERS)

        # ------------------------------------------------------------------
        # Pad to max_contacts
        # ------------------------------------------------------------------
        M = self.max_contacts
        n_valid = min(N, M)

        contacts_pad = np.zeros((M, 3), dtype=np.float32)
        normals_pad = np.zeros((M, 3), dtype=np.float32)
        forces_pad = np.zeros((M, 3), dtype=np.float32)
        finger_ids_pad = np.full(M, -1, dtype=np.int64)
        valid_mask = np.zeros(M, dtype=bool)

        contacts_pad[:n_valid] = contacts_m[:n_valid].astype(np.float32)
        normals_pad[:n_valid] = normals[:n_valid].astype(np.float32)
        forces_pad[:n_valid] = forces[:n_valid].astype(np.float32)
        finger_ids_pad[:n_valid] = finger_ids_raw[:n_valid]
        valid_mask[:n_valid] = True

        return {
            "object_points": torch.from_numpy(object_points),               # (3, N_pts)
            "object_point_normals": torch.from_numpy(object_point_normals), # (3, N_pts)
            "hand_q": torch.from_numpy(hand_q),                             # (11,)
            "wrist_pose": torch.from_numpy(wrist_pose),                     # (4, 4)
            "contacts": torch.from_numpy(contacts_pad),                     # (M, 3)
            "normals": torch.from_numpy(normals_pad),                       # (M, 3)
            "forces": torch.from_numpy(forces_pad),                         # (M, 3)
            "finger_ids": torch.from_numpy(finger_ids_pad),                 # (M,)
            "valid_mask": torch.from_numpy(valid_mask),                     # (M,)
            "object_name": object_name,
            "pc_mean": torch.from_numpy(pc_mean_obj),                       # (3,) centering offset (object frame)
            "grasp_quality": torch.tensor(
                [grasp["epsilon_quality"], grasp["volume_quality"]],
                dtype=torch.float32,
            ),  # (2,)
        }
