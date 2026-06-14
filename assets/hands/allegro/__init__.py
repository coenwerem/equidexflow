"""Bundled Allegro hand description (SDF + visual meshes).

Provenance: ``allegro_rh.sdf`` and its meshes are vendored UNMODIFIED from
FRoGGeR (Li et al., IROS 2023; https://github.com/alberthli/frogger), under that
repo's MIT License -- verified byte-identical (sha256 924dbf4d…) to
alberthli/frogger ``main/models/allegro/allegro_rh.sdf``. FRoGGeR itself adapted
it from Drake's ``allegro_hand_description`` (renamed links/joints; see the SDF
header). See the repo NOTICE for full attribution.

This directory is the single source of truth for the Allegro hand assets. It is
mapped into the installed package as ``equidexflow._allegro_hand`` (see
``pyproject.toml`` ``[tool.setuptools.package-dir]``) so the renderer can locate
the SDF via ``importlib.resources`` on non-editable installs, while editable /
clone workflows keep referencing it at ``assets/hands/allegro/``.
"""
