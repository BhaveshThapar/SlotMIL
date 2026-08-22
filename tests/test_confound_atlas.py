"""Tests for the confound atlas.

Pinned: the slice-major C-order layout contract (the silent-mis-score hazard
``slotmil/eval/lung.py`` documents), split determinism and refusals, and an
end-to-end sanity check that the pipeline recovers a known confound axis --
a peripheral-mass dataset must profile in-plane, a basal-mass one axial.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.confound_atlas import N_PATCH, analyse, fit_half
from slotmil.data.atlas import GRID, _flat_grid


def test_flat_grid_is_slice_major_c_order():
    lesion = np.zeros((3, 64, 64), dtype=bool)
    lesion[1, :4, :4] = True  # slice 1, top-left patch of a 16x16 grid
    flat = _flat_grid(lesion)
    assert flat.shape == (3 * N_PATCH,)
    assert flat[:N_PATCH].sum() == 0, "slice 0 must be empty"
    assert flat[N_PATCH] > 0, "slice 1 patch (0,0) sits at index 1*256+0"
    assert flat[2 * N_PATCH:].sum() == 0, "slice 2 must be empty"


def test_fit_half_is_deterministic():
    assert all(fit_half(f"case{i}") == fit_half(f"case{i}") for i in range(50))
    halves = {fit_half(f"case{i}") for i in range(50)}
    assert halves == {True, False}, "50 hashed ids must land in both halves"


def test_a_case_yielding_twice_is_refused():
    flat = np.zeros(N_PATCH, dtype=np.float32)
    flat[0] = 1.0
    cases = [("dup", "dup", flat), ("dup", "dup", flat)]
    with pytest.raises(SystemExit, match="yielded twice"):
        analyse(cases, 0, 0, 32)


def test_a_broken_layout_is_refused():
    cases = [("bad", "bad", np.zeros(N_PATCH + 1, dtype=np.float32))]
    with pytest.raises(SystemExit, match="layout contract"):
        analyse(cases, 0, 0, 32)


def _cases(place):
    """24 synthetic cases, 4 slices each; ``place(rng, i)`` -> (slice, patch)."""
    out = []
    rng = np.random.default_rng(0)
    for i in range(24):
        flat = np.zeros(4 * N_PATCH, dtype=np.float32)
        s, p = place(rng, i)
        flat[s * N_PATCH + p] = 1.0
        out.append((f"case{i}", f"case{i}", flat))
    return out


def test_a_peripheral_mass_dataset_profiles_in_plane():
    # Lesion always at the same in-plane patch, on a varying slice.
    cases = _cases(lambda rng, i: (int(rng.integers(0, 4)), 0))
    res = analyse(cases, 0, 0, 32)
    inplane = res["scorers"]["masks:inplane"]["flat_auc"]["mean"]
    axial = res["scorers"]["masks:axial"]["flat_auc"]["mean"]
    assert inplane > 0.95
    assert inplane > axial + 0.2


def test_a_basal_mass_dataset_profiles_axial():
    # Lesion always on the last slice, at a varying in-plane patch.
    cases = _cases(lambda rng, i: (3, int(rng.integers(0, N_PATCH))))
    res = analyse(cases, 0, 0, 32)
    inplane = res["scorers"]["masks:inplane"]["flat_auc"]["mean"]
    axial = res["scorers"]["masks:axial"]["flat_auc"]["mean"]
    # A constant-within-slice scorer ties against 255 same-slice patches, so
    # its flat AUC tops out below 1; the slice axis is where it saturates.
    assert res["scorers"]["masks:axial"]["slice_auc"]["mean"] > 0.95
    assert axial > 0.8
    assert axial > inplane + 0.2


def test_grid_is_pinned_to_the_paper_geometry():
    assert GRID == 16 and N_PATCH == 256
