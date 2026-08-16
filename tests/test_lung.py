"""Reading the lung store without silently mis-scoring the bag it restricts.

Two facts about ``data/lidc/lung_masks.h5`` decide every test here, and both were
verified against all 999 real series before this module was written:

* the store is at **cached-slice** resolution, so it aligns to a bag
  positionally and must not be reindexed through ``slice_index``;
* ``lung_thresh`` is pre-registered at 0.0 with the patch rule "any lung at
  all", so the comparison is strictly greater-than. At threshold zero ``>=``
  would mark every patch in-lung and quietly turn H8 into the unrestricted
  number it exists to be contrasted against.

The DICOM this store was derived from has been deleted, so a wrong reader cannot
be caught by regenerating it.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from slotmil.eval.lung import (
    DEFAULT_LUNG_THRESH,
    in_lung_from_grid,
    load_lung_grid,
    restrict_to_lung,
)

N_SLICES = 3
GRID = 4
N_PATCH = GRID * GRID


@pytest.fixture
def store(tmp_path):
    """One series, with a known number of in-lung patches per slice."""
    path = tmp_path / "lung.h5"
    grid = np.zeros((N_SLICES, GRID, GRID), dtype=np.float32)
    grid[0, 0, :] = 1.0        # 4 patches fully lung
    grid[1, 1, :2] = 0.01      # 2 patches barely lung -- still in, by the rule
    # slice 2 stays empty: a slice with no lung at all is normal near the apex
    with h5py.File(path, "w") as f:
        g = f.create_group("uid00")
        g.create_dataset("lung", data=grid)
        g.attrs["method"] = "fill"
        g.attrs["lung_thresh"] = 0.0
    return str(path), grid


class TestLoad:
    def test_it_returns_the_stored_grid(self, store):
        path, grid = store
        with h5py.File(path, "r") as f:
            out = load_lung_grid(f, "uid00")
        assert out.shape == (N_SLICES, GRID, GRID)
        assert np.array_equal(out, grid)

    def test_an_unknown_series_raises(self, store):
        path, _ = store
        with h5py.File(path, "r") as f, pytest.raises(KeyError, match="not in"):
            load_lung_grid(f, "nope")

    def test_a_half_written_group_raises(self, tmp_path):
        """The store's own resume check treats a missing ``lung`` as incomplete;
        so does the reader, rather than returning an empty restriction."""
        path = tmp_path / "partial.h5"
        with h5py.File(path, "w") as f:
            f.create_group("uid00").attrs["method"] = "fill"
        with h5py.File(path, "r") as f, pytest.raises(KeyError, match="incomplete"):
            load_lung_grid(f, "uid00")


class TestThePatchRule:
    def test_any_lung_at_all_counts(self, store):
        _, grid = store
        in_lung = in_lung_from_grid(grid, DEFAULT_LUNG_THRESH)
        assert in_lung.sum() == 6                      # 4 full + 2 at 0.01
        assert in_lung.shape == (N_SLICES * N_PATCH,)

    def test_the_comparison_is_strict(self, store):
        """``>=`` at threshold 0.0 would mark all 48 patches in-lung. The whole
        point of restricting is that it removes about three quarters of them."""
        _, grid = store
        assert in_lung_from_grid(grid, 0.0).sum() < grid.size

    def test_a_higher_threshold_drops_the_marginal_patches(self, store):
        _, grid = store
        assert in_lung_from_grid(grid, 0.5).sum() == 4

    def test_the_flattening_is_c_order(self, store):
        """Slice ``s`` patch ``p`` must land at ``s * N_PATCH + p`` -- the layout
        the cache's mask and the attention dumps both use."""
        _, grid = store
        in_lung = in_lung_from_grid(grid, DEFAULT_LUNG_THRESH)
        assert in_lung[:GRID].all()                    # slice 0, row 0
        assert not in_lung[2 * N_PATCH:].any()         # slice 2, empty

    def test_a_two_dimensional_grid_raises(self):
        with pytest.raises(ValueError, match=r"\[n_slices, g, g\]"):
            in_lung_from_grid(np.zeros((GRID, GRID), dtype=np.float32))


class TestAlignment:
    def test_a_bag_of_the_wrong_length_raises(self, store):
        """The failure this guard exists for is silent: a short zip truncates the
        analysis instead of raising, which is why ruff's B905 ignore is recorded
        as pending an audit."""
        _, grid = store
        with pytest.raises(ValueError, match="different caches"):
            in_lung_from_grid(grid, DEFAULT_LUNG_THRESH, n_expected=N_PATCH)

    def test_the_expected_length_passes_through(self, store):
        _, grid = store
        out = in_lung_from_grid(grid, DEFAULT_LUNG_THRESH,
                                n_expected=N_SLICES * N_PATCH)
        assert out.sum() == 6


class TestRestrict:
    def test_scores_and_targets_are_dropped_together(self, store):
        _, grid = store
        in_lung = in_lung_from_grid(grid, DEFAULT_LUNG_THRESH)
        scores = np.arange(N_SLICES * N_PATCH, dtype=np.float32)
        mask = np.zeros(N_SLICES * N_PATCH, dtype=np.int8)
        mask[0] = 1          # in lung
        mask[N_PATCH + 8] = 1  # out of lung -- must not survive
        s, t = restrict_to_lung(scores, mask, in_lung)
        assert s.shape == t.shape == (6,)
        assert t.sum() == 1
        assert np.array_equal(s, scores[in_lung])

    def test_the_target_is_binarised(self, store):
        _, grid = store
        in_lung = in_lung_from_grid(grid, DEFAULT_LUNG_THRESH)
        scores = np.zeros(N_SLICES * N_PATCH, dtype=np.float32)
        mask = np.zeros(N_SLICES * N_PATCH, dtype=np.float32)
        mask[0] = 0.03       # the cache stores a coverage fraction, not a flag
        _, t = restrict_to_lung(scores, mask, in_lung)
        assert t.dtype == np.int8 and t.sum() == 1

    def test_mismatched_lengths_raise(self, store):
        _, grid = store
        in_lung = in_lung_from_grid(grid, DEFAULT_LUNG_THRESH)
        with pytest.raises(ValueError, match="lengths disagree"):
            restrict_to_lung(np.zeros(3, dtype=np.float32),
                             np.zeros(3, dtype=np.int8), in_lung)
