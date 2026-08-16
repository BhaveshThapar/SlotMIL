"""Tests for the axis decomposition -- the split the paper's lead claim rests on.

Same design rule as ``tests/test_estimands.py``: the tests come in pairs. An axis
column is only worth reporting if it *collapses* on a scorer that carries no
information about that axis **and** *fires* on one that does. Either half alone
is satisfiable by a broken instrument -- a column that always returns 0.5 passes
every collapse test ever written, and a column that always returns 1.0 passes
every power test.

Two of the pairs are analytic rather than approximate, and they are pinned as
exact equalities on purpose:

* A purely in-plane scorer -- the same 256 numbers tiled across every slice --
  makes every slice's mean attention *identical*, so every slice-axis pair ties
  and the slice AUC is **exactly** 0.5. Not 0.5 within tolerance: 0.5.
* A purely axial scorer -- constant within each slice, varying across them --
  makes the within-slice AUC exactly 0.5 by the mirror argument, provided each
  lesion slice carries the same number of lesion patches.

The first of those is the paper's central claim in miniature. The in-plane
oracle in ``TestTheLeadClaim`` knows nothing whatsoever about depth, yet reports
a flat instance AUC of 0.995 -- a number that reads as 3D localisation. H1 says
``|flat - within_slice| < 0.02`` for every arm; this fixture shows why that gap
is the right thing to look at and a high flat AUC is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from slotmil.eval.axes import per_bag_axes
from slotmil.eval.nulls import N_PATCH

# Column positions in the returned tuple, named so the tests read as prose.
IDX, FLAT, SLICE, WITHIN, N_SLICES, N_LESION_SLICES = range(6)

# A nodule spans several slices at the *same* in-plane position, which is what
# makes a tiled in-plane map a perfect within-slice localiser and exactly chance
# axially. Real nodules do this; it is the structure the paper is about.
OFFSETS = (100, 101, 116, 117)


def stacked_bag(n_slices=8, lesion_slices=(2, 3, 4), offsets=OFFSETS):
    """A mask with `offsets` lit on each of `lesion_slices` and nowhere else."""
    m = np.zeros(n_slices * N_PATCH, dtype=np.int8)
    for z in lesion_slices:
        m[z * N_PATCH + np.array(offsets)] = 1
    return m


def in_plane_attn(m, values):
    """The same 256 numbers on every slice. Cannot encode depth, by construction."""
    return np.tile(np.asarray(values, dtype=float), len(m) // N_PATCH)[None, :]


def axial_attn(m, per_slice):
    """Constant within each slice. Cannot encode in-plane position, by construction."""
    return np.repeat(np.asarray(per_slice, dtype=float), N_PATCH)[None, :]


def in_plane_oracle(m, offsets=OFFSETS):
    v = np.zeros(N_PATCH)
    v[list(offsets)] = 1.0
    return in_plane_attn(m, v)


def one_row(attn, mask, slot=0, **kw):
    rows = per_bag_axes([attn], [mask], slot, **kw)
    assert len(rows) == 1, "fixture bag was dropped; the test is not measuring what it thinks"
    return rows[0]


class TestSliceAxis:
    def test_an_in_plane_scorer_is_exactly_chance_on_the_slice_axis(self):
        """Analytic, not approximate. Tiling one map across every slice makes
        every slice's mean identical, so every pair ties and the AUC is 0.5 on
        the nose. This is the paper's central claim, so it is pinned as an exact
        equality -- an implementation that drifted to 0.4998 would still 'pass'
        an approximate check while no longer being the statement we make."""
        m = stacked_bag()
        row = one_row(in_plane_attn(m, np.random.default_rng(0).random(N_PATCH)), m)
        assert row[SLICE] == 0.5

    def test_it_is_exactly_chance_whatever_the_in_plane_map_is(self):
        """The tie argument does not depend on the map, so neither should the
        number. Guards against passing by luck on one seed."""
        m = stacked_bag()
        for seed in range(5):
            v = np.random.default_rng(seed).random(N_PATCH)
            assert one_row(in_plane_attn(m, v), m)[SLICE] == 0.5

    def test_an_axial_scorer_is_found_by_the_slice_axis(self):
        """The other half. A column that reads 0.5 for everything would pass the
        test above and be worthless."""
        lesion_slices = (2, 3, 4)
        m = stacked_bag(n_slices=8, lesion_slices=lesion_slices)
        per_slice = np.zeros(8)
        per_slice[list(lesion_slices)] = 1.0
        assert one_row(axial_attn(m, per_slice), m)[SLICE] == 1.0


class TestWithinSliceAxis:
    def test_an_axial_scorer_is_exactly_chance_within_slice(self):
        """The mirror of the in-plane case. Restricted to lesion slices, a
        slice-constant scorer ties every pair inside a slice; with the same
        lesion count on each lesion slice the cross-slice pairs cancel exactly,
        so the AUC is 0.5 and not merely near it."""
        lesion_slices = (1, 4, 6)
        m = stacked_bag(n_slices=8, lesion_slices=lesion_slices)
        per_slice = np.random.default_rng(0).random(8)
        assert one_row(axial_attn(m, per_slice), m)[WITHIN] == 0.5

    def test_it_is_exactly_chance_whatever_the_axial_profile_is(self):
        lesion_slices = (1, 4, 6)
        m = stacked_bag(n_slices=8, lesion_slices=lesion_slices)
        for seed in range(5):
            per_slice = np.random.default_rng(seed).random(8)
            assert one_row(axial_attn(m, per_slice), m)[WITHIN] == 0.5

    def test_an_in_plane_scorer_is_found_by_the_within_slice_axis(self):
        m = stacked_bag()
        assert one_row(in_plane_oracle(m), m)[WITHIN] == 1.0


class TestTheLeadClaim:
    """H1: |flat - within_slice| < 0.02 -- the 3D metric is an in-plane metric."""

    def test_a_scorer_with_no_depth_information_still_reports_a_high_flat_auc(self):
        """The whole argument, in one fixture. This scorer is a fixed 256-number
        map; it cannot possibly know which slice holds the nodule, and its slice
        AUC says so. Its *flat* AUC is 0.995, which any reader would take as 3D
        localisation."""
        m = stacked_bag()
        row = one_row(in_plane_oracle(m), m)
        assert row[SLICE] == 0.5
        assert row[FLAT] > 0.99

    def test_flat_tracks_within_slice_and_not_the_slice_axis(self):
        m = stacked_bag()
        row = one_row(in_plane_oracle(m), m)
        assert abs(row[FLAT] - row[WITHIN]) < 0.02, "H1's threshold"
        assert abs(row[FLAT] - row[SLICE]) > 0.4


class TestOracle:
    def test_a_real_localiser_scores_high_on_all_three_axes(self):
        """The power half at the level of the whole decomposition: a scorer that
        genuinely finds *this* bag's lesion must not be flattened by any of the
        three columns."""
        rng = np.random.default_rng(0)
        m = stacked_bag()
        attn = (m > 0).astype(float) * 3.0 + rng.normal(0, 0.05, m.size)
        row = one_row(attn[None, :], m)
        assert row[FLAT] > 0.99
        assert row[SLICE] > 0.99
        assert row[WITHIN] > 0.99


class TestDegenerateBags:
    def test_an_all_negative_bag_is_dropped(self):
        m = np.zeros(3 * N_PATCH, dtype=np.int8)
        assert per_bag_axes([in_plane_oracle(m)], [m], 0) == []

    def test_an_all_positive_bag_is_dropped(self):
        m = np.ones(3 * N_PATCH, dtype=np.int8)
        assert per_bag_axes([in_plane_oracle(m)], [m], 0) == []

    def test_a_ragged_tail_is_dropped(self):
        """Length that is not a whole multiple of n_patch. Reshaping it would
        raise; truncating it would silently score a different bag than the one
        the uid names."""
        m = np.zeros(3 * N_PATCH + 10, dtype=np.int8)
        m[2 * N_PATCH + 5] = 1
        attn = np.random.default_rng(0).random((1, m.size))
        assert per_bag_axes([attn], [m], 0) == []

    def test_a_well_formed_bag_survives_alongside_them(self):
        """The paired half. A filter that dropped everything would pass all three
        tests above."""
        good = stacked_bag()
        bad = np.zeros(3 * N_PATCH, dtype=np.int8)
        rows = per_bag_axes([in_plane_oracle(bad), in_plane_oracle(good)],
                            [bad, good], 0)
        assert len(rows) == 1

    def test_the_returned_index_is_the_original_position_not_the_kept_one(self):
        """Load bearing: the caller maps this index into the uid array to get a
        patient for the cluster bootstrap. Renumbering after a drop would
        silently attribute every surviving bag to the wrong patient."""
        bad = np.zeros(3 * N_PATCH, dtype=np.int8)
        good = stacked_bag()
        rows = per_bag_axes([in_plane_oracle(bad), in_plane_oracle(bad),
                             in_plane_oracle(good)], [bad, bad, good], 0)
        assert [r[IDX] for r in rows] == [2]

    def test_a_bag_whose_every_slice_holds_a_nodule_nans_only_the_slice_axis(self):
        """Degenerate on one axis, fine on the others. Dropping the whole bag
        would shrink the flat and within-slice samples too, and the three columns
        would then be bootstrapped over different sets of patients while being
        printed side by side."""
        m = stacked_bag(n_slices=3, lesion_slices=(0, 1, 2))
        row = one_row(in_plane_oracle(m), m)
        assert np.isnan(row[SLICE])
        assert np.isfinite(row[FLAT]) and np.isfinite(row[WITHIN])

    def test_a_bag_with_a_usable_slice_axis_is_not_nanned(self):
        m = stacked_bag(n_slices=3, lesion_slices=(1,))
        assert np.isfinite(one_row(in_plane_oracle(m), m)[SLICE])


class TestBookkeeping:
    def test_slice_counts_are_reported_per_bag(self):
        m = stacked_bag(n_slices=8, lesion_slices=(2, 3, 4))
        row = one_row(in_plane_oracle(m), m)
        assert row[N_SLICES] == 8
        assert row[N_LESION_SLICES] == 3

    def test_the_named_slot_is_the_one_scored(self):
        """Multi-slot attention: scoring the wrong row of [K, N] is this
        project's most expensive past bug, so the axis is asserted."""
        m = stacked_bag()
        good = in_plane_oracle(m)[0]
        noise = np.random.default_rng(0).random(m.size)
        attn = np.stack([noise, good])
        assert one_row(attn, m, slot=1)[WITHIN] == 1.0
        assert one_row(attn, m, slot=0)[WITHIN] < 0.9


class TestNPatchIsConfigurable:
    def test_a_different_grid_size_changes_which_bags_are_whole(self):
        """256 is the DINOv2 ViT-B/14 grid, not a law. A 5x64 bag is ragged at
        the default (320 is not a multiple of 256) and whole at n_patch=64."""
        m = np.zeros(5 * 64, dtype=np.int8)
        m[64 + 3] = 1
        attn = np.random.default_rng(0).random((1, m.size))
        assert per_bag_axes([attn], [m], 0) == []
        assert len(per_bag_axes([attn], [m], 0, n_patch=64)) == 1

    def test_the_exact_chance_result_holds_at_another_grid_size(self):
        """The tie argument is about tiling, not about 256."""
        m = np.zeros(6 * 64, dtype=np.int8)
        for z in (1, 3):
            m[z * 64 + np.array([10, 11])] = 1
        v = np.random.default_rng(0).random(64)
        attn = np.tile(v, 6)[None, :]
        assert per_bag_axes([attn], [m], 0, n_patch=64)[0][SLICE] == 0.5

    def test_the_default_is_the_dinov2_grid(self):
        m = stacked_bag()
        assert one_row(in_plane_oracle(m), m) == one_row(
            in_plane_oracle(m), m, n_patch=256)
