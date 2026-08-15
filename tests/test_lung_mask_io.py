"""Tests for the lung-mask writer's HDF5 attribute handling.

These exist because of a bug that cost eight concurrent SLURM tasks a round of
downloads. The containment stats grew a nested ``contained_at`` mapping when the
threshold sweep was added; h5py attributes accept scalars and arrays but not
mappings, so every write in the production path raised. The sweep runs never
caught it, because ``--compare-methods`` skips the HDF5 write entirely -- the
selection path and the production path had quietly diverged, and only the one
nobody was exercising was broken.

So the rule pinned here is: whatever shape the stats dict grows into, it must
still be storable. A test that only checked today's keys would have missed this
one too, so the tests assert the *property*, not the current schema.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from scripts.lung_mask_lidc import containment, done_uids, write_attrs


@pytest.fixture
def group(tmp_path):
    with h5py.File(tmp_path / "t.h5", "w") as f:
        yield f.create_group("series")


class TestWriteAttrs:
    def test_nested_mappings_survive_a_round_trip(self):
        """The exact shape that broke the array job."""
        stats = {"contained": 1.0, "contained_at": {"0.0": 1.0, "0.5": 0.81}}
        with h5py.File("/dev/null", "w", driver="core", backing_store=False) as f:
            g = f.create_group("s")
            write_attrs(g, stats)
            assert json.loads(g.attrs["contained_at"])["0.5"] == pytest.approx(0.81)

    def test_none_becomes_a_sentinel_because_hdf5_has_no_null(self, group):
        write_attrs(group, {"contained": None})
        assert group.attrs["contained"] == -1.0

    def test_scalars_pass_through(self, group):
        write_attrs(group, {"n_lesion_patches": 42, "lung_frac": 0.243})
        assert group.attrs["n_lesion_patches"] == 42
        assert group.attrs["lung_frac"] == pytest.approx(0.243)

    def test_a_real_containment_dict_is_writable(self, group):
        """The property that matters: whatever shape containment() returns, the
        writer must handle it. Pinning the schema instead would let the two drift
        apart again."""
        rng = np.random.default_rng(0)
        lung = rng.random((4, 16, 16)).astype(np.float32)
        lesion = (rng.random((4, 16, 16)) < 0.01).astype(np.float32)
        write_attrs(group, containment(lung, lesion, lung_thresh=0.0))
        assert "contained" in group.attrs and "contained_at" in group.attrs

    def test_containment_with_no_lesions_is_writable(self, group):
        """~13% of LIDC series have no nodules, so this path runs often and its
        `contained` is None."""
        lung = np.ones((2, 16, 16), np.float32)
        stats = containment(lung, np.zeros((2, 16, 16), np.float32))
        assert stats["contained"] is None
        write_attrs(group, stats)
        assert group.attrs["contained"] == -1.0


class TestResumeCompleteness:
    """A group that exists is not a series that finished.

    The failed array wrote 828 groups holding valid masks but no containment
    attributes, because it crashed between the dataset write and the attribute
    write. Treating "group exists" as "done" would skip all 828 forever on
    resume, shipping a mask store whose gate could never be evaluated -- a silent
    hole, not a loud failure."""

    def test_a_complete_series_counts(self, tmp_path):
        p = tmp_path / "m.h5"
        with h5py.File(p, "w") as f:
            g = f.create_group("uid1")
            g.create_dataset("lung", data=np.zeros((2, 16, 16), np.float16))
            write_attrs(g, {"contained": 1.0})
        assert done_uids(str(p)) == {"uid1"}

    def test_a_half_written_series_does_not(self, tmp_path):
        """Exactly the state the crash left behind."""
        p = tmp_path / "m.h5"
        with h5py.File(p, "w") as f:
            g = f.create_group("uid1")
            g.create_dataset("lung", data=np.zeros((2, 16, 16), np.float16))
            g.attrs["method"] = "fill"          # written before the crash point
        assert done_uids(str(p)) == set()

    def test_an_empty_group_does_not(self, tmp_path):
        p = tmp_path / "m.h5"
        with h5py.File(p, "w") as f:
            f.create_group("uid1")
        assert done_uids(str(p)) == set()

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert done_uids(str(tmp_path / "nope.h5")) == set()


class TestContainment:
    def test_any_lung_is_more_permissive_than_a_majority_rule(self):
        """The finding that drove the amendment: a strict cut drops boundary
        patches, and subpleural nodules live in exactly those."""
        lung = np.full((1, 16, 16), 0.3, np.float32)   # every patch 30% lung
        lesion = np.zeros((1, 16, 16), np.float32)
        lesion[0, 8, 8] = 0.01
        c = containment(lung, lesion)
        assert c["contained_at"]["0.0"] == 1.0
        assert c["contained_at"]["0.5"] == 0.0

    def test_thresholds_are_monotone_non_increasing(self):
        rng = np.random.default_rng(1)
        lung = rng.random((3, 16, 16)).astype(np.float32)
        lesion = (rng.random((3, 16, 16)) < 0.05).astype(np.float32)
        c = containment(lung, lesion)
        vals = [c["contained_at"][k] for k in sorted(c["contained_at"], key=float)]
        assert all(a >= b for a, b in zip(vals, vals[1:]))
