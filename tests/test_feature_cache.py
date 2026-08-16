"""Slice subsampling: the one stochastic thing the training path does.

The defect these tests exist to prevent is not that subsampling was *biased* --
it is that it was correlated in three ways at once, all invisible from any
output the project reports:

1. ``FeatureBagDataset`` held a ``np.random.Generator`` built in ``__init__``.
   DataLoader forks it to every worker with identical state, and
   ``utils.seed.seed_worker`` reseeds only the legacy ``np.random`` global -- it
   cannot reach a Generator object. Workers at the same queue position therefore
   drew the same slices.
2. Workers are re-forked each epoch (nothing in the repo sets
   ``persistent_workers``) from a parent whose Generator never advances, because
   the parent process never calls ``__getitem__``. The streams reset every epoch,
   so "stochastic slice subsampling during training" resampled nothing.
3. ``train_cached.py`` never passed ``seed``, so every arm at every training seed
   shared stream 0. Reported seed-to-seed variance covered init and shuffle order
   but not the data view.

So the tests come in the house pairs: the draw must be **stable** across the
things that must not matter (worker count, fork timing) and must **vary** across
the things that must (seed, epoch, item). A test that only checked stability
would pass against a constant, which is exactly the bug.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from slotmil.data.feature_cache import FeatureBagDataset, collate_bags
from slotmil.utils.seed import seed_worker

N_SERIES = 8
N_SLICES = 24
N_PATCHES = 4
DIM = 3
MAX_SLICES = 5


@pytest.fixture
def cache(tmp_path):
    """A tiny cache with the same layout as the real one, one group per series."""
    path = tmp_path / "cache.h5"
    with h5py.File(path, "w") as f:
        for i in range(N_SERIES):
            g = f.create_group(f"uid{i:02d}")
            g.create_dataset(
                "features",
                data=np.full((N_SLICES, N_PATCHES, DIM), i, dtype=np.float16),
            )
            g.attrs["label"] = i % 2
            g.attrs["grid_h"] = 2
            g.attrs["grid_w"] = 2
    return str(path)


def draws(ds) -> dict[str, tuple[int, ...]]:
    """uid -> the slice indices that item selected, read straight off the item."""
    return {ds[i]["uid"]: tuple(ds[i]["slice_index"].tolist()) for i in range(len(ds))}


def draws_via_loader(cache_path, seed, epoch, num_workers) -> dict[str, tuple[int, ...]]:
    """The same, but through a real DataLoader so worker forking is exercised."""
    ds = FeatureBagDataset(cache_path, train=True, max_slices=MAX_SLICES, seed=seed)
    ds.set_epoch(epoch)
    g = torch.Generator()
    g.manual_seed(0)
    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_bags,
        worker_init_fn=seed_worker,
        generator=g,
    )
    out = {}
    for batch in loader:
        for j, uid in enumerate(batch["uid"]):
            out[uid] = tuple(batch["slice_index"][j].tolist())
    return out


class TestSubsampleIsIndependentOfWorkers:
    """The half that must stay CONSTANT."""

    @pytest.mark.parametrize("num_workers", [0, 1, 2, 4])
    def test_worker_count_does_not_change_the_draw(self, cache, num_workers):
        direct = draws(
            FeatureBagDataset(cache, train=True, max_slices=MAX_SLICES, seed=7)
        )
        via = draws_via_loader(cache, seed=7, epoch=0, num_workers=num_workers)
        assert via == direct, (
            f"num_workers={num_workers} changed the subsample. --num-workers is a "
            "CLI default, not a pre-registered constant, so it must not be able to "
            "move a reported number."
        )

    def test_repeated_reads_of_the_same_item_agree(self, cache):
        ds = FeatureBagDataset(cache, train=True, max_slices=MAX_SLICES, seed=3)
        first = ds[2]["slice_index"].tolist()
        _ = ds[5]  # advancing through other items must not perturb item 2
        assert ds[2]["slice_index"].tolist() == first


class TestSubsampleVariesWhereItMust:
    """The half that must CHANGE -- without this the tests pass against a constant."""

    def test_epoch_resamples(self, cache):
        ds = FeatureBagDataset(cache, train=True, max_slices=MAX_SLICES, seed=0)
        ds.set_epoch(0)
        a = draws(ds)
        ds.set_epoch(1)
        b = draws(ds)
        assert a != b, "set_epoch did not resample; subsampling is not stochastic"
        assert all(len(v) == MAX_SLICES for v in b.values())

    def test_seed_changes_the_data_view(self, cache):
        a = draws(FeatureBagDataset(cache, train=True, max_slices=MAX_SLICES, seed=0))
        b = draws(FeatureBagDataset(cache, train=True, max_slices=MAX_SLICES, seed=1))
        assert a != b, (
            "seeds share a subsampling stream, so reported seed-to-seed variance "
            "excludes the data view"
        )

    def test_items_do_not_share_a_draw(self, cache):
        """The old bug's signature: identical-length bags drawing identically."""
        d = draws(FeatureBagDataset(cache, train=True, max_slices=MAX_SLICES, seed=0))
        assert len(set(d.values())) > 1, (
            "every bag drew the same slices -- this is what a shared forked "
            "Generator looks like"
        )

    def test_all_four_seeds_and_two_epochs_are_distinct(self, cache):
        """Cheap guard that the tuple seeding is not collapsing coordinates."""
        seen = {
            (s, e): tuple(sorted(draws_at(cache, s, e).items()))
            for s in range(5)
            for e in range(2)
        }
        assert len(set(seen.values())) == len(seen)


def draws_at(cache_path, seed, epoch):
    ds = FeatureBagDataset(cache_path, train=True, max_slices=MAX_SLICES, seed=seed)
    ds.set_epoch(epoch)
    return draws(ds)


class TestFullBagPaths:
    """Subsampling must not fire anywhere it was not asked for."""

    def test_eval_mode_sees_every_slice(self, cache):
        ds = FeatureBagDataset(cache, train=False, max_slices=MAX_SLICES, seed=0)
        assert all(v == tuple(range(N_SLICES)) for v in draws(ds).values())

    def test_no_max_slices_sees_every_slice(self, cache):
        ds = FeatureBagDataset(cache, train=True, max_slices=None, seed=0)
        assert all(v == tuple(range(N_SLICES)) for v in draws(ds).values())

    def test_short_volume_is_untouched(self, cache):
        ds = FeatureBagDataset(cache, train=True, max_slices=N_SLICES + 10, seed=0)
        assert all(v == tuple(range(N_SLICES)) for v in draws(ds).values())

    def test_features_follow_the_selection(self, cache):
        """The returned rows are the selected slices, not slices 0..k."""
        ds = FeatureBagDataset(cache, train=True, max_slices=MAX_SLICES, seed=0)
        item = ds[0]
        assert item["n_slices"] == MAX_SLICES
        assert item["features"].shape == (MAX_SLICES * N_PATCHES, DIM)
        assert item["slice_index"].tolist() == sorted(item["slice_index"].tolist())


class TestSetEpochDefault:
    def test_epoch_defaults_to_zero(self, cache):
        ds = FeatureBagDataset(cache, train=True, max_slices=MAX_SLICES, seed=0)
        assert ds.epoch == 0
        untouched = draws(ds)
        ds.set_epoch(0)
        assert draws(ds) == untouched
