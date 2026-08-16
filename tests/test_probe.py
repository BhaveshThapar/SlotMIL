"""H7's numerator, and the one property that makes it a gate rather than a number.

The pre-registration says ``score_coverage: every_patch_of_every_bag``. That
clause is doing real work: the probe's skill is a ratio whose numerator is the
probe's AUC and whose denominator is a template's AUC over the same bags, and if
the numerator is computed over a subsampled patch population while the
denominator is not, the ratio is a different quantity wearing H7's name. The old
``probe_ceiling.py`` subsampled at score time -- 20 negatives per positive -- so
the assertion here is the difference between the two instruments, not a nicety.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from slotmil.eval.axes import per_bag_axes
from slotmil.eval.estimands import prior_normalised_skill
from slotmil.eval.nulls import global_template, template_scores
from slotmil.eval.probe import collect_fit_set, fit_probe, score_bags

N_SERIES = 6
N_SLICES = 3
GRID = 4
N_PATCH = GRID * GRID   # 16, so a bag is a whole number of "slices"
DIM = 5


@pytest.fixture
def cache(tmp_path):
    """A tiny cache laid out like the real one, with a learnable lesion signal.

    Lesion patches carry a constant offset on dimension 0, so a logistic probe
    can find them and a template cannot -- the separation H7 is asking about.
    """
    rng = np.random.default_rng(0)
    path = tmp_path / "cache.h5"
    with h5py.File(path, "w") as f:
        for i in range(N_SERIES):
            g = f.create_group(f"uid{i:02d}")
            mask = np.zeros((N_SLICES, GRID, GRID), dtype=np.float32)
            # One lesion patch per slice, walked around so it is not a fixed
            # position -- a positional template must not be able to learn it.
            for s in range(N_SLICES):
                mask[s, (i + s) % GRID, (i + 2 * s) % GRID] = 0.03
            feats = rng.standard_normal((N_SLICES, N_PATCH, DIM)).astype(np.float32)
            feats[..., 0] += 4.0 * (mask.reshape(N_SLICES, -1) > 0)
            g.create_dataset("features", data=feats)
            g.create_dataset("mask", data=mask)
            g.attrs["grid_h"] = GRID
            g.attrs["grid_w"] = GRID
            g.attrs["label"] = 1
    return str(path)


@pytest.fixture
def uids():
    return [f"uid{i:02d}" for i in range(N_SERIES)]


class TestScoreCoverage:
    """``score_coverage: every_patch_of_every_bag``, asserted rather than declared."""

    def test_every_patch_of_every_bag_is_scored(self, cache, uids):
        with h5py.File(cache, "r") as f:
            clf = fit_probe(f, uids, neg_per_pos=2, seed=0)
            attns, masks, scored = score_bags(clf, f, uids)
        assert len(scored) == N_SERIES
        for a, m in zip(attns, masks):
            assert a.shape == (1, N_SLICES * N_PATCH)
            assert m.shape == (N_SLICES * N_PATCH,)

    def test_the_scored_population_ignores_the_fit_subsample(self, cache, uids):
        """Changing ``neg_per_pos`` changes the classifier, never the patch count.

        This is the property the old driver did not have: there, the same knob
        set the score-time population too, so the estimand moved with a fitting
        choice."""
        counts = []
        for neg_per_pos in (1, 2, 8):
            with h5py.File(cache, "r") as f:
                clf = fit_probe(f, uids, neg_per_pos=neg_per_pos, seed=0)
                attns, _, _ = score_bags(clf, f, uids)
            counts.append([a.shape[1] for a in attns])
        assert counts[0] == counts[1] == counts[2]

    def test_the_fit_set_does_shrink_with_the_subsample(self, cache, uids):
        """The complement of the test above: if the fit set were also invariant,
        the previous assertion would be passing against a constant."""
        with h5py.File(cache, "r") as f:
            small, _ = collect_fit_set(f, uids, 1, np.random.default_rng(0))
            large, _ = collect_fit_set(f, uids, 8, np.random.default_rng(0))
        assert large.shape[0] > small.shape[0]

    def test_the_fit_set_owns_its_memory(self, cache, uids):
        """Not a style point -- this is the defect that OOM-killed a 20 GB node.

        A basic index into an HDF5-read block returns a *view*, so holding one
        patch holds the entire series it came from; 608 lesion-bearing training
        series at up to 275 MB each is ~55 GB. Advanced indexing copies. The old
        driver's 60-scan cap kept it under the cliff and out of sight."""
        with h5py.File(cache, "r") as f:
            X, _ = collect_fit_set(f, uids, 2, np.random.default_rng(0))
        assert X.base is None
        assert X.nbytes == X.shape[0] * X.shape[1] * 4


class TestOwnDenominator:
    """``probe_denominator: own_fitted_inplane_template`` -- fit on val, per scorer."""

    def test_a_real_probe_clears_the_gate(self, cache, uids):
        with h5py.File(cache, "r") as f:
            clf = fit_probe(f, uids, neg_per_pos=2, seed=0)
            attns, masks, _ = score_bags(clf, f, uids)
        skill = _skill(attns, masks)
        assert skill > 0.50, skill

    def test_a_content_free_scorer_scores_zero_against_itself(self, cache, uids):
        """``fitted_template`` scores exactly 0 against its own fitted template.

        The config leans on this: it is why the content-free set is never empty.
        A tiled template is constant across slices, so the template fit to it is
        that same map and the two AUCs coincide."""
        with h5py.File(cache, "r") as f:
            _, masks, _ = score_bags(
                fit_probe(f, uids, neg_per_pos=2, seed=0), f, uids)
        tmpl = global_template(
            [np.tile(np.arange(N_PATCH, dtype=np.float32), N_SLICES)[None, :]
             for _ in masks], masks, slot=0, n_patch=N_PATCH)
        scores = template_scores(masks, tmpl, n_patch=N_PATCH)
        assert _skill(scores, masks) == pytest.approx(0.0, abs=1e-9)

    def test_a_constant_scorer_is_degenerate_not_skilful(self, cache, uids):
        """All-ties gives AUC 0.5 on both halves of the ratio, hence skill 0."""
        with h5py.File(cache, "r") as f:
            _, masks, _ = score_bags(
                fit_probe(f, uids, neg_per_pos=2, seed=0), f, uids)
        flat = [np.ones((1, len(m)), dtype=np.float32) for m in masks]
        assert _skill(flat, masks) == pytest.approx(0.0, abs=1e-9)


class TestDeterminism:
    def test_seed_zero_reproduces(self, cache, uids):
        out = []
        for _ in range(2):
            with h5py.File(cache, "r") as f:
                clf = fit_probe(f, uids, neg_per_pos=2, seed=0)
                attns, _, _ = score_bags(clf, f, uids)
            out.append(np.concatenate([a.ravel() for a in attns]))
        assert np.array_equal(out[0], out[1])

    def test_a_different_seed_moves_the_fit(self, cache, uids):
        with h5py.File(cache, "r") as f:
            a, _ = collect_fit_set(f, uids, 2, np.random.default_rng(0))
            b, _ = collect_fit_set(f, uids, 2, np.random.default_rng(1))
        assert not np.array_equal(a, b)


class TestRefusals:
    def test_an_all_negative_split_raises(self, tmp_path):
        path = tmp_path / "empty.h5"
        with h5py.File(path, "w") as f:
            g = f.create_group("uid00")
            g.create_dataset("features",
                             data=np.zeros((N_SLICES, N_PATCH, DIM), dtype=np.float32))
            g.create_dataset("mask",
                             data=np.zeros((N_SLICES, GRID, GRID), dtype=np.float32))
            g.attrs["grid_h"] = GRID
            g.attrs["grid_w"] = GRID
        with h5py.File(path, "r") as f, pytest.raises(ValueError, match="no lesion"):
            fit_probe(f, ["uid00"], neg_per_pos=2, seed=0)

    def test_a_mask_that_disagrees_with_the_features_raises(self, tmp_path, cache, uids):
        """Silently mis-scoring is the failure mode; raising is the requirement."""
        with h5py.File(cache, "r") as f:
            clf = fit_probe(f, uids, neg_per_pos=2, seed=0)
        bad = tmp_path / "bad.h5"
        with h5py.File(bad, "w") as f:
            g = f.create_group("uid00")
            g.create_dataset("features",
                             data=np.zeros((N_SLICES, N_PATCH, DIM), dtype=np.float32))
            g.create_dataset("mask",
                             data=np.zeros((N_SLICES + 1, GRID, GRID), dtype=np.float32))
            g.attrs["grid_h"] = GRID
            g.attrs["grid_w"] = GRID
        with h5py.File(bad, "r") as f, pytest.raises(ValueError, match="patches"):
            score_bags(clf, f, ["uid00"])


def _skill(attns, masks) -> float:
    """Flat AUC over its own val-fitted in-plane template, the H7 convention."""
    rows = per_bag_axes(attns, masks, 0, n_patch=N_PATCH)
    numer = float(np.mean([r[1] for r in rows]))
    tmpl = global_template(attns, masks, slot=0, n_patch=N_PATCH)
    denom_rows = per_bag_axes(template_scores(masks, tmpl, n_patch=N_PATCH),
                              masks, 0, n_patch=N_PATCH)
    return prior_normalised_skill(numer, float(np.mean([r[1] for r in denom_rows])))
