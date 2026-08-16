"""Factorised content-free references: the axis each one is allowed to use.

These come in the house pairs, and here the pairing is unusually load-bearing
because the module's whole purpose is a *dissociation*. Each reference must be
shown to carry information on its own axis **and** to carry exactly none on the
other. A test that only checked the first half would pass against a reference
that quietly reads both, which is precisely the confusion the module exists to
resolve: Harvey et al. measure the axial axis alone and Arun et al. the in-plane
axis alone, and neither can say which is carrying the score.

Two of the assertions here are exact rather than approximate, and they are the
paper's central claim in test form:

* the in-plane template's **slice AUC is exactly 0.5** -- it is one map tiled
  identically across every slice, so every slice's mean is the same number and
  every pair ties;
* the axial template's **within-slice AUC is exactly 0.5** -- it is constant
  within each slice, for the mirror reason.

If either drifts off 0.5 the reference is reading an axis it was built not to
see, and every skill number computed against it is wrong.

The remaining guard is content-freedom: a reference fit on validation must give
a test bag the same scores no matter what is *in* that bag. The mask-fitted
family reads labels on the fit split, so this is the property that keeps it a
legitimate baseline rather than an oracle leaking into test.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from slotmil.eval.nulls import global_template
from slotmil.eval.templates import (
    DEFAULT_DEPTH_BINS,
    depth_bins_for,
    fit_family,
    inplane_template,
    joint_template,
    lesion_rows,
    score_family,
    separable_scores,
)

NP_ = 16  # small n_patch so the fixtures stay readable
N_BINS = 4


def make_bags(n_bags=6, n_slices=8, n_patch=NP_, seed=0):
    """Bags whose lesions sit centrally in-plane AND centrally in depth.

    Both priors are present, so a reference restricted to one axis has something
    real to find on it -- without that the "carries none on the other axis" half
    of each pair would be vacuous.
    """
    rng = np.random.default_rng(seed)
    masks, attns = [], []
    for _ in range(n_bags):
        m = np.zeros((n_slices, n_patch), dtype=np.float32)
        mid_s = n_slices // 2
        for s in range(mid_s - 1, mid_s + 1):
            m[s, n_patch // 2] = 1.0
        masks.append(m.ravel())
        attns.append(rng.random((3, n_slices * n_patch)).astype(np.float32))
    return attns, masks


def axes(scores, masks, n_patch=NP_):
    """Per-bag flat / slice / within-slice AUC, averaged. Mirrors eval.axes."""
    fl, sl, wi = [], [], []
    for a, m in zip(scores, masks):
        t = (m > 0).astype(np.int8)
        if t.sum() in (0, len(t)):
            continue
        ns = len(m) // n_patch
        sc = a[0][: ns * n_patch].reshape(ns, n_patch)
        st = t[: ns * n_patch].reshape(ns, n_patch)
        fl.append(roc_auc_score(st.ravel(), sc.ravel()))
        has = (st.sum(1) > 0).astype(np.int8)
        if 0 < has.sum() < ns:
            sl.append(roc_auc_score(has, sc.mean(1)))
        sel = has.astype(bool)
        sub_t, sub_s = st[sel].ravel(), sc[sel].ravel()
        if 0 < sub_t.sum() < len(sub_t):
            wi.append(roc_auc_score(sub_t, sub_s))
    return (float(np.mean(fl)), float(np.mean(sl)) if sl else np.nan,
            float(np.mean(wi)) if wi else np.nan)


@pytest.fixture
def fitted():
    attns, masks = make_bags()
    fam = fit_family(masks, source="masks", n_patch=NP_, n_bins=N_BINS)
    return fam, masks


class TestAxisDissociation:
    """The exact half. These two numbers are 0.5 by construction, not by luck."""

    def test_inplane_template_has_no_axial_information(self, fitted):
        fam, masks = fitted
        sc = score_family(masks, fam)
        _, slice_auc, within = axes(sc["inplane"], masks)
        assert slice_auc == 0.5, (
            f"in-plane template scored {slice_auc} on the slice axis. It is one "
            "map tiled identically per slice, so every slice mean ties; anything "
            "but exactly 0.5 means it is reading depth."
        )
        assert within > 0.5, "and it must still carry real in-plane information"

    def test_axial_template_has_no_in_plane_information(self, fitted):
        fam, masks = fitted
        sc = score_family(masks, fam)
        _, slice_auc, within = axes(sc["axial"], masks)
        assert within == 0.5, (
            f"axial template scored {within} within slices. It is constant "
            "within each slice, so every in-plane pair ties."
        )
        assert slice_auc > 0.5, "and it must still carry real axial information"

    def test_the_two_are_a_dissociation_not_a_pair_of_null_scorers(self, fitted):
        """Guards against both halves passing because nothing scores anywhere.

        The axial bar is looser than the in-plane one on purpose: the fixture has
        8 slices binned into 4, so the axial reference cannot resolve which of a
        bin's two slices holds the lesion and tops out below 1.0. Coarseness is
        the expected cost of normalised-depth binning, not a defect."""
        fam, masks = fitted
        sc = score_family(masks, fam)
        assert axes(sc["inplane"], masks)[2] > 0.9
        assert axes(sc["axial"], masks)[1] > 0.8


class TestContentFreedom:
    """A reference must score a test bag without reading it."""

    def test_scores_ignore_what_is_in_the_bag(self, fitted):
        fam, masks = fitted
        original = score_family(masks, fam)
        # Same lengths, entirely different lesion content.
        rng = np.random.default_rng(1)
        scrambled = [rng.permutation(m) for m in masks]
        after = score_family(scrambled, fam)
        for key in original:
            for a, b in zip(original[key], after[key]):
                assert np.array_equal(a, b), (
                    f"{key} changed when the test bag's content changed -- it is "
                    "not content-free and cannot serve as a baseline"
                )

    def test_every_bag_of_a_given_length_gets_the_same_scores(self, fitted):
        fam, masks = fitted
        sc = score_family(masks, fam)
        for key, rows in sc.items():
            first = rows[0]
            assert all(np.array_equal(first, r) for r in rows), key


class TestAgreementWithTheDeclaredReference:
    def test_inplane_reproduces_nulls_global_template(self):
        """The pre-registration declares `nulls.global_template` as
        `fitted_template`. If this family's in-plane cell were a different
        object, every skill number in the paper would have two denominators."""
        attns, masks = make_bags()
        mine = inplane_template([a[0] for a in attns], masks, n_patch=NP_).values
        theirs = np.asarray(global_template(attns, masks, slot=0, n_patch=NP_),
                            dtype=np.float64)
        # global_template casts its return to float32; agreement is to that
        # precision, not bit-for-bit, and claiming otherwise would be wrong.
        assert np.allclose(mine, theirs, rtol=1e-6, atol=0)


class TestSeparable:
    def test_is_exactly_the_outer_product(self, fitted):
        fam, masks = fitted
        sep = separable_scores(masks, fam["axial"], fam["inplane"])[0][0]
        ns = len(masks[0]) // NP_
        bins = depth_bins_for(ns, fam["axial"].n_bins)
        expect = (fam["axial"].values[bins][:, None]
                  * fam["inplane"].values[None, :]).ravel()
        assert np.allclose(sep, expect.astype(np.float32), rtol=1e-6)

    def test_rejects_a_swapped_argument_order(self, fitted):
        fam, masks = fitted
        with pytest.raises(ValueError, match="need \\(axial, inplane\\)"):
            separable_scores(masks, fam["inplane"], fam["axial"])


class TestDepthBins:
    @pytest.mark.parametrize("ns", [1, 2, 3, 7, 8, 31, 32, 33, 171, 700])
    def test_bins_stay_in_range_and_are_monotone(self, ns):
        b = depth_bins_for(ns, DEFAULT_DEPTH_BINS)
        assert b.shape == (ns,)
        assert b.min() >= 0 and b.max() < DEFAULT_DEPTH_BINS
        assert np.all(np.diff(b) >= 0)

    def test_the_last_slice_never_falls_off_the_end(self):
        """`j * n_bins // ns` reaches n_bins exactly when ns divides j*n_bins;
        unclipped that would index past the fitted profile entirely.

        For a volume with at least as many slices as bins the last slice lands in
        the top bin; for a shorter one it lands lower, which is correct -- a
        3-slice volume occupies 3 bins, not 32."""
        for ns in range(1, 200):
            last = depth_bins_for(ns, DEFAULT_DEPTH_BINS)[-1]
            assert last <= DEFAULT_DEPTH_BINS - 1
            if ns >= DEFAULT_DEPTH_BINS:
                assert last == DEFAULT_DEPTH_BINS - 1

    def test_a_long_volume_uses_every_bin(self):
        assert len(set(depth_bins_for(171, 32).tolist())) == 32

    def test_a_short_volume_leaves_bins_empty_rather_than_erroring(self):
        b = depth_bins_for(3, 32)
        assert len(set(b.tolist())) == 3

    @pytest.mark.parametrize("bad", [0, -1])
    def test_degenerate_arguments_raise(self, bad):
        with pytest.raises(ValueError):
            depth_bins_for(bad, 4)
        with pytest.raises(ValueError):
            depth_bins_for(4, bad)


class TestFitSources:
    def test_the_two_sources_give_different_fits(self):
        """If they agreed, the mask-fitted oracle would be measuring nothing the
        attention-fitted one does not already say, and the confound it exists to
        remove would not exist."""
        attns, masks = make_bags()
        by_attn = fit_family(masks, attns=attns, slot=0, source="attention",
                             n_patch=NP_, n_bins=N_BINS)
        by_mask = fit_family(masks, source="masks", n_patch=NP_, n_bins=N_BINS)
        assert not np.allclose(by_attn["inplane"].values, by_mask["inplane"].values)

    def test_attention_source_needs_its_arguments(self):
        _, masks = make_bags()
        with pytest.raises(ValueError, match="needs attns and slot"):
            fit_family(masks, source="attention")

    def test_an_unknown_source_raises(self):
        _, masks = make_bags()
        with pytest.raises(ValueError, match="must be 'attention' or 'masks'"):
            fit_family(masks, source="lesions")

    def test_lesion_rows_binarise(self):
        _, masks = make_bags()
        rows = lesion_rows(masks)
        assert set(np.unique(rows[0]).tolist()) <= {0.0, 1.0}


class TestFitBookkeeping:
    def test_unobserved_cells_are_counted_not_hidden(self):
        """More bins than slices leaves cells the fit never saw. They fall back
        to the global mean, and the count says so rather than the fit silently
        presenting a guess as an estimate."""
        attns, masks = make_bags(n_slices=3)
        fit = joint_template(lesion_rows(masks), masks, n_patch=NP_, n_bins=16)
        assert fit.n_unobserved > 0
        assert np.isfinite(fit.values).all()

    def test_a_fully_observed_fit_reports_zero(self, fitted):
        fam, _ = fitted
        assert fam["joint"].n_unobserved == 0
        assert fam["axial"].n_unobserved == 0

    def test_a_ragged_bag_contributes_its_whole_slices(self):
        """Matches nulls.global_template rather than skipping the bag; a silent
        divergence would make the four references incomparable."""
        attns, masks = make_bags(n_bags=2)
        ragged = [m[: len(m) - 3] for m in masks]
        fit = inplane_template(lesion_rows(ragged), ragged, n_patch=NP_)
        assert fit.n_bags == 2

    def test_a_bag_shorter_than_one_slice_cannot_be_fit(self):
        short = [np.zeros(NP_ - 1, dtype=np.float32)]
        with pytest.raises(ValueError, match="whole number of slices"):
            inplane_template(lesion_rows(short), short, n_patch=NP_)

    def test_shapes_are_what_the_family_table_says(self, fitted):
        fam, _ = fitted
        assert fam["inplane"].values.shape == (NP_,)
        assert fam["axial"].values.shape == (N_BINS,)
        assert fam["joint"].values.shape == (N_BINS, NP_)
