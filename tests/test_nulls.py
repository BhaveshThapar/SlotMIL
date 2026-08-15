"""Tests for the reference baselines.

These are the paper's instrument, so the properties that make each one a valid
null are pinned here rather than assumed. Two are load-bearing enough to name:

``roll_masks`` must preserve per-bag prevalence *exactly*. If it does not, its
score stops being the metric's floor and becomes a floor plus a prevalence
artefact -- and the whole argument rests on that number being clean.

``shuffle_masks_across_bags`` must be a derangement, not a shuffle. A plain
permutation leaves some bags matched with themselves, which pulls the
cross-patient null toward the real score and understates how much of the signal
is population anatomy.

The entropy convention is the third: slot attention normalises over *slots*, and
measuring the wrong axis is exactly the error that produced this project's first
reversal, so the axis is asserted rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from slotmil.eval.nulls import (
    GRID,
    N_PATCH,
    binary_findings,
    centre_prior_scores,
    global_template,
    match_scale,
    random_attn,
    roll_masks,
    shuffle_masks_across_bags,
    slot_entropy,
    template_scores,
)


def bags(n_bags=6, n_slices=3, seed=0):
    rng = np.random.default_rng(seed)
    masks = []
    for _ in range(n_bags):
        m = np.zeros(n_slices * N_PATCH, dtype=np.int8)
        m[rng.choice(m.size, 5, replace=False)] = 1
        masks.append(m)
    return masks


class TestRollMasks:
    def test_prevalence_is_preserved_exactly(self):
        """The property that makes this the metric's floor rather than a floor
        plus a prevalence artefact."""
        rng = np.random.default_rng(0)
        ms = bags()
        for before, after in zip(ms, roll_masks(ms, rng)):
            assert before.sum() == after.sum()

    def test_the_target_actually_moves(self):
        rng = np.random.default_rng(0)
        ms = bags(n_slices=8)
        rolled = roll_masks(ms, rng)
        assert any(not np.array_equal(a, b) for a, b in zip(ms, rolled))

    def test_shifts_by_patches_not_whole_slices(self):
        """Deliberate, and the opposite of what looks tidy. A whole-slice shift
        would leave every lesion's in-plane position untouched, so an in-plane
        positional prior would still score against the rolled target and the
        'floor' would come out at the prior's value rather than at chance."""
        rng = np.random.default_rng(0)
        m = np.zeros(4 * N_PATCH, dtype=np.int8)
        m[N_PATCH + 7] = 1
        offsets = {int(np.flatnonzero(roll_masks([m], rng)[0])[0]) % N_PATCH
                   for _ in range(40)}
        assert len(offsets) > 1, "in-plane position must move, not just the slice"

    def test_a_single_slice_bag_does_not_crash(self):
        rng = np.random.default_rng(0)
        m = np.zeros(N_PATCH, dtype=np.int8)
        m[3] = 1
        assert roll_masks([m], rng)[0].sum() == 1


class TestShuffleAcrossBags:
    def test_is_a_derangement(self):
        """No bag keeps its own mask. A plain shuffle would leave some matched
        with themselves and inflate the cross-patient null."""
        rng = np.random.default_rng(0)
        ms = bags(n_bags=20)
        for original, swapped in zip(ms, shuffle_masks_across_bags(ms, rng)):
            assert not np.array_equal(original, swapped)

    def test_is_a_permutation_not_a_resample(self):
        rng = np.random.default_rng(0)
        ms = bags(n_bags=12)
        got = shuffle_masks_across_bags(ms, rng)
        assert sorted(m.tobytes() for m in got) == sorted(m.tobytes() for m in ms)

    def test_donor_masks_are_fitted_to_the_recipient_length(self):
        """LIDC series run 58 to 700 slices, so donor and recipient rarely match.
        Returning the donor's own length would not raise -- it would silently
        mis-score against a target of the wrong shape."""
        rng = np.random.default_rng(0)
        ms = [np.ones(3 * N_PATCH, np.int8), np.ones(7 * N_PATCH, np.int8),
              np.ones(5 * N_PATCH, np.int8)]
        got = shuffle_masks_across_bags(ms, rng)
        assert [len(g) for g in got] == [len(m) for m in ms]

    def test_a_single_bag_is_returned_unchanged(self):
        """Nothing to swap with -- must not spin for 1000 retries or crash."""
        ms = bags(n_bags=1)
        assert np.array_equal(shuffle_masks_across_bags(ms, np.random.default_rng(0))[0],
                              ms[0])

    def test_does_not_alias_the_inputs(self):
        rng = np.random.default_rng(0)
        ms = bags(n_bags=5)
        got = shuffle_masks_across_bags(ms, rng)
        got[0][:] = 0
        assert any(m.sum() > 0 for m in ms)


class TestSlotEntropy:
    def test_uniform_attention_has_maximum_entropy(self):
        k, n = 8, 100
        assert slot_entropy([np.full((k, n), 1.0 / k)]) == pytest.approx(np.log(k))

    def test_one_hot_attention_has_zero_entropy(self):
        a = np.zeros((8, 50)); a[3] = 1.0
        assert slot_entropy([a]) == pytest.approx(0.0, abs=1e-6)

    def test_entropy_is_measured_over_slots_not_instances(self):
        """The axis that produced this project's first reversal. Attention that is
        one-hot over slots but uniform over instances must read as zero entropy,
        not maximum."""
        k, n = 4, 64
        a = np.zeros((k, n)); a[1] = 1.0          # every instance picks slot 1
        assert slot_entropy([a]) == pytest.approx(0.0, abs=1e-6)


class TestMatchScale:
    def test_recovers_the_entropy_it_was_asked_for(self):
        k = 8
        for target in (0.5, 1.0, 1.5):
            scale = match_scale(target, k)
            got = slot_entropy(random_attn([4096], k, scale,
                                           np.random.default_rng(0)))
            assert got == pytest.approx(target, abs=0.05)

    def test_is_deterministic(self):
        assert match_scale(1.0, 8) == match_scale(1.0, 8)

    def test_larger_scale_means_lower_entropy(self):
        k = 8
        lo = slot_entropy(random_attn([2048], k, 0.5, np.random.default_rng(0)))
        hi = slot_entropy(random_attn([2048], k, 5.0, np.random.default_rng(0)))
        assert hi < lo


class TestRandomAttn:
    def test_normalises_over_slots(self):
        a = random_attn([50], 8, 1.0, np.random.default_rng(0))[0]
        np.testing.assert_allclose(a.sum(axis=0), 1.0, rtol=1e-5)

    def test_slot_bias_persists_across_bags(self):
        """A model with a global slot preference but no image knowledge -- without
        this the random null is *too* null to be a fair comparison."""
        bias = np.array([5.0, 0, 0, 0])
        out = random_attn([200, 200], 4, 0.1, np.random.default_rng(0), slot_bias=bias)
        assert all(a[0].mean() > 0.8 for a in out)

    def test_shapes_follow_the_lengths(self):
        out = random_attn([10, 25], 6, 1.0, np.random.default_rng(0))
        assert [a.shape for a in out] == [(6, 10), (6, 25)]


class TestContentFreeReferences:
    def test_centre_prior_peaks_at_the_middle_of_the_grid(self):
        m = np.zeros(N_PATCH, dtype=np.int8)
        s = centre_prior_scores([m])[0][0]
        centre = (GRID // 2) * GRID + GRID // 2
        assert s[centre] > s[0] and s[centre] > s[-1]

    def test_centre_prior_never_reads_the_mask(self):
        """It is model-free and target-free; two different targets of the same
        length must produce identical scores."""
        a = np.zeros(2 * N_PATCH, dtype=np.int8); a[5] = 1
        b = np.zeros(2 * N_PATCH, dtype=np.int8); b[500] = 1
        np.testing.assert_array_equal(centre_prior_scores([a])[0],
                                      centre_prior_scores([b])[0])

    def test_template_is_256_numbers(self):
        ms = bags(n_slices=3)
        attns = [np.random.default_rng(i).random((4, m.size)).astype(np.float32)
                 for i, m in enumerate(ms)]
        assert global_template(attns, ms, slot=0).shape == (N_PATCH,)

    def test_template_scores_are_identical_across_bags(self):
        """The whole claim about this baseline: at test time it cannot read the
        image, so every bag of the same length gets the same map."""
        t = np.arange(N_PATCH, dtype=np.float32)
        a, b = template_scores([np.zeros(2 * N_PATCH, np.int8),
                                np.zeros(2 * N_PATCH, np.int8)], t)
        np.testing.assert_array_equal(a, b)

    def test_template_recovers_a_constant_attention_map(self):
        pattern = np.random.default_rng(0).random(N_PATCH).astype(np.float32)
        ms = bags(n_bags=4, n_slices=3)
        attns = [np.tile(pattern, 3)[None, :].astype(np.float32) for _ in ms]
        np.testing.assert_allclose(global_template(attns, ms, 0), pattern, rtol=1e-5)

    def test_template_rejects_a_wrong_shaped_map(self):
        with pytest.raises(ValueError, match="template must be"):
            template_scores([np.zeros(N_PATCH, np.int8)], np.zeros(10, np.float32))

    def test_template_needs_at_least_one_whole_slice(self):
        with pytest.raises(ValueError, match="whole number of slices"):
            global_template([np.zeros((2, 7), np.float32)],
                            [np.zeros(7, np.int8)], slot=0)


class TestBinaryFindings:
    def test_channels_are_complementary(self):
        m = np.zeros(20, dtype=np.int8); m[3:6] = 1
        f = binary_findings([m])[0]
        np.testing.assert_allclose(f[0] + f[1], 1.0)

    def test_lesion_channel_is_first(self):
        """Downstream code takes column 0 as the lesion, so the order is load
        bearing rather than cosmetic."""
        m = np.zeros(10, dtype=np.int8); m[2] = 1
        assert binary_findings([m])[0][0, 2] == 1.0
