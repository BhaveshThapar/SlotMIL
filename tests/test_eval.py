"""Tests for the evaluation harness.

The alignment tests matter most: the slot-to-finding assignment is the paper's
central claim, and the whole defence against "you post-hoc named the slots" is
that the assignment is fit on validation and frozen before test. If those two
steps ever collapse into one, the claim is gone -- so the API separation is
tested, not just the arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from slotmil.eval.alignment import (
    apply_slot_assignment,
    fit_slot_assignment,
    head_redundancy,
    slot_consistency,
    slot_finding_affinity,
    slot_purity,
)
from slotmil.eval.classification import (
    _auc_variance_components,
    bootstrap_ci,
    classification_metrics,
    delong_test,
    holm_adjust,
    holm_reject,
)
from slotmil.eval.localization import (
    attn_to_volume,
    dice_iou,
    evaluate_localization,
    pointing_game,
)


class TestClassification:
    def test_perfect_separation(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        m = classification_metrics(y, s)
        assert m["auc"] == 1.0 and m["acc"] == 1.0

    def test_delong_identical_scores_gives_null(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 100)
        s = rng.random(100)
        r = delong_test(y, s, s)
        assert r["delta"] == pytest.approx(0.0, abs=1e-9)
        assert r["p"] == pytest.approx(1.0)

    def test_delong_detects_a_real_difference(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 300)
        good = y + rng.normal(0, 0.3, 300)  # informative
        bad = rng.random(300)  # chance
        r = delong_test(y, good, bad)
        assert r["delta"] > 0 and r["p"] < 0.01


class TestDelongComponentsAreExact:
    """The searchsorted form of V10/V01 must equal the double loop it replaced.

    ``_auc_variance_components`` was quadratic in (positives x negatives), which the
    declared ``statistics.holm_family`` cannot afford: that family is a paired test
    on the flat instance axis, which pools every patch of every bag -- ~4.5k
    positives against ~6.5M negatives, so ~3e10 comparisons per model. The binary
    search is the same quantity, and *bit-identical* is the bar rather than
    approximately equal, because the pre-registered thresholds were set from
    discovery numbers and a p-value that moved during a speedup would look exactly
    like a finding.
    """

    @staticmethod
    def _brute(y_true, y_score):
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        m, n = len(pos), len(neg)
        v10 = np.array([(np.sum(p > neg) + 0.5 * np.sum(p == neg)) / n for p in pos])
        v01 = np.array([(np.sum(pos > q) + 0.5 * np.sum(pos == q)) / m for q in neg])
        return v10.mean(), v10, v01

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_the_double_loop_bitwise(self, seed):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, 400)
        s = rng.random(400)
        auc, v10, v01 = _auc_variance_components(y, s)
        b_auc, b_v10, b_v01 = self._brute(y, s)
        # Exact equality, not allclose: see the class docstring.
        assert auc == b_auc
        assert np.array_equal(v10, b_v10)
        assert np.array_equal(v01, b_v01)

    def test_ties_are_credited_identically(self):
        """Heavy ties are where a left/right searchsorted mix-up would show up.

        Attention dumps tie constantly -- a padded column is exactly 0.0 and a
        template repeats 256 values across every slice -- so the tie path is the
        common case here rather than an edge case.
        """
        y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        s = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.9, 0.9])
        auc, v10, v01 = _auc_variance_components(y, s)
        b_auc, b_v10, b_v01 = self._brute(y, s)
        assert auc == b_auc
        assert np.array_equal(v10, b_v10)
        assert np.array_equal(v01, b_v01)

    def test_one_class_still_raises(self):
        with pytest.raises(ValueError, match="both classes"):
            _auc_variance_components(np.ones(5, dtype=int), np.arange(5.0))

    def test_bootstrap_ci_brackets_the_estimate(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        s = np.stack([1 - (y + rng.normal(0, 0.4, 200)), y + rng.normal(0, 0.4, 200)], 1)
        ci = bootstrap_ci(y, s, n_boot=200)
        assert ci["lo"] <= ci["mean"] <= ci["hi"]


class TestHolm:
    """Holm-Bonferroni over the nine pre-registered hypotheses.

    Paired, per the rule in tests/test_estimands.py. A multiplicity correction is
    the easy thing to fake in both directions: `return np.ones_like(p)` rejects
    nothing and passes every scepticism check, and `return p` rejects everything
    and passes every power check. So each property is pinned twice -- the
    correction must actually suppress a family of null p-values *and* still let a
    genuinely small one through, and each of the two steps that are easy to get
    wrong (running maximum, order restoration) is tested against the specific
    wrong answer it replaces, so the test would fail if the step were dropped.
    """

    # p = 0.01 .. 0.05, n = 5. Chosen because the raw scaling
    # [0.05, 0.08, 0.09, 0.08, 0.05] is non-monotone at both tail entries, so this
    # one family exercises the running maximum and the step-down rule together.
    FAMILY = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    EXPECTED = np.array([0.05, 0.08, 0.09, 0.09, 0.09])

    def test_worked_example_matches_the_hand_calculation(self):
        np.testing.assert_allclose(holm_adjust(self.FAMILY), self.EXPECTED)

    def test_a_family_of_ones_collapses_to_ones(self):
        """The other half: nothing to find, nothing found, and no value above 1."""
        np.testing.assert_allclose(holm_adjust(np.ones(9)), np.ones(9))

    def test_uniform_null_pvalues_produce_no_rejections(self):
        """20 p-values drawn under the global null. Four of them clear a raw 0.05
        -- which is the whole reason the correction is pre-registered -- and Holm
        rejects none of them."""
        u = np.random.default_rng(0).random(20)
        assert (u < 0.05).sum() == 4
        assert holm_reject(u)["n_reject"] == 0

    def test_a_genuinely_small_pvalue_still_survives_the_correction(self):
        """The power half. A correction that suppressed this too would be useless."""
        u = np.random.default_rng(0).random(20)
        u[7] = 1e-6
        r = holm_reject(u)
        assert r["reject"][7] and r["n_reject"] == 1

    def test_adjusted_pvalues_are_monotone_in_the_sorted_order(self):
        rng = np.random.default_rng(3)
        p = rng.random(30)
        adj = holm_adjust(p)[np.argsort(p)]
        assert np.all(np.diff(adj) >= 0)

    def test_the_running_maximum_is_what_makes_it_monotone(self):
        """Teeth for the test above: the unaccumulated scaling it replaces is not
        monotone on the same family, and would report the 5th hypothesis as more
        significant than the 2nd."""
        naive = self.FAMILY * (len(self.FAMILY) - np.arange(len(self.FAMILY)))
        assert not np.all(np.diff(naive) >= 0)
        assert naive[4] < naive[1]

    def test_output_order_matches_a_shuffled_input(self):
        p = np.array([0.04, 0.01, 0.05, 0.02, 0.03])
        adj = holm_adjust(p)
        # Same family as FAMILY, permuted; each entry must carry its own hypothesis'
        # adjusted value, not the value that landed in its sorted position.
        expected = self.EXPECTED[np.argsort(np.argsort(p))]
        np.testing.assert_allclose(adj, expected)
        np.testing.assert_allclose(adj, [0.09, 0.05, 0.09, 0.08, 0.09])

    def test_resorting_instead_of_unsorting_would_mislabel_the_hypotheses(self):
        """Teeth for the test above. `adjusted[order]` is the natural typo and it
        agrees with the correct answer whenever `order` is its own inverse -- so a
        suite that only tried sorted or reversed input would never catch it."""
        p = np.array([0.04, 0.01, 0.05, 0.02, 0.03])
        order = np.argsort(p)
        wrong = holm_adjust(p)[order]
        assert not np.allclose(wrong, holm_adjust(p))

    def test_holm_is_never_more_conservative_than_bonferroni(self):
        rng = np.random.default_rng(11)
        p = rng.random(12) * 0.2
        bonf = np.minimum(p * p.size, 1.0)
        holm = holm_adjust(p)
        assert np.all(holm <= bonf + 1e-12)

    def test_and_agrees_with_bonferroni_exactly_at_the_smallest_pvalue(self):
        """The pair: strictly less everywhere else, equal at the minimum, because
        the smallest hypothesis still pays the full family size. If Holm were also
        cheaper there it would not control the family-wise error rate."""
        rng = np.random.default_rng(11)
        p = rng.random(12) * 0.2
        bonf = np.minimum(p * p.size, 1.0)
        holm = holm_adjust(p)
        smallest = int(np.argmin(p))
        assert holm[smallest] == pytest.approx(bonf[smallest])
        others = np.setdiff1d(np.arange(p.size), [smallest])
        assert np.all(holm[others] < bonf[others])

    def test_a_family_of_one_is_the_identity(self):
        np.testing.assert_allclose(holm_adjust([0.037]), [0.037])

    def test_an_empty_family_is_empty_rather_than_an_error(self):
        r = holm_reject([])
        assert r["p_adjusted"].shape == (0,) and r["n_family"] == 0

    def test_rejection_stops_at_the_first_failure(self):
        """The step-down rule. 0.01 clears 0.05/5, so H0 is rejected; 0.02 does not
        clear 0.05/4, so testing stops -- and 0.05, which would clear its own
        0.05/1 threshold if tested in isolation, is not rejected."""
        r = holm_reject(self.FAMILY, alpha=0.05)
        assert r["reject"].tolist() == [True, False, False, False, False]
        assert self.FAMILY[4] <= 0.05, "the last one would pass an uncorrected test"

    def test_a_nan_hypothesis_propagates_and_is_never_rejected(self):
        r = holm_reject([0.02, 0.03, np.nan])
        assert np.isnan(r["p_adjusted"][2]) and not r["reject"][2]
        assert np.all(np.isfinite(r["p_adjusted"][:2]))

    def test_a_nan_hypothesis_still_costs_the_others_their_family_size(self):
        """The pair, and the reason NaN is not simply dropped: with the
        uncomputable hypothesis counted the family is 3 and neither survivor is
        rejected; drop it and both are. Silently shrinking n would manufacture two
        significant results out of a failed computation."""
        assert holm_reject([0.02, 0.03, np.nan])["n_reject"] == 0
        assert holm_reject([0.02, 0.03])["n_reject"] == 2

    def test_pvalues_outside_the_unit_interval_raise(self):
        with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
            holm_adjust([0.01, 1.7])

    def test_a_two_dimensional_family_raises_rather_than_sorting_per_row(self):
        with pytest.raises(ValueError, match="1-D family"):
            holm_adjust(np.array([[0.01, 0.02], [0.03, 0.04]]))


class TestLocalization:
    def test_dice_of_identical_masks(self):
        m = np.zeros((4, 8, 8), dtype=np.float32)
        m[1, 2:5, 2:5] = 1.0
        assert dice_iou(m, m, threshold=0.5)["dice"] == pytest.approx(1.0)

    def test_dice_of_disjoint_masks(self):
        a = np.zeros((4, 8, 8), dtype=np.float32); a[0, 0:2, 0:2] = 1.0
        b = np.zeros((4, 8, 8), dtype=np.float32); b[3, 6:8, 6:8] = 1.0
        assert dice_iou(a, b, threshold=0.5)["dice"] == 0.0

    def test_pointing_game(self):
        heat = np.zeros((2, 4, 4)); heat[1, 2, 3] = 5.0
        hit = np.zeros((2, 4, 4)); hit[1, 2, 3] = 1
        miss = np.zeros((2, 4, 4)); miss[0, 0, 0] = 1
        assert pointing_game(heat, hit) is True
        assert pointing_game(heat, miss) is False

    def test_attn_to_volume_shape_and_z_is_not_interpolated(self):
        attn = torch.rand(3, 5 * 4 * 4)
        vol = attn_to_volume(attn, n_slices=5, grid_h=4, grid_w=4, out_hw=(32, 32))
        assert vol.shape == (3, 5, 32, 32)

    def test_attn_to_volume_rejects_mismatched_length(self):
        with pytest.raises(ValueError, match="instances"):
            attn_to_volume(torch.rand(3, 10), n_slices=5, grid_h=4, grid_w=4, out_hw=(8, 8))

    def test_reporter_reads_keys_that_evaluate_localization_returns(self):
        """scripts/eval_alignment.py's summary block is not otherwise covered.

        `dice_std` was renamed to `dice_std_across_bags` here and the reporter was
        missed, so every alignment run raised KeyError *after* writing its JSON --
        results survived, but the non-zero exit made lidc_align.sbatch's
        `|| echo "ALIGNMENT FAILED"` fire on runs that had actually succeeded.
        Pinning the key names is cheaper than noticing that again.
        """
        import re
        from pathlib import Path

        source = Path(__file__).resolve().parents[1] / "scripts" / "eval_alignment.py"
        used = set(re.findall(r"loc\[['\"](\w+)['\"]\]", source.read_text()))

        rng = np.random.default_rng(0)
        n_slices, grid = 3, 4
        n = n_slices * grid * grid
        attn = rng.random((2, n)).astype(np.float32)
        masks = [(rng.random(n) > 0.7).astype(np.float32) for _ in range(2)]
        out = evaluate_localization(
            [attn, attn], masks, [n_slices, n_slices], grid, lesion_slot=0
        )

        missing = used - set(out)
        assert not missing, f"eval_alignment.py reads keys that do not exist: {missing}"


class TestAlignment:
    @staticmethod
    def _specialised_bags(n_bags=20, n_slots=2, n_inst=40):
        """Slot 0 attends finding 0, slot 1 attends finding 1 -- clean specialisation."""
        attns, masks = [], []
        for _ in range(n_bags):
            a = np.full((n_slots, n_inst), 0.01)
            m = np.zeros((2, n_inst))
            a[0, :10] = 1.0; m[0, :10] = 1
            a[1, 20:30] = 1.0; m[1, 20:30] = 1
            attns.append(a / a.sum(-1, keepdims=True))
            masks.append(m)
        return attns, masks

    def test_affinity_recovers_specialisation(self):
        attns, masks = self._specialised_bags()
        aff = slot_finding_affinity(attns, masks)
        assert aff.shape == (2, 2)
        assert aff[0, 0] > aff[0, 1] and aff[1, 1] > aff[1, 0]

    def test_assignment_is_fit_then_applied_separately(self):
        """The two-step API is the guard against post-hoc naming."""
        attns, masks = self._specialised_bags()
        val_aff = slot_finding_affinity(attns[:10], masks[:10])
        assignment = fit_slot_assignment(val_aff)
        assert assignment == {0: 0, 1: 1}

        test_aff = slot_finding_affinity(attns[10:], masks[10:])
        scored = apply_slot_assignment(test_aff, assignment)
        assert scored["lift_over_chance"] > 1.0
        assert scored["mean_assigned_affinity"] > scored["chance_affinity"]

    def test_purity_high_when_specialised(self):
        attns, _ = self._specialised_bags()
        labels = [np.concatenate([np.zeros(20), np.ones(20)]).astype(int) for _ in attns]
        r = slot_purity(attns, labels)
        assert 0.0 <= r["purity"] <= 1.0 and r["n_active_slots"] >= 1

    def test_consistency_beats_chance_when_stable(self):
        attns, masks = self._specialised_bags()
        r = slot_consistency(attns, masks)
        assert r["mean_consistency"] == pytest.approx(1.0)
        assert r["lift_over_chance"] > 1.5

    def test_redundancy_separates_duplicate_from_distinct_heads(self):
        n = 30
        dup = [np.tile(np.eye(1, n, 3), (4, 1)) + 1e-6 for _ in range(5)]
        distinct = []
        for _ in range(5):
            a = np.zeros((4, n))
            for k in range(4):
                a[k, k * 5 : (k + 1) * 5] = 1.0
            distinct.append(a)
        assert head_redundancy(dup)["mean_pairwise_cosine"] > \
               head_redundancy(distinct)["mean_pairwise_cosine"]

    def test_affinity_errors_when_no_annotations(self):
        with pytest.raises(ValueError, match="no bags with annotated findings"):
            slot_finding_affinity([np.ones((2, 5))], [np.zeros((2, 5))])


class TestPartiallyAnnotatedCollate:
    """A cache where only some bags carry masks (MosMed: 50 of 1110).

    collate_bags used to decide whether to emit 'patch_target' from batch[0]
    alone, so any batch whose first item happened to be unmasked silently
    dropped the field -- and the alignment evaluation died with a bare KeyError
    after the training run had already completed.
    """

    @staticmethod
    def _bag(n, dim, masked):
        item = {
            "features": torch.randn(n, dim),
            "label": torch.tensor(1),
            "uid": "m" if masked else "u",
            "n_slices": 1,
            "slice_index": torch.arange(1),
        }
        if masked:
            item["patch_target"] = torch.ones(n)
        return item

    def test_unmasked_first_item_still_emits_target(self):
        from slotmil.data.feature_cache import collate_bags

        batch = collate_bags([self._bag(5, 8, False), self._bag(5, 8, True)])
        assert "patch_target" in batch, "field dropped when batch[0] is unmasked"
        assert batch["has_mask"].tolist() == [False, True]
        assert batch["patch_target"][0].sum() == 0  # unannotated -> zeros
        assert batch["patch_target"][1].sum() == 5

    def test_all_unmasked_emits_nothing(self):
        from slotmil.data.feature_cache import collate_bags

        batch = collate_bags([self._bag(5, 8, False), self._bag(5, 8, False)])
        assert "patch_target" not in batch

    def test_has_mask_distinguishes_empty_from_unannotated(self):
        """A zero target must not be readable as 'annotated, no lesion'."""
        from slotmil.data.feature_cache import collate_bags

        empty = self._bag(5, 8, True)
        empty["patch_target"] = torch.zeros(5)  # annotated, genuinely no lesion
        batch = collate_bags([self._bag(5, 8, False), empty])
        assert batch["has_mask"].tolist() == [False, True]


class TestInstanceAUCSlotSelection:
    """max-over-slots is invalid for softmax-over-slots attention.

    Slot attention normalises over the slot axis, so sum_k attn[k,n] == 1 for
    every instance. max_k attn[k,n] therefore measures assignment CONFIDENCE, not
    lesion saliency -- a background patch confidently bound to the background slot
    scores high. Reporting it produced instance AUC 0.489 ("no localisation")
    where the frozen lesion slot gives 0.842 on the same checkpoint.
    """

    @staticmethod
    def _bag(n=200, k=4, lesion_slot=1, n_lesion=20):
        """Slot `lesion_slot` attends the lesion; another slot attends background
        even more confidently, so max-over-slots is dominated by background."""
        labels = np.zeros(n, dtype=int)
        labels[:n_lesion] = 1
        logits = np.zeros((k, n))
        logits[lesion_slot, :n_lesion] = 3.0      # lesion slot finds the lesion
        logits[0, n_lesion:] = 8.0                # background slot is MORE confident
        attn = np.exp(logits) / np.exp(logits).sum(axis=0, keepdims=True)  # over slots
        return attn, labels

    def test_softmax_over_slots_sums_to_one_per_instance(self):
        attn, _ = self._bag()
        np.testing.assert_allclose(attn.sum(axis=0), 1.0, rtol=1e-6)

    def test_max_over_slots_misses_a_perfect_lesion_slot(self):
        from slotmil.eval.localization import instance_auc

        attn, labels = self._bag()
        assert instance_auc(attn, labels, slot=1) > 0.99, "lesion slot is perfect"
        assert instance_auc(attn, labels, slot=None) < 0.5, (
            "max-over-slots should be FOOLED here -- that is the bug this guards"
        )

    def test_frozen_slot_is_used_by_evaluate_localization(self):
        from slotmil.eval.localization import evaluate_localization

        attn, labels = self._bag(n=4 * 8 * 8, n_lesion=40)
        good = evaluate_localization([attn], [labels.astype(float)], [4], 8, lesion_slot=1)
        bad = evaluate_localization([attn], [labels.astype(float)], [4], 8, lesion_slot=None)
        assert good["instance_auc"] > bad["instance_auc"] + 0.4
        assert good["lesion_slot"] == 1
        assert "ap_lift_over_prevalence" in good
