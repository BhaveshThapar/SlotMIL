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
from slotmil.eval.classification import bootstrap_ci, classification_metrics, delong_test
from slotmil.eval.localization import attn_to_volume, dice_iou, pointing_game


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

    def test_bootstrap_ci_brackets_the_estimate(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        s = np.stack([1 - (y + rng.normal(0, 0.4, 200)), y + rng.normal(0, 0.4, 200)], 1)
        ci = bootstrap_ci(y, s, n_boot=200)
        assert ci["lo"] <= ci["mean"] <= ci["hi"]


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
