"""Tests for the two estimands the paper asks the field to report.

The failure mode worth guarding is not a wrong number, it is a *useless* one. It
is trivial to write a "corrected" metric that scores zero for every content-free
baseline -- return zero always and you are done. Such a metric would sail through
every sceptical check in this project and be worthless to anyone, because it
would also score zero for a method that genuinely localises.

So the tests come in pairs. For each estimand: it must collapse on the things
that carry no patient information (a position-only scorer, a cross-patient
target), *and* it must stay high on a scorer that really does find the lesion.
That second half is hypothesis H7, the power gate, and it is what licenses
recommending the protocol at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from slotmil.eval.estimands import (
    cluster_bootstrap,
    patient_specific_skill,
    prior_normalised_skill,
    reference_chain,
    stratified_auc,
    stratified_auc_detail,
)

GRID = 16
N_PATCH = GRID * GRID


def synthetic_bags(n_bags=60, n_slices=8, seed=0):
    """Bags whose lesions sit in a stereotyped in-plane place, like real nodules.

    Lesion patches cluster near the grid centre with per-bag jitter. That is the
    structure that makes a position-only scorer look good: it reproduces the
    population prior exactly while knowing nothing about any individual bag.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(-1, 1, GRID), np.linspace(-1, 1, GRID),
                         indexing="ij")
    radius = np.sqrt(yy**2 + xx**2).ravel()

    labels = []
    for _ in range(n_bags):
        lab = np.zeros(n_slices * N_PATCH, dtype=bool)
        cy, cx = rng.normal(0, 0.15, 2)
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).ravel()
        z = int(rng.integers(0, n_slices))
        lab[z * N_PATCH + np.argsort(d)[:4]] = True
        labels.append(lab)
    return labels, radius


def position_only_scores(labels, radius):
    """Content-free: the same map for every bag, never reads the target."""
    n_slices = len(labels[0]) // N_PATCH
    return [np.tile(-radius, n_slices) for _ in labels]


def oracle_scores(labels, seed=1):
    """A scorer that genuinely finds *this* bag's lesion, with noise."""
    rng = np.random.default_rng(seed)
    return [lab.astype(float) * 3.0 + rng.normal(0, 1.0, lab.size) for lab in labels]


def mean_auc(scores, labels):
    from sklearn.metrics import roc_auc_score
    return float(np.mean([roc_auc_score(l, s) for s, l in zip(scores, labels)]))


class TestPriorNormalisedSkill:
    def test_zero_when_the_method_only_matches_the_prior(self):
        assert prior_normalised_skill(0.80, 0.80) == pytest.approx(0.0)

    def test_one_at_a_perfect_score(self):
        assert prior_normalised_skill(1.0, 0.77) == pytest.approx(1.0)

    def test_negative_when_the_prior_wins(self):
        """Happens on real seeds: SlotMIL seed 2 (0.7508) loses to its own
        content-free template (0.7709)."""
        assert prior_normalised_skill(0.7508, 0.7709) < 0

    def test_a_stronger_reference_lowers_the_reported_skill(self):
        """The direction that matters -- a weak baseline flatters every method."""
        assert prior_normalised_skill(0.84, 0.78) < prior_normalised_skill(0.84, 0.60)

    def test_rejects_a_degenerate_reference(self):
        with pytest.raises(ValueError, match="auc_prior"):
            prior_normalised_skill(0.9, 1.0)

    def test_harvey_chest_ct_reproduces(self):
        """Their published chest-CT gain restated as skill over their own
        content-free baseline. Pins the paper's opening table."""
        assert prior_normalised_skill(0.866, 0.780) == pytest.approx(0.391, abs=1e-3)


class TestPowerGate:
    """H7. Both halves, because either one alone is worthless."""

    def test_a_position_only_scorer_collapses(self):
        labels, radius = synthetic_bags()
        auc = mean_auc(position_only_scores(labels, radius), labels)
        assert auc > 0.7, "the synthetic prior should be strong, like the real one"
        assert prior_normalised_skill(auc, auc) == pytest.approx(0.0)

    def test_a_real_localiser_keeps_most_of_its_skill(self):
        """The half that stops this being nihilism."""
        labels, radius = synthetic_bags()
        prior = mean_auc(position_only_scores(labels, radius), labels)
        assert prior_normalised_skill(mean_auc(oracle_scores(labels), labels),
                                      prior) > 0.5

    def test_the_gate_separates_them(self):
        labels, radius = synthetic_bags()
        prior = mean_auc(position_only_scores(labels, radius), labels)
        real = prior_normalised_skill(mean_auc(oracle_scores(labels), labels), prior)
        assert prior_normalised_skill(prior, prior) < 0.05 < 0.50 < real


class TestPatientSpecificSkill:
    def test_zero_when_the_swap_changes_nothing(self):
        assert patient_specific_skill(0.84, 0.84) == pytest.approx(0.0)

    def test_a_position_only_scorer_survives_a_cross_patient_swap(self):
        """Its score barely depends on whose masks it is scored against, so the
        difference is ~0 -- which is the finding, not a bug."""
        labels, radius = synthetic_bags()
        pos = position_only_scores(labels, radius)
        swapped = mean_auc(pos, labels[1:] + labels[:1])
        assert abs(patient_specific_skill(mean_auc(pos, labels), swapped)) < 0.05

    def test_a_real_localiser_does_not_survive_it(self):
        labels, _ = synthetic_bags()
        orc = oracle_scores(labels)
        swapped = mean_auc(orc, labels[1:] + labels[:1])
        assert patient_specific_skill(mean_auc(orc, labels), swapped) > 0.3


class TestReferenceChain:
    def test_steps_sum_to_the_total(self):
        links = {"chance": 0.4965, "template": 0.7453, "cross": 0.7724, "full": 0.8424}
        rows = reference_chain(links, ["chance", "template", "cross", "full"])
        assert sum(r["delta"] for r in rows[:-1]) == pytest.approx(rows[-1]["delta"])

    def test_fractions_sum_to_one(self):
        rows = reference_chain({"a": 0.5, "b": 0.7, "c": 0.9}, ["a", "b", "c"])
        assert sum(r["frac"] for r in rows[:-1]) == pytest.approx(1.0)

    def test_missing_link_raises_rather_than_silently_shortening(self):
        with pytest.raises(KeyError, match="undefined"):
            reference_chain({"a": 0.5}, ["a", "b"])

    def test_a_single_link_is_not_a_chain(self):
        with pytest.raises(ValueError, match="at least two"):
            reference_chain({"a": 0.5}, ["a"])


class TestClusterBootstrap:
    def test_interval_brackets_the_mean(self):
        rng = np.random.default_rng(0)
        ci = cluster_bootstrap(rng.normal(0.8, 0.05, 200),
                               [f"p{i // 2}" for i in range(200)], reps=500)
        assert ci["lo"] < ci["mean"] < ci["hi"]
        assert ci["n_clusters"] == 100

    def test_clustering_widens_the_interval_versus_treating_rows_as_independent(self):
        """Series from one patient share anatomy. Resampling rows pretends they
        are independent and reports an interval that is too narrow."""
        rng = np.random.default_rng(0)
        v = np.repeat(rng.normal(0, 0.10, 50), 4) + rng.normal(0, 0.01, 200)
        clustered = cluster_bootstrap(v, np.repeat(np.arange(50), 4), reps=800)
        independent = cluster_bootstrap(v, np.arange(200), reps=800)
        assert (clustered["hi"] - clustered["lo"]) > (independent["hi"] - independent["lo"])

    def test_non_finite_values_are_dropped_not_propagated(self):
        ci = cluster_bootstrap([0.8, np.nan, 0.9, np.inf], list("abcd"), reps=200)
        assert ci["n"] == 2 and np.isfinite(ci["mean"])

    def test_empty_input_returns_none_rather_than_nan_soup(self):
        assert cluster_bootstrap([], [], reps=10)["mean"] is None

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="disagree"):
            cluster_bootstrap([1.0, 2.0], ["a"], reps=10)

    def test_is_deterministic_given_a_seed(self):
        v, c = np.linspace(0, 1, 40), [f"p{i // 2}" for i in range(40)]
        assert cluster_bootstrap(v, c, reps=200, seed=7) == cluster_bootstrap(
            v, c, reps=200, seed=7)


class TestStratifiedAUC:
    def test_a_purely_positional_scorer_collapses_to_chance(self):
        """The load-bearing property. If stratifying by position did not remove a
        position-only score's advantage, the confirmatory analysis would be
        measuring nothing."""
        rng = np.random.default_rng(0)
        strata = rng.integers(0, 6, 4000)
        labels = rng.random(4000) < 0.3
        assert stratified_auc(strata.astype(float), labels, strata) == pytest.approx(
            0.5, abs=0.02)

    def test_a_within_stratum_signal_survives(self):
        rng = np.random.default_rng(0)
        strata = rng.integers(0, 6, 4000)
        labels = rng.random(4000) < 0.3
        scores = strata * 10.0 + labels * 2.0 + rng.normal(0, 0.5, 4000)
        assert stratified_auc(scores, labels, strata) > 0.9

    def test_ties_are_credited_as_half_a_pair(self):
        assert stratified_auc(np.zeros(4), [True, True, False, False],
                              np.zeros(4)) == pytest.approx(0.5)

    def test_single_class_strata_contribute_nothing(self):
        d = stratified_auc_detail([1.0, 2.0, 3.0, 4.0], [True, False, True, True],
                                  [0, 0, 1, 1])
        assert d["n_strata_used"] == 1 and d["n_pairs"] == 1

    def test_reports_when_stratification_discarded_everything(self):
        """A fine enough stratification leaves no comparable pairs, and the value
        must be nan rather than a confident-looking number."""
        d = stratified_auc_detail([1.0, 2.0], [True, False], [0, 1])
        assert np.isnan(d["auc"]) and d["n_pairs"] == 0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            stratified_auc([1.0, 2.0], [True], [0, 0])
