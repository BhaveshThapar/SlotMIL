"""Tests for the H2 paired-interval sensitivity.

The analysis exists because H2's verdict rests on a 0.0109 point-estimate
margin, so what matters here is that the paired number cannot quietly become a
different quantity than the one the verdict compares. Pinned:

* the sign convention is ``prior - arm``: a positive delta means the prior
  sits above the arm, matching the direction H2's PASS asserts;
* alignment is by originating bag index, not by list position -- a scorer
  that dropped a bag must pair the surviving bags correctly;
* the local delta rows reproduce ``template_family._paired`` exactly, because
  the pooled interval is built from them and ``_paired`` is the tested path;
* pooling five seeds clusters by *patient*: a patient contributing rows to
  every seed resamples as one unit, not five;
* the arm set is resolved through ``Prereg.arm_set("H2")``, so the two arms
  whose slice behaviour is fixed by construction cannot leak in.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

# scripts/ has no __init__.py and is not installed; pyproject's `pythonpath = ["."]`
# is what puts the repo root on sys.path for a bare `pytest`, and scripts/ then
# resolves as a namespace package. Same import route tests/test_h5_floor.py uses.
from scripts.h2_paired_slice import OUTCOME, delta_rows
from scripts.template_family import _paired
from slotmil import prereg
from slotmil.eval.estimands import cluster_bootstrap, h10_outcome

UIDS = np.array(["u0", "u1", "u2", "u3"])
PATS = {"u0": "p0", "u1": "p1", "u2": "p2", "u3": "p3"}


def _rows(slice_vals, idx=None):
    """per_bag_axes-shaped tuples with only the slice column populated."""
    idx = range(len(slice_vals)) if idx is None else idx
    return [(i, 0.5, v, 0.5, 2, 1) for i, v in zip(idx, slice_vals)]


def test_sign_convention_is_prior_minus_arm():
    prior = _rows([0.70, 0.70, 0.70])
    arm = _rows([0.60, 0.65, 0.55])
    deltas, pats = delta_rows(prior, arm, UIDS, PATS)
    assert deltas == pytest.approx([0.10, 0.05, 0.15])
    assert pats == ["p0", "p1", "p2"]


def test_alignment_is_by_bag_index_not_position():
    # The arm's rows are missing bag 1; positional pairing would subtract
    # bag 2's value from bag 1's and report three deltas instead of two.
    prior = _rows([0.70, 0.70, 0.70], idx=[0, 1, 2])
    arm = _rows([0.60, 0.50], idx=[0, 2])
    deltas, pats = delta_rows(prior, arm, UIDS, PATS)
    assert deltas == pytest.approx([0.10, 0.20])
    assert pats == ["p0", "p2"]


def test_nan_deltas_are_dropped_by_the_bootstrap_not_averaged():
    prior = _rows([0.70, np.nan, 0.70])
    arm = _rows([0.60, 0.60, 0.60])
    deltas, pats = delta_rows(prior, arm, UIDS, PATS)
    ci = cluster_bootstrap(deltas, pats, 0, 0)
    assert ci["n"] == 2
    assert ci["mean"] == pytest.approx(0.10)


def test_local_deltas_reproduce_the_tested_paired_path():
    prior = _rows([0.70, 0.64, 0.61, np.nan])
    arm = _rows([0.60, 0.66, 0.55, 0.50])
    deltas, pats = delta_rows(prior, arm, UIDS, PATS)
    ours = cluster_bootstrap(deltas, pats, 200, 0)
    ref = _paired(prior, arm, UIDS, PATS, 200, 0, col=2)
    assert ours == ref


def test_pooling_clusters_by_patient_across_seeds():
    # One patient's per-bag deltas from two seeds must resample as one unit.
    deltas = [0.1, 0.2, 0.1, 0.2]        # seed0 bags, then seed1 bags
    pats = ["p0", "p1", "p0", "p1"]      # the same two patients both times
    ci = cluster_bootstrap(deltas, pats, 0, 0)
    assert ci["n_clusters"] == 2
    assert ci["n"] == 4


def test_outcome_labels_translate_h10_without_reimplementing_it():
    prior_wins = {"mean": 0.02, "lo": 0.01, "hi": 0.03}
    arm_wins = {"mean": -0.02, "lo": -0.03, "hi": -0.01}
    ties = {"mean": 0.00, "lo": -0.01, "hi": 0.01}
    undefined = {"mean": None, "lo": None, "hi": None}
    assert OUTCOME[h10_outcome(prior_wins)] == "prior_wins"
    assert OUTCOME[h10_outcome(arm_wins)] == "arm_wins"
    assert OUTCOME[h10_outcome(ties)] == "indistinguishable"
    assert OUTCOME[h10_outcome(undefined)] is None


def test_arm_set_comes_from_the_prereg_not_from_the_tags_present():
    h2 = set(prereg.load().arm_set("H2"))
    assert "mean" not in h2, "uniform attention ties every axis at 0.5"
    assert "centre_gaussian" not in h2, (
        "an arm that IS a centre prior would tie the reference, and a tie is "
        "not 'exceeds'")


@pytest.mark.parametrize("condition", ["nodule_present", "malignancy",
                                       "balanced_presence"])
def test_the_shipped_artefacts_label_their_own_intervals(condition):
    """Outcome labels must agree with the interval signs they summarise."""
    path = f"runs/nulls_{condition}_confirmatory/h2_paired_slice.json"
    try:
        doc = json.loads(open(path).read())
    except FileNotFoundError:
        pytest.skip(f"{path} not generated")
    assert doc["declared_in_prereg"] is False
    assert doc["role"] == "sensitivity_outside_confirmatory_family"
    for v in doc["per_arm"].values():
        assert v["pooled"]["outcome"] == OUTCOME[h10_outcome(v["pooled"])]
        for r in v["per_seed"]:
            assert r["outcome"] == OUTCOME[h10_outcome(r["delta"])]
