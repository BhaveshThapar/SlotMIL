"""Tests for the Harvey bridge.

What is pinned: the MosMed rows can never masquerade as confirmatory (the
note rides in the artefact and a confirmatory output path refuses), the
per-arm aggregation reads the stored axis_gate rows rather than recomputing
them, and the published-numbers arithmetic is the imported single source.
"""

from __future__ import annotations

import json
import statistics

import pytest

from scripts.harvey_bridge import (
    MOSMED_NOTE,
    per_arm_axes,
    what_their_metric_cannot_see,
)
from scripts.harvey_reanalysis import HARVEY_TABLE1, prior_normalised_skill
from slotmil import prereg


def _row(tag, flat, sl, wi):
    return {"tag": tag,
            "flat_auc": {"mean": flat}, "slice_auc": {"mean": sl},
            "within_slice_auc": {"mean": wi}}


def test_per_arm_is_the_mean_of_the_stored_rows():
    pre = prereg.load()
    name = pre.arm_set("H1")[0]
    spec = pre.arm(name)["spec"].replace(":", "_")
    rows = {f"{spec}_seed0": _row(f"{spec}_seed0", 0.80, 0.50, 0.78),
            f"{spec}_seed1": _row(f"{spec}_seed1", 0.70, 0.40, 0.72)}
    got = per_arm_axes(pre, rows, [0, 1])
    assert got[name]["flat_auc"] == pytest.approx(statistics.fmean([0.80, 0.70]))
    assert got[name]["slice_auc"] == pytest.approx(0.45)
    assert got[name]["n_seeds"] == 2


def test_arm_set_excludes_the_construction_fixed_arms():
    pre = prereg.load()
    rows = {}
    for name in ["mean", "centre_gaussian"]:
        spec = pre.arm(name)["spec"].replace(":", "_")
        rows[f"{spec}_seed0"] = _row(f"{spec}_seed0", 0.5, 0.5, 0.5)
    assert per_arm_axes(pre, rows, [0]) == {}


def test_the_gap_their_metric_cannot_see_counts_flat_vs_within():
    lidc = {"nodule_present": {"per_arm": {
        "a": {"flat_auc": 0.80, "within_slice_auc": 0.79, "slice_auc": 0.5,
              "n_seeds": 5},
        "b": {"flat_auc": 0.70, "within_slice_auc": 0.63, "slice_auc": 0.6,
              "n_seeds": 5},
    }}}
    got = what_their_metric_cannot_see(lidc)
    assert got["n_arms_gap_over_0.02"] == 1
    assert got["abs_flat_minus_within_per_arm"]["b"] == pytest.approx(0.07)


def test_published_arithmetic_is_the_single_imported_source():
    for _, (_, prior, ng) in HARVEY_TABLE1.items():
        assert prior_normalised_skill(ng, prior) == (ng - prior) / (1 - prior)


def test_shipped_artefact_reads_not_recomputes_the_lidc_rows():
    path = "runs/harvey_bridge/harvey_bridge.json"
    try:
        doc = json.loads(open(path).read())
    except FileNotFoundError:
        pytest.skip(f"{path} not generated")
    assert doc["analysis_role"] == "exploratory_outside_confirmatory_family"
    assert doc["mosmed_exploratory"]["note"] == MOSMED_NOTE
    # The family condition's numbers must equal the stored artefact exactly.
    ag = json.loads(open(
        "runs/nulls_nodule_present_confirmatory/axis_gate.json").read())
    rows = {r["tag"]: r for r in ag["results"]}
    pre = prereg.load()
    fam = doc["their_metric_on_our_arms"]["nodule_present"]["per_arm"]
    recomputed = per_arm_axes(pre, rows, [0, 1, 2, 3, 4])
    for arm, v in fam.items():
        for k in ("flat_auc", "slice_auc", "within_slice_auc"):
            assert v[k] == pytest.approx(recomputed[arm][k], abs=1e-12), (arm, k)
    fam_prior = doc["their_metric_on_our_arms"]["nodule_present"][
        "centre_prior_slice_auc"]
    assert fam_prior == rows["centre_prior"]["slice_auc"]["mean"]
