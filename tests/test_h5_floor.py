"""Tests for the H5 floored-denominator sensitivity.

This analysis sits *outside* the confirmatory family, which makes it more
dangerous rather than less: an unregistered number that contradicts a registered
one is exactly what a reader will check first, so the two have to be the same
statistic under two denominators and not two statistics that happen to share a
name. What is pinned here:

* Above the floor the floored value equals the published estimand **exactly**. If
  those ever drift, the sensitivity is measuring something the verdict is not, and
  the comparison in the paper is meaningless.
* Below the floor the floored value equals ``(auc - floor) / (1 - floor)`` -- the
  conservative direction, and the reason the analysis exists.
* The arm set is resolved through ``Prereg.arm_set("H5")`` rather than by
  iterating whatever tags are present, so ``mean`` and ``centre_gaussian`` cannot
  leak in. Their denominators are 0.5 by construction, so including them would put
  two arithmetic constants into a distribution reported as measured.
* The denominator cross-check fires. ``attention:inplane`` and the probe's
  template follow the identical fitting rule and score 0.42 and 0.836
  respectively; normalising against the wrong one silently changes every number.
"""

from __future__ import annotations

import json

import pytest

# scripts/ has no __init__.py and is not installed; pyproject's `pythonpath = ["."]`
# is what puts the repo root on sys.path for a bare `pytest`, and scripts/ then
# resolves as a namespace package. Same import route tests/test_verdict.py uses.
from scripts.h5_floored_denominator import analyse, by_arm, per_tag_skills
from slotmil import prereg
from slotmil.eval.estimands import prior_normalised_skill

FLOOR = 0.5


def _row(tag, auc, den):
    """One ``template_family.json`` result row, trimmed to what this script reads."""
    return {
        "tag": tag,
        "scorers": [
            {"scorer": "trained", "flat_auc": {"mean": auc}},
            {"scorer": "attention:inplane", "flat_auc": {"mean": den}},
        ],
        "prior_normalised_skill": {
            "auc_template": den,
            "trained": prior_normalised_skill(auc, den),
        },
    }


def test_above_floor_is_identical_to_the_published_estimand():
    doc = {"results": [_row("abmil_seed0", 0.90, 0.78)]}
    got = per_tag_skills(doc, FLOOR)["abmil_seed0"]
    assert got["template_below_floor"] is False
    assert got["floored"] == got["published"]
    assert got["floored"] == pytest.approx(prior_normalised_skill(0.90, 0.78))


def test_below_floor_uses_the_floor_and_moves_conservatively():
    auc, den = 0.60, 0.34
    doc = {"results": [_row("abmil_seed0", auc, den)]}
    got = per_tag_skills(doc, FLOOR)["abmil_seed0"]
    assert got["template_below_floor"] is True
    assert got["floored"] == pytest.approx((auc - FLOOR) / (1.0 - FLOOR))
    # a sub-chance denominator inflates `1 - AUC_t` above 1 and so shrinks the
    # published skill; flooring must therefore move it *down*, never up.
    assert got["floored"] < got["published"]


def test_denominator_mismatch_is_refused_rather_than_normalised_away():
    row = _row("abmil_seed0", 0.90, 0.42)
    row["prior_normalised_skill"]["auc_template"] = 0.8358  # the probe's template
    with pytest.raises(SystemExit, match="not the one that was scored"):
        per_tag_skills({"results": [row]}, FLOOR)


def test_arm_set_comes_from_the_prereg_not_from_the_tags_present():
    pre = prereg.load()
    h5 = set(pre.arm_set("H5"))
    assert "mean" not in h5 and "centre_gaussian" not in h5, (
        "uniform and fixed-geometric attention have outcomes fixed by arithmetic")

    seeds = [0]
    per_tag = {}
    for name in list(h5) + ["mean", "centre_gaussian"]:
        spec = pre.arm(name)["spec"].replace(":", "_")
        per_tag[f"{spec}_seed0"] = {
            "auc": 0.9, "auc_template": 0.5, "template_below_floor": False,
            "published": 0.8, "floored": 0.8,
        }
    assert set(by_arm(pre, per_tag, seeds)) == h5


def test_threshold_is_read_from_the_frozen_config():
    pre = prereg.load()
    doc = {"results": [
        _row(f"{pre.arm(n)['spec'].replace(':', '_')}_seed0", 0.55, 0.50)
        for n in pre.arm_set("H5")]}
    res = analyse(doc, pre, [0], FLOOR)
    assert res["threshold_read_from_prereg"] == pre.hypothesis("H5")["threshold"]
    assert res["declared_in_prereg"] is False
    assert res["role"] == "sensitivity_outside_confirmatory_family"


def test_a_floored_value_over_the_bar_is_reported_as_a_flip():
    pre = prereg.load()
    names = pre.arm_set("H5")
    rows = [_row(f"{pre.arm(n)['spec'].replace(':', '_')}_seed0", 0.55, 0.50)
            for n in names]
    rows[0] = _row(f"{pre.arm(names[0])['spec'].replace(':', '_')}_seed0", 0.90, 0.50)
    res = analyse({"results": rows}, pre, [0], FLOOR)
    assert res["verdict_would_flip"] is True
    assert res["arms_over_threshold_floored"] == [names[0]]


@pytest.mark.parametrize("condition", ["nodule_present", "malignancy",
                                       "balanced_presence"])
def test_the_shipped_artefacts_do_not_flip_h5(condition):
    """The claim the paper makes, pinned against the artefacts on disk."""
    path = (f"runs/nulls_{condition}_confirmatory/h5_floored_denominator.json")
    try:
        doc = json.loads(open(path).read())
    except FileNotFoundError:
        pytest.skip(f"{path} not generated")
    assert doc["verdict_would_flip"] is False
    assert doc["max_floored"] <= doc["threshold_read_from_prereg"]
    # flooring must not flatter the result in any condition
    assert doc["max_floored"] <= doc["max_published"] + 1e-12
