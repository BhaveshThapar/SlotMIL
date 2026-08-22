"""Tests for the arms' own raw-vs-in-lung sensitivity.

The failure classes pinned here are the ones ``h8_in_lung.py``'s comments
document from experience, plus the refusals that tie this artefact to the
confirmatory record:

* the scored quantity is the arm's **frozen slot**, not row 0 -- a dump whose
  lesion slot happens to be 0 scoring correctly while every other dump scores
  a different slot under the same name is the quiet version of the bug;
* restriction goes through :func:`slotmil.eval.lung.restrict_to_lung`, which
  drops score and target together;
* a frozen-slot disagreement with ``template_family.json`` refuses;
* a raw mean that does not reproduce the stored ``trained`` ``flat_auc.mean``
  refuses -- the raw column already exists in the confirmatory artefact and
  this script must land on exactly that number;
* the arm set comes from the pre-registration, not from the tags present.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

# scripts/ has no __init__.py and is not installed; pyproject's `pythonpath = ["."]`
# is what puts the repo root on sys.path for a bare `pytest`, and scripts/ then
# resolves as a namespace package. Same import route tests/test_h5_floor.py uses.
from scripts.arm_in_lung import ARM_SET_HYPOTHESIS, analyse_tag, by_arm
from slotmil import prereg

GRID = 16
N_PATCH = GRID * GRID
N_SLICES = 2
N = N_SLICES * N_PATCH  # one bag's instance count


def _bag(rng):
    """One synthetic bag: slot 1 attends the lesion, slot 0 the background."""
    mask = np.zeros(N, dtype=np.uint8)
    lesion = rng.choice(N, size=8, replace=False)
    mask[lesion] = 1
    attn = rng.uniform(0.0, 0.2, size=(2, N)).astype(np.float32)
    attn[1, lesion] += 0.8       # slot 1: strong on the lesion
    attn[0, mask == 0] += 0.8    # slot 0: strong on the background
    return attn, mask


def _dump(path, bags, uids):
    attn = np.concatenate([a for a, _ in bags], axis=1)
    mask = np.concatenate([m for _, m in bags])
    lengths = np.array([m.shape[0] for _, m in bags], dtype=np.int64)
    np.savez(path, attn=attn, mask=mask, lengths=lengths,
             uids=np.array(uids))


def _lung_store(path, uids, in_lung_flat):
    """Every uid gets the same [n_slices, g, g] lung-fraction grid."""
    grid = in_lung_flat.astype(np.float32).reshape(N_SLICES, GRID, GRID)
    with h5py.File(path, "w") as f:
        for uid in uids:
            f.create_group(uid).create_dataset("lung", data=grid)


def _fixture(tmp_path, n_bags=4, seed=0):
    rng = np.random.default_rng(seed)
    uids = [f"u{i}" for i in range(n_bags)]
    bags = [_bag(rng) for _ in range(n_bags)]
    d = tmp_path / "nulls"
    d.mkdir()
    _dump(d / "arm_seed0_val.npz", bags, uids)
    _dump(d / "arm_seed0_test.npz", bags, uids)

    # Half of each bag is lung; every lesion patch is inside it (containment
    # 1.0, like the real store). Outside the lung, slot 1 still carries some
    # attention mass, so raw and in-lung genuinely differ.
    in_lung = np.zeros(N, dtype=bool)
    for i, (_, m) in enumerate(bags):
        lesions = np.flatnonzero(m)
        in_lung[lesions] = True
    in_lung[: N // 2] = True
    lung_path = tmp_path / "lung.h5"
    _lung_store(lung_path, uids, in_lung)

    raw_expect = [float(roc_auc_score(m, a[1])) for a, m in bags]
    stored = {"arm_seed0": {"frozen_slot": 1,
                            "flat_auc_mean": float(np.mean(raw_expect))}}
    pats = {u: f"p{u}" for u in uids}
    return d, lung_path, pats, stored, bags, in_lung


def test_scores_the_frozen_slot_not_row_zero(tmp_path):
    d, lung, pats, stored, bags, in_lung = _fixture(tmp_path)
    r = analyse_tag("arm_seed0", d / "arm_seed0_val.npz",
                    d / "arm_seed0_test.npz", lung, pats, 0, 0, 0.0, stored)
    assert r["frozen_slot"] == 1
    # Scoring slot 0 instead would invert the ranking and land near 1 - AUC.
    slot0 = float(np.mean([roc_auc_score(m, a[0]) for a, m in bags]))
    assert r["raw_auc"]["mean"] == pytest.approx(stored["arm_seed0"]["flat_auc_mean"])
    assert abs(r["raw_auc"]["mean"] - slot0) > 0.2


def test_restriction_drops_score_and_target_together(tmp_path):
    d, lung, pats, stored, bags, in_lung = _fixture(tmp_path)
    r = analyse_tag("arm_seed0", d / "arm_seed0_val.npz",
                    d / "arm_seed0_test.npz", lung, pats, 0, 0, 0.0, stored)
    expect = float(np.mean([
        roc_auc_score(m[in_lung], a[1][in_lung]) for a, m in bags]))
    assert r["in_lung_auc"]["mean"] == pytest.approx(expect)
    assert r["mean_in_lung_fraction"] == pytest.approx(in_lung.mean())
    # containment 1.0 in the fixture, as in the real store
    assert r["n_bags_with_no_lesion_in_lung"] == 0


def test_frozen_slot_mismatch_is_refused(tmp_path):
    d, lung, pats, stored, *_ = _fixture(tmp_path)
    stored["arm_seed0"]["frozen_slot"] = 0
    with pytest.raises(SystemExit, match="different slots"):
        analyse_tag("arm_seed0", d / "arm_seed0_val.npz",
                    d / "arm_seed0_test.npz", lung, pats, 0, 0, 0.0, stored)


def test_raw_disagreement_with_the_stored_artefact_is_refused(tmp_path):
    d, lung, pats, stored, *_ = _fixture(tmp_path)
    stored["arm_seed0"]["flat_auc_mean"] += 0.01
    with pytest.raises(SystemExit, match="paper cites"):
        analyse_tag("arm_seed0", d / "arm_seed0_val.npz",
                    d / "arm_seed0_test.npz", lung, pats, 0, 0, 0.0, stored)


def test_a_missing_stored_row_is_refused_not_skipped(tmp_path):
    d, lung, pats, stored, *_ = _fixture(tmp_path)
    with pytest.raises(SystemExit, match="no trained row"):
        analyse_tag("arm_seed0", d / "arm_seed0_val.npz",
                    d / "arm_seed0_test.npz", lung, pats, 0, 0, 0.0, {})


def test_arm_set_comes_from_the_prereg_not_from_the_tags_present():
    pre = prereg.load()
    scored = set(pre.arm_set(ARM_SET_HYPOTHESIS))
    assert "mean" not in scored and "centre_gaussian" not in scored

    row = {"raw_auc": {"mean": 0.8}, "in_lung_auc": {"mean": 0.5},
           "delta_auc": {"mean": 0.3}}
    per_tag = {}
    for name in list(scored) + ["mean", "centre_gaussian"]:
        spec = pre.arm(name)["spec"].replace(":", "_")
        per_tag[f"{spec}_seed0"] = row
    assert set(by_arm(pre, per_tag, [0])) == scored


@pytest.mark.parametrize("condition", ["nodule_present", "malignancy",
                                       "balanced_presence"])
def test_the_shipped_artefacts_keep_their_stamps_and_containment(condition):
    path = f"runs/nulls_{condition}_confirmatory/arm_in_lung.json"
    try:
        doc = json.loads(open(path).read())
    except FileNotFoundError:
        pytest.skip(f"{path} not generated")
    assert doc["declared_in_prereg"] is False
    assert doc["role"] == "sensitivity_outside_confirmatory_family"
    # protocol.lung_mask's containment 1.0 predicts the restriction never
    # empties a bag of positives; a nonzero count here contradicts a measured
    # selection rule and needs investigating, not averaging away.
    for tag, r in doc["per_tag"].items():
        assert r["n_bags_with_no_lesion_in_lung"] == 0, tag
