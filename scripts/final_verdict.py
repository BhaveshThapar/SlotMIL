#!/usr/bin/env python
"""Consolidated verdict table: reported protocol vs fairer / confound-controlled ones.

**This is the discovery-era table, not the pre-registration scorer.** It reads
``runs/control_*.npz`` from the seed-0 split, hardcodes ``CONDS``, and answers
"does the reported protocol survive a fairer one?" -- a question that predates the
freeze. The pre-registered hypotheses are scored by
``scripts/prereg_verdict.py`` against stamped artefacts, with every threshold read
from ``configs/prereg/isbi2027.yaml``. Do not extend this file into that one: its
inputs carry no prereg stamp, so nothing it prints can be confirmatory, and the name
is close enough to invite the mistake.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment

CF = dict(np.load("runs/control_confound.npz", allow_pickle=True))
UT = dict(np.load("runs/control_untrained.npz", allow_pickle=True))
CONDS = ["nodule_present", "malignancy", "balanced_presence"]

# runs/control_confound.npz predates the aff_norm field; the frozen head for the
# trained checkpoints comes from the pass-1 dump, which used the identical val split.
_p1 = {}
for f in ["runs/control_diag_balanced.npz", "runs/control_diag_rest.npz"]:
    d = np.load(f, allow_pickle=True)
    for k in d.files:
        _p1.setdefault(k, []).append(d[k])
P1 = {k: np.concatenate(v) for k, v in _p1.items()}


def frozen(D, cond, arm, seed):
    src = D if "aff_norm" in D else P1
    v = ((src["cond"] == cond) & (src["arm"] == arm) & (src["seed"] == seed)
         & (src["split"] == "val"))
    an = src["aff_norm"][v].mean(0)
    r, c = linear_sum_assignment(-np.stack([an, 1 - an], axis=1))
    a = {int(x): int(y) for x, y in zip(r, c)}
    return next((k for k, val in a.items() if val == 0), 0)


def collect(D, arm, mode, rule):
    """rule: 'hung' | 'valauc' | 'signcal'."""
    out = []
    for cond in CONDS:
        for seed in sorted(set(D["seed"][(D["cond"] == cond) & (D["arm"] == arm)].tolist())):
            v = (D["cond"] == cond) & (D["arm"] == arm) & (D["seed"] == seed) & (D["split"] == "val")
            t = (D["cond"] == cond) & (D["arm"] == arm) & (D["seed"] == seed) & (D["split"] == "test")
            if v.sum() == 0 or t.sum() == 0:
                continue
            va = np.nanmean(D[f"auc_{mode}"][v], axis=0)
            ta = np.nanmean(D[f"auc_{mode}"][t], axis=0)
            if rule == "hung":
                out.append(ta[frozen(D, cond, arm, seed)])
            elif rule == "valauc":
                out.append(ta[int(np.argmax(va))])
            else:
                k = int(np.argmax(np.abs(va - 0.5)))
                out.append(ta[k] if va[k] >= 0.5 else 1 - ta[k])
    return np.array(out)


print("=" * 108)
print("FINAL: pooled-over-conditions test instance AUC (11 SlotMIL ckpts, 11 MH-ABMIL ckpts,")
print("15 untrained inits), under selection rule x confound stratification")
print("=" * 108)
hdr = f"{'rule':<12}{'stratification':<16}{'SlotMIL':>18}{'MH-ABMIL':>18}{'delta':>10}{'p':>11}"
hdr += f"{'untrained slot':>18}{'untrained mh':>18}"
print(hdr)
for rule, rname in [("hung", "R1 Hungarian"), ("valauc", "R2 val-AUC"), ("signcal", "R3 signcal")]:
    for mode in ["raw", "joint"]:
        s = collect(CF, "slot_div=0.5", mode, rule)
        m = collect(CF, "mh_abmil", mode, rule)
        us = collect(UT, "UNTRAINED_slot", mode, rule)
        um = collect(UT, "UNTRAINED_mh_abmil", mode, rule)
        p = stats.ttest_ind(s, m, equal_var=False).pvalue
        print(f"{rname:<12}{mode:<16}{s.mean():>11.4f}+/-{s.std():.3f}"
              f"{m.mean():>11.4f}+/-{m.std():.3f}{s.mean()-m.mean():>+10.4f}{p:>11.2e}"
              f"{us.mean():>11.4f}+/-{us.std():.3f}{um.mean():>11.4f}+/-{um.std():.3f}")

print()
print("=" * 108)
print("Is trained SlotMIL better than an UNTRAINED network under the same protocol?")
print("=" * 108)
for mode in ["raw", "slice", "rad", "joint"]:
    s = collect(CF, "slot_div=0.5", mode, "hung")
    for tag, U in [("untrained slot", "UNTRAINED_slot"), ("untrained mh_abmil", "UNTRAINED_mh_abmil")]:
        u = collect(UT, U, mode, "hung")
        p = stats.ttest_ind(s, u, equal_var=False).pvalue
        print(f"  {mode:<8} slot {s.mean():.4f} vs {tag:<20} {u.mean():.4f}  "
              f"delta={s.mean()-u.mean():+.4f}  p={p:.3f}")
    print()
