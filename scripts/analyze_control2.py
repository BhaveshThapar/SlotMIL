#!/usr/bin/env python
"""Per-bag supplementary analysis on runs/control_diag_*.npz.

How systematic is the below-chance result -- is it a few bags dragging the mean,
or is nearly every bag anti-correlated? And is SlotMIL's advantage present bag by
bag, or driven by the same handful of bags?
"""

from __future__ import annotations

import glob
import sys

import numpy as np
from scipy import stats

FILES = sys.argv[1:] or sorted(glob.glob("runs/control_diag_*.npz"))
rows = {}
for f in FILES:
    d = np.load(f, allow_pickle=True)
    for k in d.files:
        rows.setdefault(k, []).append(d[k])
D = {k: np.concatenate(v) for k, v in rows.items()}
CONDS = ["nodule_present", "malignancy", "balanced_presence"]
ARMS = ["slot_div=0.5", "mh_abmil", "UNTRAINED_slot", "UNTRAINED_mh_abmil"]


def sel(cond, arm, seed, split):
    return (D["cond"] == cond) & (D["arm"] == arm) & (D["seed"] == seed) & (D["split"] == split)


def units(cond, arm):
    return sorted(set(D["seed"][(D["cond"] == cond) & (D["arm"] == arm)].tolist()))


def hung(an):
    from scipy.optimize import linear_sum_assignment
    aff = np.stack([an, 1 - an], axis=1)
    r, c = linear_sum_assignment(-aff)
    a = {int(x): int(y) for x, y in zip(r, c)}
    return next((k for k, v in a.items() if v == 0), 0)


print("=" * 104)
print("Per-bag systematicity of the frozen-head score (R1 protocol), TEST split")
print("frac<0.5 = fraction of bags where the frozen head's AUC is below chance; "
      "binomial p vs 0.5")
print("=" * 104)
store = {}
for cond in CONDS:
    for arm in ARMS:
        for seed in units(cond, arm):
            v, t = sel(cond, arm, seed, "val"), sel(cond, arm, seed, "test")
            if v.sum() == 0 or t.sum() == 0:
                continue
            k = hung(D["aff_norm"][v].mean(0))
            per_bag = D["auc"][t][:, k]
            store[(cond, arm, seed)] = per_bag
            frac = float((per_bag < 0.5).mean())
            p = stats.binomtest(int((per_bag < 0.5).sum()), len(per_bag), 0.5).pvalue
            print(f"{cond:<18}{arm:<20}s{seed}  head={k}  mean={per_bag.mean():.4f}  "
                  f"median={np.median(per_bag):.4f}  frac<0.5={frac:.3f}  "
                  f"binom p={p:.2e}  n_bags={len(per_bag)}")

print()
print("=" * 104)
print("Paired per-bag comparison, matched bags: SlotMIL frozen slot vs mh_abmil frozen head")
print("=" * 104)
for cond in CONDS:
    ss = [store[k] for k in store if k[0] == cond and k[1] == "slot_div=0.5"]
    mm = [store[k] for k in store if k[0] == cond and k[1] == "mh_abmil"]
    if not ss or not mm:
        continue
    s = np.mean(np.stack(ss), axis=0)   # seed-averaged, per bag
    m = np.mean(np.stack(mm), axis=0)
    w = stats.wilcoxon(s, m)
    print(f"{cond:<20} slot={s.mean():.4f} mh={m.mean():.4f} "
          f"frac bags slot>mh = {(s > m).mean():.3f}  wilcoxon p={w.pvalue:.2e}  n={len(s)}")
    # is the SlotMIL slot better than the model-free centre prior on the same bags?
    g = D["geo_auc_r"][(D["cond"] == cond) & (D["split"] == "test") & (D["arm"] == "mh_abmil")
                       & (D["seed"] == units(cond, "mh_abmil")[0])]
    wg = stats.wilcoxon(s, g)
    print(f"{'':<20} vs centre-prior baseline: geo={g.mean():.4f}  "
          f"frac bags slot>geo = {(s > g).mean():.3f}  wilcoxon p={wg.pvalue:.2e}")

print()
print("=" * 104)
print("Head-level: correlation between a head's val AUC and its val lesion-mass affinity")
print("(does the Hungarian criterion track the metric it is used to select for?)")
print("=" * 104)
for cond in CONDS:
    for arm in ("slot_div=0.5", "mh_abmil"):
        rs = []
        for seed in units(cond, arm):
            v = sel(cond, arm, seed, "val")
            if v.sum() == 0:
                continue
            a = np.nanmean(D["auc"][v], axis=0)
            f = D["aff_norm"][v].mean(0)
            rs.append(stats.spearmanr(a, f).statistic)
        if rs:
            print(f"{cond:<20}{arm:<20} spearman(head val AUC, head val affinity) "
                  f"per seed = [{' '.join(f'{x:+.2f}' for x in rs)}]  mean={np.mean(rs):+.2f}")

print()
print("=" * 104)
print("Val->test stability of the per-head AUC ordering (is the frozen choice transferable?)")
print("=" * 104)
for cond in CONDS:
    for arm in ARMS:
        rs = []
        for seed in units(cond, arm):
            v, t = sel(cond, arm, seed, "val"), sel(cond, arm, seed, "test")
            if v.sum() == 0 or t.sum() == 0:
                continue
            rs.append(stats.spearmanr(np.nanmean(D["auc"][v], 0), np.nanmean(D["auc"][t], 0)).statistic)
        if rs:
            print(f"{cond:<20}{arm:<22} spearman(val head AUC, test head AUC) = "
                  f"[{' '.join(f'{x:+.2f}' for x in rs)}]  mean={np.mean(rs):+.2f}")
