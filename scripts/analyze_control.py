#!/usr/bin/env python
"""Offline analysis of runs/control_diag_*.npz produced by scripts/diagnose_control.py.

Answers, per condition/arm/seed:
  Q1  does the affinity renormalisation asymmetry matter?
  Q2  full per-head instance-AUC distribution (is EVERY head below chance?)
  Q3  is the K x 2 affinity matrix degenerate / is the ordering informative?
  Q4  do fairer selection rules (val-AUC head selection, sign-calibrated) rescue
      the control, and by how much does the reported delta change?
"""

from __future__ import annotations

import glob
import sys

import numpy as np
from scipy import stats

FILES = sys.argv[1:] or sorted(glob.glob("runs/control_diag_*.npz"))


def load():
    rows = {}
    for f in FILES:
        d = np.load(f, allow_pickle=True)
        for k in d.files:
            rows.setdefault(k, []).append(d[k])
    return {k: np.concatenate(v) for k, v in rows.items()}


D = load()
CONDS = ["nodule_present", "malignancy", "balanced_presence"]
ARMS = ["slot_div=0.5", "mh_abmil", "UNTRAINED_slot", "UNTRAINED_mh_abmil"]


def sel(cond, arm, seed, split):
    return (D["cond"] == cond) & (D["arm"] == arm) & (D["seed"] == seed) & (D["split"] == split)


def units(cond, arm):
    return sorted(set(D["seed"][(D["cond"] == cond) & (D["arm"] == arm)].tolist()))


def hungarian_lesion_head(aff_norm_mean):
    """Reproduce fit_slot_assignment for the F=2 complementary case.

    affinity[:,0] = mean lesion mass share, affinity[:,1] = 1 - that. With a
    rectangular K x 2 cost, linear_sum_assignment returns min(K,2)=2 pairs, and
    maximising aff[k0,0] + (1 - aff[k1,0]) is exactly argmax / argmin of column 0.
    """
    from scipy.optimize import linear_sum_assignment
    aff = np.stack([aff_norm_mean, 1 - aff_norm_mean], axis=1)
    r, c = linear_sum_assignment(-aff)
    a = {int(rr): int(cc) for rr, cc in zip(r, c)}
    return next((k for k, v in a.items() if v == 0), 0), aff, a


print("=" * 100)
print("Q3 / Q1 -- validation affinity matrices (the Hungarian objective) and the renorm asymmetry")
print("=" * 100)
for cond in CONDS:
    for arm in ARMS:
        for seed in units(cond, arm):
            v = sel(cond, arm, seed, "val")
            if v.sum() == 0:
                continue
            an = D["aff_norm"][v].mean(0)          # what slot_finding_affinity computes
            ar = (D["aff_raw"][v] / D["mass"][v].clip(1e-12)).mean(0)
            prev = D["prev"][v].mean()
            k_h, aff, assign = hungarian_lesion_head(an)
            k_raw = int(np.argmax(D["aff_raw"][v].mean(0)))     # no renormalisation
            spread = an.max() / max(an.min(), 1e-12)
            print(f"{cond:<18}{arm:<20}s{seed}  prev={prev:.6f}  "
                  f"col0=[{' '.join(f'{x:.5f}' for x in an)}]")
            print(f"{'':<38}  lesion head Hungarian={k_h}  argmax-col0={int(np.argmax(an))}  "
                  f"argmax-unnormalised={k_raw}  max/min={spread:.2f}  "
                  f"all-heads-below-prevalence={bool((an < prev).all())}  "
                  f"assign={assign}")


print()
print("=" * 100)
print("Q2 -- full per-head TEST instance AUC (mean over bags), all heads")
print("=" * 100)
for cond in CONDS:
    for arm in ARMS:
        for seed in units(cond, arm):
            t = sel(cond, arm, seed, "test")
            if t.sum() == 0:
                continue
            per_head = np.nanmean(D["auc"][t], axis=0)
            ent = D["ent"][t].mean(0)
            print(f"{cond:<18}{arm:<20}s{seed}  "
                  f"heads=[{' '.join(f'{x:.3f}' for x in per_head)}]  "
                  f"min={per_head.min():.3f} max={per_head.max():.3f} "
                  f"n>0.5={int((per_head > 0.5).sum())}/8  meanH={ent.mean():.3f}")


print()
print("=" * 100)
print("Q4 -- test AUC under four selection rules")
print("  R1 Hungarian affinity on val (CURRENT PROTOCOL)")
print("  R2 argmax val per-head instance AUC (fair, uses the same statistic as the metric)")
print("  R3 sign-calibrated: argmax val |AUC-0.5|, score flipped on test if val AUC<0.5")
print("  R4 per-bag oracle over heads on test (upper bound, not reportable)")
print("=" * 100)
res = {}
for cond in CONDS:
    for arm in ARMS:
        for seed in units(cond, arm):
            v, t = sel(cond, arm, seed, "val"), sel(cond, arm, seed, "test")
            if v.sum() == 0 or t.sum() == 0:
                continue
            val_auc = np.nanmean(D["auc"][v], axis=0)
            test_auc = np.nanmean(D["auc"][t], axis=0)
            k1, _, _ = hungarian_lesion_head(D["aff_norm"][v].mean(0))
            k2 = int(np.argmax(val_auc))
            k3 = int(np.argmax(np.abs(val_auc - 0.5)))
            r3 = test_auc[k3] if val_auc[k3] >= 0.5 else 1 - test_auc[k3]
            r4 = np.nanmean(np.nanmax(D["auc"][t], axis=1))
            res[(cond, arm, seed)] = dict(R1=test_auc[k1], R2=test_auc[k2], R3=r3, R4=r4,
                                          k1=k1, k2=k2, k3=k3,
                                          val_best=val_auc.max(), val_worst=val_auc.min())
            print(f"{cond:<18}{arm:<20}s{seed}  R1={test_auc[k1]:.4f}(k={k1})  "
                  f"R2={test_auc[k2]:.4f}(k={k2})  R3={r3:.4f}(k={k3})  R4={r4:.4f}  "
                  f"| val best head={val_auc.max():.4f} worst={val_auc.min():.4f}")


print()
print("=" * 100)
print("Q4 summary -- arm means +/- sd across seeds, and the SlotMIL-minus-control delta")
print("=" * 100)
hdr = f"{'condition':<20}{'arm':<20}" + "".join(f"{r:>18}" for r in ("R1 Hungarian", "R2 val-AUC", "R3 signcal", "R4 oracle"))
print(hdr)
agg = {}
for cond in CONDS:
    for arm in ARMS:
        ks = [k for k in res if k[0] == cond and k[1] == arm]
        if not ks:
            continue
        row = {r: np.array([res[k][r] for k in ks]) for r in ("R1", "R2", "R3", "R4")}
        agg[(cond, arm)] = row
        print(f"{cond:<20}{arm:<20}" + "".join(
            f"{row[r].mean():>11.4f}+/-{row[r].std():.3f}" for r in ("R1", "R2", "R3", "R4")))
    if (cond, "slot_div=0.5") in agg and (cond, "mh_abmil") in agg:
        s, m = agg[(cond, "slot_div=0.5")], agg[(cond, "mh_abmil")]
        line = f"{'  -> delta (p Welch)':<40}"
        for r in ("R1", "R2", "R3", "R4"):
            p = stats.ttest_ind(s[r], m[r], equal_var=False).pvalue
            line += f"{s[r].mean() - m[r].mean():>+10.4f} p={p:.3f}"
        print(line)
        print()

print()
print("=" * 100)
print("Pooled across conditions (the '11 v 11' comparison), per rule")
print("=" * 100)
for r in ("R1", "R2", "R3", "R4"):
    s = np.concatenate([agg[(c, "slot_div=0.5")][r] for c in CONDS if (c, "slot_div=0.5") in agg])
    m = np.concatenate([agg[(c, "mh_abmil")][r] for c in CONDS if (c, "mh_abmil") in agg])
    t = stats.ttest_ind(s, m, equal_var=False)
    u = stats.mannwhitneyu(s, m, alternative="two-sided")
    print(f"{r:<14} slot={s.mean():.4f}+/-{s.std():.3f} (n={len(s)})  "
          f"mh={m.mean():.4f}+/-{m.std():.3f} (n={len(m)})  "
          f"delta={s.mean()-m.mean():+.4f}  p_welch={t.pvalue:.2e}  p_mwu={u.pvalue:.4f}")
    for tag in ("UNTRAINED_slot", "UNTRAINED_mh_abmil"):
        if (CONDS[0], tag) in agg:
            u_ = np.concatenate([agg[(c, tag)][r] for c in CONDS if (c, tag) in agg])
            print(f"{'':<14} {tag:<22} {u_.mean():.4f}+/-{u_.std():.3f}")

print()
print("=" * 100)
print("Geometry baselines (positional priors, model-free) -- test split, mean over bags")
print("=" * 100)
for cond in CONDS:
    t = (D["cond"] == cond) & (D["split"] == "test") & (D["arm"] == "mh_abmil")
    if t.sum() == 0:
        continue
    # one row per (bag, seed); dedupe on bag
    bags, idx = np.unique(D["bag"][t], return_index=True)
    z = D["geo_auc_z"][t][idx]
    r = D["geo_auc_r"][t][idx]
    print(f"{cond:<20} AUC(-|z-zcentre|)={np.nanmean(z):.4f}   "
          f"AUC(-radius from image centre)={np.nanmean(r):.4f}   n_bags={len(bags)}")

print()
print("=" * 100)
print("Attention concentration and head redundancy proxies (test)")
print("=" * 100)
for cond in CONDS:
    for arm in ARMS:
        for seed in units(cond, arm):
            t = sel(cond, arm, seed, "test")
            if t.sum() == 0:
                continue
            print(f"{cond:<18}{arm:<20}s{seed}  meanNormEntropy={D['ent'][t].mean():.4f}  "
                  f"headEntropySpread=[{D['ent'][t].mean(0).min():.3f},{D['ent'][t].mean(0).max():.3f}]  "
                  f"gate=[{' '.join(f'{x:.2f}' for x in D['gate'][t].mean(0))}]")
