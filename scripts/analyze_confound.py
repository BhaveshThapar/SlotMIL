#!/usr/bin/env python
"""Analysis of runs/control_confound.npz -- how much lesion AUC survives once the
positional prior is stratified out.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment

F = sys.argv[1] if len(sys.argv) > 1 else "runs/control_confound.npz"
D = dict(np.load(F, allow_pickle=True))
# frozen lesion-head choice comes from pass 1 (same protocol, same val split)
P1 = {}
for f in sys.argv[2:] or ["runs/control_diag_balanced.npz", "runs/control_diag_rest.npz"]:
    d = np.load(f, allow_pickle=True)
    for k in d.files:
        P1.setdefault(k, []).append(d[k])
P1 = {k: np.concatenate(v) for k, v in P1.items()}

CONDS = ["nodule_present", "malignancy", "balanced_presence"]
ARMS = ["slot_div=0.5", "mh_abmil", "UNTRAINED_slot", "UNTRAINED_mh_abmil"]
MODES = ["raw", "slice", "rad", "joint"]
NBIN = 6


def frozen_head(cond, arm, seed):
    v = ((P1["cond"] == cond) & (P1["arm"] == arm) & (P1["seed"] == seed)
         & (P1["split"] == "val"))
    an = P1["aff_norm"][v].mean(0)
    aff = np.stack([an, 1 - an], axis=1)
    r, c = linear_sum_assignment(-aff)
    a = {int(x): int(y) for x, y in zip(r, c)}
    return next((k for k, val in a.items() if val == 0), 0)


def units(cond, arm):
    return sorted(set(D["seed"][(D["cond"] == cond) & (D["arm"] == arm)].tolist()))


print("=" * 112)
print("Positional confound: lesion AUC of the FROZEN head, TEST split, under four stratifications")
print("  raw = unrestricted (reported protocol) | slice = lesion-bearing slices only")
print("  rad = stratified by in-plane radial bin | joint = stratified by (slice x radial bin)")
print("=" * 112)
print(f"{'condition':<19}{'arm':<21}{'seed':<6}" + "".join(f"{m:>10}" for m in MODES))
agg = {}
for cond in CONDS:
    for arm in ARMS:
        for seed in units(cond, arm):
            t = ((D["cond"] == cond) & (D["arm"] == arm) & (D["seed"] == seed)
                 & (D["split"] == "test"))
            if t.sum() == 0:
                continue
            k = frozen_head(cond, arm, seed)
            vals = [np.nanmean(D[f"auc_{m}"][t][:, k]) for m in MODES]
            agg.setdefault((cond, arm), []).append(vals)
            print(f"{cond:<19}{arm:<21}{seed:<6}" + "".join(f"{v:>10.4f}" for v in vals))
    g = (D["cond"] == cond) & (D["arm"] == "__geo__") & (D["split"] == "test")
    if g.sum():
        for j, nm in enumerate(["geo -|z-zc|", "geo -radius"]):
            print(f"{cond:<19}{nm:<21}{'--':<6}"
                  + "".join(f"{np.nanmean(D['geo_' + m][g][:, j]):>10.4f}" for m in MODES))
    print()

print("=" * 112)
print("Arm means +/- sd across seeds, and the delta, per stratification")
print("=" * 112)
print(f"{'condition':<19}{'arm':<21}" + "".join(f"{m:>16}" for m in MODES))
for cond in CONDS:
    for arm in ARMS:
        if (cond, arm) not in agg:
            continue
        a = np.array(agg[(cond, arm)])
        print(f"{cond:<19}{arm:<21}"
              + "".join(f"{a[:, j].mean():>10.4f}+/-{a[:, j].std():.3f}" for j in range(4)))
    if (cond, "slot_div=0.5") in agg and (cond, "mh_abmil") in agg:
        s, m = np.array(agg[(cond, "slot_div=0.5")]), np.array(agg[(cond, "mh_abmil")])
        line = f"{'  -> delta (p)':<40}"
        for j in range(4):
            p = stats.ttest_ind(s[:, j], m[:, j], equal_var=False).pvalue
            line += f"{s[:, j].mean() - m[:, j].mean():>+9.4f} p={p:.3f}"
        print(line)
    print()

print("=" * 112)
print("Pooled (11 v 11), per stratification")
print("=" * 112)
for j, mname in enumerate(MODES):
    s = np.concatenate([np.array(agg[(c, 'slot_div=0.5')])[:, j] for c in CONDS
                        if (c, 'slot_div=0.5') in agg])
    m = np.concatenate([np.array(agg[(c, 'mh_abmil')])[:, j] for c in CONDS
                        if (c, 'mh_abmil') in agg])
    p = stats.ttest_ind(s, m, equal_var=False).pvalue
    extra = ""
    for tag in ("UNTRAINED_slot", "UNTRAINED_mh_abmil"):
        u = np.concatenate([np.array(agg[(c, tag)])[:, j] for c in CONDS if (c, tag) in agg])
        extra += f"  {tag.split('_', 1)[1]}_untrained={u.mean():.4f}"
    print(f"{mname:<8} slot={s.mean():.4f}+/-{s.std():.3f}  mh={m.mean():.4f}+/-{m.std():.3f}  "
          f"delta={s.mean()-m.mean():+.4f}  p={p:.2e}{extra}")

print()
print("=" * 112)
print("Where does attention live? mean per-instance attention by in-plane radial bin,")
print("normalised so a flat map = 1.0 in every bin. bin0 = image centre, bin5 = periphery.")
print("Also lesion density by bin, same normalisation.")
print("=" * 112)
for cond in CONDS:
    g = (D["cond"] == cond) & (D["arm"] == "__geo__") & (D["split"] == "test")
    pos = D["pos_by_rbin"][g].sum(0)
    nn = D["n_by_rbin"][g].sum(0)
    dens = (pos / nn) / (pos.sum() / nn.sum())
    print(f"{cond:<19}{'LESION DENSITY':<21}" + "".join(f"{x:>8.2f}" for x in dens))
    for arm in ARMS:
        for seed in units(cond, arm):
            t = ((D["cond"] == cond) & (D["arm"] == arm) & (D["seed"] == seed)
                 & (D["split"] == "test"))
            if t.sum() == 0:
                continue
            k = frozen_head(cond, arm, seed)
            prof = D["attn_by_rbin"][t][:, k, :].mean(0)
            prof = prof / prof.mean()
            print(f"{cond:<19}{arm + ' s' + str(seed):<21}"
                  + "".join(f"{x:>8.2f}" for x in prof))
    print()
