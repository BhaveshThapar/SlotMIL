#!/usr/bin/env python
"""Error bars on the untrained baselines: random-init networks under the exact
frozen-Hungarian protocol, five random inits, all three conditions.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment

D = dict(np.load(sys.argv[1] if len(sys.argv) > 1 else "runs/control_untrained.npz",
                 allow_pickle=True))
CONDS = ["nodule_present", "malignancy", "balanced_presence"]
MODES = ["raw", "slice", "rad", "joint"]


def frozen(cond, arm, seed):
    v = ((D["cond"] == cond) & (D["arm"] == arm) & (D["seed"] == seed) & (D["split"] == "val"))
    an = D["aff_norm"][v].mean(0)
    r, c = linear_sum_assignment(-np.stack([an, 1 - an], axis=1))
    a = {int(x): int(y) for x, y in zip(r, c)}
    return next((k for k, val in a.items() if val == 0), 0)


print("=" * 96)
print("Untrained (random-init) networks, frozen-Hungarian protocol, TEST split, 5 inits")
print("=" * 96)
print(f"{'condition':<20}{'arm':<22}" + "".join(f"{m:>16}" for m in MODES))
allv = {}
for cond in CONDS:
    for arm in ["UNTRAINED_slot", "UNTRAINED_mh_abmil"]:
        seeds = sorted(set(D["seed"][(D["cond"] == cond) & (D["arm"] == arm)].tolist()))
        vals = []
        for s in seeds:
            t = ((D["cond"] == cond) & (D["arm"] == arm) & (D["seed"] == s)
                 & (D["split"] == "test"))
            k = frozen(cond, arm, s)
            vals.append([np.nanmean(D[f"auc_{m}"][t][:, k]) for m in MODES])
        v = np.array(vals)
        allv[(cond, arm)] = v
        print(f"{cond:<20}{arm:<22}"
              + "".join(f"{v[:, j].mean():>10.4f}+/-{v[:, j].std():.3f}" for j in range(4))
              + f"  n={len(seeds)}")
print()
print("Pooled over conditions (15 inits x conditions):")
for arm in ["UNTRAINED_slot", "UNTRAINED_mh_abmil"]:
    v = np.concatenate([allv[(c, arm)] for c in CONDS])
    print(f"  {arm:<22}" + "".join(f"{v[:, j].mean():>10.4f}+/-{v[:, j].std():.3f}"
                                   for j in range(4)) + f"  n={len(v)}")
