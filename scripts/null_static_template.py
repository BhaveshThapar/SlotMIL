#!/usr/bin/env python
"""Is the frozen lesion slot a lesion detector, or a static in-plane template?

Motivated by the decomposition in scripts/null_decompose.py, which found for
runs/lidc/slot_div=0.5/seed0 that the frozen slot's instance AUC of 0.8424
splits into slice_auc = 0.4822 (chance at saying WHICH slice holds the nodule)
and within_slice_auc = 0.8411 (strong at saying WHERE in a lesion-bearing slice).
A detector that cannot find the lesion's z but can find its (x, y) is behaving
like a fixed in-plane spatial prior applied identically to every slice.

Three progressively stricter reductions of the slot to a content-free template:

  bag_static     -- average the slot's attention over the slices of THAT bag to
                    get one 16x16 map, retile it over every slice, rescore. Any
                    slice-to-slice, i.e. content-dependent, variation is deleted.
  global_static  -- one 16x16 map averaged over every slice of every VALIDATION
                    bag, applied unchanged to every test patch. No test image is
                    read at all: this is a single 256-number lookup table fit on
                    val, the most content-free predictor that still uses the
                    trained model.
  residual       -- attention with its bag-static template divided out, so only
                    the slice-to-slice variation survives. If the slot is a real
                    detector this keeps most of the AUC; if it is a template this
                    collapses to chance.

The comparison is per-bag paired against the slot's own AUC, because the
question is not "is the template above chance" (it is) but "does the slot beat
its own template".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import roc_auc_score

from null_battery import N_PATCH, load, pick_lesion_slot

GRID = 16


def bag_arrays(a_slot, m):
    n = len(m)
    ns = n // N_PATCH
    return a_slot[: ns * N_PATCH].reshape(ns, N_PATCH), (m > 0).astype(np.int8)[: ns * N_PATCH].reshape(ns, N_PATCH), ns


def global_template(attns, masks, slot):
    """Mean 16x16 attention map over every slice of every bag. Fit on VAL."""
    acc = np.zeros(N_PATCH, dtype=np.float64)
    n = 0
    for a, m in zip(attns, masks):
        s, _, ns = bag_arrays(a[slot], m)
        acc += s.sum(axis=0)
        n += ns
    return acc / n


def evaluate(attns, masks, slot, gtemplate):
    rows = {"slot": [], "bag_static": [], "global_static": [], "residual": []}
    for a, m in zip(attns, masks):
        s, t, ns = bag_arrays(a[slot], m)
        tt = t.ravel()
        if tt.sum() == 0 or tt.sum() == len(tt):
            continue
        rows["slot"].append(roc_auc_score(tt, s.ravel()))
        bt = s.mean(axis=0)
        rows["bag_static"].append(roc_auc_score(tt, np.tile(bt, ns)))
        rows["global_static"].append(roc_auc_score(tt, np.tile(gtemplate, ns)))
        res = s / np.clip(bt, 1e-12, None)[None, :]
        rows["residual"].append(roc_auc_score(tt, res.ravel()))
    return {k: np.asarray(v) for k, v in rows.items()}


def paired(a, b):
    d = a - b
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "mean_diff": float(d.mean()),
            "sem_diff": float(d.std(ddof=1) / np.sqrt(len(d))),
            "frac_a_better": float((d > 0).mean()),
            "t_p": float(ttest_rel(a, b).pvalue),
            "wilcoxon_p": float(wilcoxon(a, b).pvalue),
            "n_bags": int(len(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/nulls")
    ap.add_argument("--tags", nargs="+", default=["real_seed0"])
    ap.add_argument("--out", default="runs/nulls/static_template.json")
    args = ap.parse_args()

    d, R = Path(args.dir), {}
    for tag in args.tags:
        va, vm = load(d / f"{tag}_val.npz")
        ta, tm = load(d / f"{tag}_test.npz")
        slot, _ = pick_lesion_slot(va, vm)
        gt = global_template(va, vm, slot)          # fit on VAL only
        r = evaluate(ta, tm, slot, gt)

        R[tag] = {
            "lesion_slot": int(slot),
            "means": {k: float(v.mean()) for k, v in r.items()},
            "slot_vs_bag_static": paired(r["slot"], r["bag_static"]),
            "slot_vs_global_static": paired(r["slot"], r["global_static"]),
            "template_spatial_concentration": {
                "top16_share_of_template_mass": float(
                    np.sort(gt)[::-1][:16].sum() / gt.sum()),
                "template_min": float(gt.min()), "template_max": float(gt.max()),
            },
        }
        print(f"[{tag}] slot={slot}", flush=True)
        for k, v in r.items():
            print(f"   {k:<14} test AUC {v.mean():.4f}", flush=True)
        p = R[tag]["slot_vs_global_static"]
        print(f"   slot - global_static = {p['mean_diff']:+.4f} +/- {p['sem_diff']:.4f} "
              f"(slot better in {p['frac_a_better']:.2f} of bags, t_p={p['t_p']:.3g})",
              flush=True)
        p = R[tag]["slot_vs_bag_static"]
        print(f"   slot - bag_static    = {p['mean_diff']:+.4f} +/- {p['sem_diff']:.4f} "
              f"(slot better in {p['frac_a_better']:.2f} of bags, t_p={p['t_p']:.3g})",
              flush=True)
        del va, vm, ta, tm

    Path(args.out).write_text(json.dumps(R, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
