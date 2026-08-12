#!/usr/bin/env python
"""The one table: {scorer} x {mask condition} for the frozen-slot protocol.

Scorers, from most to least informed by the test image:
  slot           the frozen lesion slot's attention (the published statistic)
  residual       that attention divided by its own per-bag slice-averaged map,
                 so only slice-to-slice (content-dependent) variation survives
  bag_static     that attention averaged over the bag's slices and retiled --
                 one 16x16 map per volume, no slice-to-slice information
  global_static  ONE 16x16 map averaged over every slice of every VALIDATION bag,
                 applied unchanged to every test patch. Reads no test image.
  centre_prior   -distance from the volume centre. No model at all.

Mask conditions:
  real           the bag's own lesion mask
  shuffle        another patient's lesion mask (deranged across bags). Destroys
                 patient-specific correspondence but PRESERVES the population
                 prior on where in the frame a nodule sits.
  roll           the bag's own mask circularly shifted. Destroys the population
                 prior too, while preserving per-bag prevalence exactly.

Reading the table: real-minus-shuffle is the part of the score that is specific
to this patient's nodule; shuffle-minus-roll is the part explained by nodules
being in anatomically stereotyped places; roll is the metric's floor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score

from null_battery import (N_PATCH, load, pick_lesion_slot, roll_masks,
                          shuffle_masks_across_bags)
from null_static_template import global_template

GRID = 16
SCORERS = ["slot", "residual", "bag_static", "global_static", "centre_prior"]


def centre_map():
    yy, xx = np.meshgrid(np.linspace(-1, 1, GRID), np.linspace(-1, 1, GRID), indexing="ij")
    return np.sqrt(yy.ravel() ** 2 + xx.ravel() ** 2)


def build_scores(a_slot, n, gtemplate, cmap):
    ns = n // N_PATCH
    s = a_slot[: ns * N_PATCH].reshape(ns, N_PATCH)
    bt = s.mean(axis=0)
    z = np.repeat(np.linspace(-1, 1, ns), N_PATCH)
    return {
        "slot": s.ravel(),
        "residual": (s / np.clip(bt, 1e-12, None)[None, :]).ravel(),
        "bag_static": np.tile(bt, ns),
        "global_static": np.tile(gtemplate, ns),
        "centre_prior": -np.sqrt(z ** 2 + np.tile(cmap, ns) ** 2),
    }, ns


def run(attns, masks_by_cond, slot, gtemplate):
    cmap = centre_map()
    acc = {c: {k: [] for k in SCORERS} for c in masks_by_cond}
    for i, a in enumerate(attns):
        n = len(masks_by_cond["real"][i])
        if n % N_PATCH:
            continue
        sc, ns = build_scores(a[slot], n, gtemplate, cmap)
        for cond, masks in masks_by_cond.items():
            t = (masks[i] > 0).astype(np.int8)[: ns * N_PATCH]
            if t.sum() == 0 or t.sum() == len(t):
                continue
            for k in SCORERS:
                acc[cond][k].append(roc_auc_score(t, sc[k]))
    return {c: {k: np.asarray(v) for k, v in d.items()} for c, d in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/nulls")
    ap.add_argument("--tags", nargs="+", default=["real_seed0", "real_seed1", "real_seed2"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="runs/nulls/crosstab.json")
    args = ap.parse_args()

    d, R = Path(args.dir), {}
    for tag in args.tags:
        va, vm = load(d / f"{tag}_val.npz")
        ta, tm = load(d / f"{tag}_test.npz")
        slot, _ = pick_lesion_slot(va, vm)
        gt = global_template(va, vm, slot)

        per_cond = {c: {k: [] for k in SCORERS} for c in ("real", "shuffle", "roll")}
        for r in range(args.reps):
            rr = np.random.default_rng(7000 + r)
            conds = {"real": tm,
                     "shuffle": shuffle_masks_across_bags(tm, rr),
                     "roll": roll_masks(tm, rr)}
            out = run(ta, conds, slot, gt)
            for c in per_cond:
                for k in SCORERS:
                    per_cond[c][k].append(float(out[c][k].mean()))
            if r == 0:
                real_bagwise = out["real"]

        R[tag] = {
            "lesion_slot": int(slot),
            "n_bags": int(len(real_bagwise["slot"])),
            "table": {c: {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                          for k, v in dd.items()} for c, dd in per_cond.items()},
            "paired_slot_vs_global_static_real": {
                "mean_diff": float((real_bagwise["slot"] - real_bagwise["global_static"]).mean()),
                "t_p": float(ttest_rel(real_bagwise["slot"], real_bagwise["global_static"]).pvalue),
            },
        }
        print(f"\n[{tag}] lesion_slot={slot}  n_bags={R[tag]['n_bags']}", flush=True)
        print(f"  {'scorer':<15}{'real':>10}{'shuffle':>10}{'roll':>10}"
              f"{'real-shuf':>12}{'shuf-roll':>12}", flush=True)
        for k in SCORERS:
            re_, sh, ro = (np.mean(per_cond[c][k]) for c in ("real", "shuffle", "roll"))
            print(f"  {k:<15}{re_:>10.4f}{sh:>10.4f}{ro:>10.4f}"
                  f"{re_-sh:>12.4f}{sh-ro:>12.4f}", flush=True)
        del va, vm, ta, tm

    Path(args.out).write_text(json.dumps(R, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
