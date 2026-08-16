#!/usr/bin/env python
"""Decompose the frozen-slot instance AUC into its z and in-plane components.

Not one of the four requested nulls, but the confound they do not cover. A slot
that attends to "lung parenchyma near the middle of the volume" scores a high
instance AUC without ever having found a lesion, because lesion patches are a
subset of the region it likes. Two conditional statistics separate that from
genuine localisation:

  slice_auc      -- per-bag AUC of the slot's mean attention per slice against
                    "this slice contains lesion". Pure z-localisation.
  within_slice   -- per-bag AUC over ONLY the patches belonging to
                    lesion-containing slices. The z information is held fixed by
                    construction, so this is in-plane localisation alone.

If the headline 0.84 is anatomy, within_slice collapses to ~0.5 while slice_auc
stays high. If the slot really binds the lesion, both stay above chance. Each is
also computed under the circular-roll mask null, so each has its own null.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from null_battery import N_PATCH, load, pick_lesion_slot, roll_masks, shuffle_masks_across_bags
from sklearn.metrics import roc_auc_score

from slotmil.eval.axes import per_bag_axes


def decompose(attns, masks, slot):
    """Pooled means of the per-bag axis split.

    The split itself is :func:`slotmil.eval.axes.per_bag_axes`; this file used to
    carry a second copy of it that had drifted into pooling as it went, which is
    why the two could not be compared. Pooling here rather than in the library is
    deliberate: the per-bag rows are what a patient-level bootstrap needs, and
    ``axis_gate.py`` wants them, so the library returns rows and the averaging
    stays at the call site that actually wants an average.

    A bag that is degenerate on one axis carries ``nan`` in that column, so each
    axis is averaged over its own finite rows -- not over a common subset. That
    is what the original did by appending to three separate lists, and it is why
    ``n_flat``, ``n_slice`` and ``n_within`` are reported separately.
    """
    rows = per_bag_axes(attns, masks, slot)
    flat = [r[1] for r in rows]
    slice_auc = [r[2] for r in rows if np.isfinite(r[2])]
    within = [r[3] for r in rows if np.isfinite(r[3])]
    return {
        "instance_auc_flat": float(np.mean(flat)), "n_flat": len(flat),
        "slice_auc": float(np.mean(slice_auc)), "n_slice": len(slice_auc),
        "within_slice_auc": float(np.mean(within)), "n_within": len(within),
        "mean_lesion_slices_per_bag": None,
    }


def centre_prior_scores(masks):
    """The lesion-agnostic geometric heuristic, in the same [K,N] shape as
    attention (K=1) so it can go through the identical decomposition."""
    yy, xx = np.meshgrid(np.linspace(-1, 1, 16), np.linspace(-1, 1, 16), indexing="ij")
    inplane = np.sqrt(yy.ravel() ** 2 + xx.ravel() ** 2)
    out = []
    for m in masks:
        n = len(m)
        ns = int(np.ceil(n / N_PATCH))
        z = np.repeat(np.linspace(-1, 1, ns), N_PATCH)[:n]
        d = np.sqrt(z ** 2 + np.tile(inplane, ns)[:n] ** 2)
        out.append((-d)[None, :].astype(np.float32))
    return out


def paired_vs_prior(attns, masks, slot, priors):
    """Per-bag paired comparison of the frozen slot against the centre prior.

    The centre prior is not a null in the "no signal" sense -- it is a fixed
    lesion-agnostic rule that happens to score well because lesions live in the
    middle of the volume. If the slot does not beat it bag-for-bag, the headline
    AUC is measuring anatomy, not lesion binding."""
    from scipy.stats import ttest_rel, wilcoxon
    a_s, a_p = [], []
    for a, m, p in zip(attns, masks, priors):
        t = (m > 0).astype(np.int8)
        s = int(t.sum())
        if s == 0 or s == len(t):
            continue
        a_s.append(roc_auc_score(t, a[slot]))
        a_p.append(roc_auc_score(t, p[0]))
    a_s, a_p = np.asarray(a_s), np.asarray(a_p)
    d = a_s - a_p
    return {
        "slot_auc": float(a_s.mean()), "prior_auc": float(a_p.mean()),
        "mean_diff": float(d.mean()),
        "sem_diff": float(d.std(ddof=1) / np.sqrt(len(d))),
        "frac_bags_slot_better": float((d > 0).mean()),
        "t_p": float(ttest_rel(a_s, a_p).pvalue),
        "wilcoxon_p": float(wilcoxon(a_s, a_p).pvalue),
        "n_bags": int(len(d)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/nulls")
    ap.add_argument("--tags", nargs="+", default=["real_seed0"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default="runs/nulls/decompose.json")
    args = ap.parse_args()

    d, R = Path(args.dir), {}
    done_prior = False
    for tag in args.tags:
        va, vm = load(d / f"{tag}_val.npz")
        ta, tm = load(d / f"{tag}_test.npz")

        if not done_prior:
            cp = decompose(centre_prior_scores(tm), tm, 0)
            R["centre_prior"] = cp
            print(f"[centre_prior] flat={cp['instance_auc_flat']:.4f} "
                  f"slice={cp['slice_auc']:.4f} within_slice={cp['within_slice_auc']:.4f}",
                  flush=True)
            done_prior = True

        slot, _ = pick_lesion_slot(va, vm)
        real = decompose(ta, tm, slot)
        real["lesion_slot"] = int(slot)
        R[tag] = {"real": real}
        pv = paired_vs_prior(ta, tm, slot, centre_prior_scores(tm))
        R[tag]["paired_vs_centre_prior"] = pv
        print(f"[{tag}] slot vs centre-prior: {pv['slot_auc']:.4f} vs {pv['prior_auc']:.4f} "
              f"diff={pv['mean_diff']:+.4f}+/-{pv['sem_diff']:.4f} "
              f"better_in={pv['frac_bags_slot_better']:.2f} of bags  "
              f"t_p={pv['t_p']:.3g} wilcoxon_p={pv['wilcoxon_p']:.3g}", flush=True)
        print(f"[{tag}] slot={slot} flat={real['instance_auc_flat']:.4f} "
              f"slice={real['slice_auc']:.4f} (n={real['n_slice']}) "
              f"within_slice={real['within_slice_auc']:.4f} (n={real['n_within']})",
              flush=True)

        for name, fn in (("circular_roll", roll_masks),
                         ("shuffle_across_bags", shuffle_masks_across_bags)):
            runs = []
            for r in range(args.reps):
                rr = np.random.default_rng(9000 + r)
                pvm, ptm = fn(vm, rr), fn(tm, rr)
                ps, _ = pick_lesion_slot(va, pvm)
                runs.append(decompose(ta, ptm, ps))
            R[tag][f"null_{name}"] = {
                k: float(np.mean([x[k] for x in runs]))
                for k in ("instance_auc_flat", "slice_auc", "within_slice_auc")
            }
            n = R[tag][f"null_{name}"]
            print(f"  null[{name}] flat={n['instance_auc_flat']:.4f} "
                  f"slice={n['slice_auc']:.4f} within_slice={n['within_slice_auc']:.4f}",
                  flush=True)

    Path(args.out).write_text(json.dumps(R, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
