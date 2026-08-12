#!/usr/bin/env python
"""WP2 gate: does the axial/in-plane dissociation survive proper sampling?

The paper's lead claim is that flat instance AUC (~0.84) contains *no* 3D
localisation -- slice AUC 0.4822, pure chance at saying which slice holds the
nodule. That number currently comes from one seed, one arm, in
``runs/nulls/decompose_seed0.json``. Before it can lead a paper it has to hold
across every arm and seed available, with an interval attached.

This runs the frozen-slot protocol (fit the lesion slot on val, freeze, score on
test) over every attention dump in ``runs/nulls`` and reports, per bag:

    flat AUC          -- lesion patch vs the rest, the reported statistic
    slice AUC         -- mean attention per slice vs "this slice has a nodule"
    within-slice AUC  -- lesion patch vs the rest, restricted to lesion slices

with a **patient-level cluster bootstrap** over bags. Patients, not bags: a LIDC
patient can contribute more than one series, so a naive bag bootstrap would
understate the interval. The uid->patient map is ``runs/_audit_meta.json``.

The centre-distance prior goes through the identical decomposition as a
reference row -- it has no model and no fitting, so whatever it scores on the
slice axis is what geometry alone buys.

    python scripts/axis_gate.py --out runs/nulls/axis_gate.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from null_battery import N_PATCH, load, pick_lesion_slot
from null_decompose import centre_prior_scores

from slotmil import prereg


def per_bag_axes(attns, masks, slot):
    """Per-bag flat / slice / within-slice AUC, plus the bag index each came from.

    Bags are dropped from an axis when that axis is degenerate for them (a bag
    whose every slice holds a nodule has no slice-axis contrast). Keeping the
    indices lets the bootstrap resample patients rather than rows.
    """
    rows = []
    for i, (a, m) in enumerate(zip(attns, masks)):
        t = (m > 0).astype(np.int8)
        s = int(t.sum())
        if s == 0 or s == len(t):
            continue
        n = len(m)
        ns = n // N_PATCH
        if ns * N_PATCH != n:
            continue

        flat = roc_auc_score(t, a[slot])

        sc = a[slot][: ns * N_PATCH].reshape(ns, N_PATCH)
        st = t[: ns * N_PATCH].reshape(ns, N_PATCH)
        has = (st.sum(axis=1) > 0).astype(np.int8)

        sl = np.nan
        if 0 < has.sum() < ns:
            sl = roc_auc_score(has, sc.mean(axis=1))

        wi = np.nan
        sel = has.astype(bool)
        sub_t, sub_s = st[sel].ravel(), sc[sel].ravel()
        if 0 < sub_t.sum() < len(sub_t):
            wi = roc_auc_score(sub_t, sub_s)

        rows.append((i, flat, sl, wi, ns, int(has.sum())))
    return rows


def cluster_bootstrap(values, clusters, reps=10000, seed=0):
    """Mean and percentile CI, resampling *clusters* (patients) with replacement."""
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    ok = np.isfinite(values)
    values, clusters = values[ok], clusters[ok]
    if values.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0, "n_clusters": 0}

    uniq = np.unique(clusters)
    by = {c: np.flatnonzero(clusters == c) for c in uniq}
    rng = np.random.default_rng(seed)
    boot = np.empty(reps)
    for r in range(reps):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([by[c] for c in pick])
        boot[r] = values[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(values.size),
        "n_clusters": int(uniq.size),
    }


def analyse(tag, val_npz, test_npz, uid_to_pat, reps, seed):
    val_a, val_m = load(val_npz)
    test_a, test_m = load(test_npz)
    slot, _ = pick_lesion_slot(val_a, val_m)

    uids = np.load(test_npz, allow_pickle=True)["uids"]
    rows = per_bag_axes(test_a, test_m, slot)
    idx = [r[0] for r in rows]
    pats = [uid_to_pat.get(str(uids[i]), str(uids[i])) for i in idx]

    out = {"tag": tag, "frozen_slot": int(slot), "n_bags_scored": len(rows)}
    for name, col in (("flat_auc", 1), ("slice_auc", 2), ("within_slice_auc", 3)):
        out[name] = cluster_bootstrap([r[col] for r in rows], pats, reps, seed)
    out["mean_slices_per_bag"] = float(np.mean([r[4] for r in rows])) if rows else None
    out["mean_lesion_slices_per_bag"] = float(np.mean([r[5] for r in rows])) if rows else None
    return out


def analyse_prior(test_npz, uid_to_pat, reps, seed):
    """Centre-distance prior through the identical decomposition. Slot 0 of a
    K=1 'attention' that never saw an image."""
    _, test_m = load(test_npz)
    priors = centre_prior_scores(test_m)
    uids = np.load(test_npz, allow_pickle=True)["uids"]
    rows = per_bag_axes(priors, test_m, 0)
    idx = [r[0] for r in rows]
    pats = [uid_to_pat.get(str(uids[i]), str(uids[i])) for i in idx]

    out = {"tag": "centre_prior", "frozen_slot": None, "n_bags_scored": len(rows)}
    for name, col in (("flat_auc", 1), ("slice_auc", 2), ("within_slice_auc", 3)):
        out[name] = cluster_bootstrap([r[col] for r in rows], pats, reps, seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/nulls")
    ap.add_argument("--meta", default="runs/_audit_meta.json")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="default: every tag with both a _val.npz and a _test.npz")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/nulls/axis_gate.json")
    ap.add_argument("--role", choices=["exploratory", "confirmatory"],
                    default="exploratory",
                    help="the attention dumps in runs/nulls were produced on the "
                         "discovery split, so anything scored from them is "
                         "exploratory; pass confirmatory only for dumps from the "
                         "seed-2027 split")
    args = ap.parse_args()

    d = Path(args.dir)
    tags = args.tags or sorted(
        p.name[: -len("_test.npz")] for p in d.glob("*_test.npz")
        if (d / f"{p.name[:-len('_test.npz')]}_val.npz").exists()
    )
    uid_to_pat = json.loads(Path(args.meta).read_text())["pat"]

    results = []
    for tag in tags:
        print(f"[{tag}]", flush=True)
        r = analyse(tag, d / f"{tag}_val.npz", d / f"{tag}_test.npz",
                    uid_to_pat, args.reps, args.seed)
        results.append(r)
        for k in ("flat_auc", "slice_auc", "within_slice_auc"):
            s = r[k]
            print(f"   {k:18s} {s['mean']:.4f}  [{s['lo']:.4f}, {s['hi']:.4f}]"
                  f"  n={s['n']} patients={s['n_clusters']}", flush=True)

    if tags:
        print("[centre_prior]", flush=True)
        r = analyse_prior(d / f"{tags[0]}_test.npz", uid_to_pat, args.reps, args.seed)
        results.append(r)
        for k in ("flat_auc", "slice_auc", "within_slice_auc"):
            s = r[k]
            print(f"   {k:18s} {s['mean']:.4f}  [{s['lo']:.4f}, {s['hi']:.4f}]", flush=True)

    payload = prereg.load().stamp(
        {"reps": args.reps, "analysis_role": args.role, "results": results})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}  ({args.role}, prereg {payload['prereg']['prereg_hash']})",
          flush=True)

    sl = [r["slice_auc"] for r in results if r["tag"] != "centre_prior"]
    spans = [s for s in sl if s["lo"] is not None]
    if spans:
        covers = sum(1 for s in spans if s["lo"] <= 0.5 <= s["hi"])
        print(f"\nGATE: {covers}/{len(spans)} arms have a slice-AUC 95% CI covering 0.5")


if __name__ == "__main__":
    main()
