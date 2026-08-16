#!/usr/bin/env python
"""H8: does the fitted template still score once the lung is all that is left?

H8 is two-sided and carries no falsifier -- it is reported either way, and if
restricting to lung removes the positional prior, that becomes the paper's
actionable recommendation instead of a defeat. It needs the lung store that
``scripts/lung_mask_lidc.py`` built for all 999 series and that, until
``slotmil/eval/lung.py``, nothing read.

**``--estimand`` has no default, on purpose.** The pre-registration says two
different things and only one of them can be H8's number:

    PREREGISTRATION.md / H8.statement    "in-lung fitted-template AUC exceeds 0.65"
    estimands.secondary                  in_lung_stratified_auc =
                                         "stratified_auc restricted to protocol.lung_mask"

A plain AUC and a Mantel-Haenszel AUC over (slice x radial bin) strata are not
the same quantity, and one 0.65 threshold cannot mean both. Until an amendment
picks one, this driver refuses to guess: the caller states which estimand is
being computed, the choice is recorded in the artefact, and no ``h8`` verdict is
emitted. Running it both ways and then deciding would be choosing the rule after
seeing the number, which is the error the whole amendment chain exists to
prevent.

Restriction removes roughly three quarters of every bag -- the global in-lung
fraction under ``method: fill`` is 0.243 -- so the in-lung number is computed
over far fewer patches than the unrestricted one, and the unrestricted number is
reported beside it so the reader can see what restricting cost.

    python scripts/h8_in_lung.py --tags f32_seed0 --estimand auc --role exploratory
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from null_battery import load, pick_lesion_slot
from sklearn.metrics import roc_auc_score

from slotmil import prereg
from slotmil.eval.estimands import cluster_bootstrap, stratified_auc_detail
from slotmil.eval.lung import DEFAULT_LUNG_THRESH, in_lung_from_grid, load_lung_grid
from slotmil.eval.nulls import GRID, N_PATCH, global_template, template_scores

N_RADIAL_BINS = 6   # estimands.secondary.stratified_auc.n_radial_bins


def radial_bins(grid: int = GRID, n_bins: int = N_RADIAL_BINS) -> np.ndarray:
    """In-plane radius per patch position, into ``n_bins`` equal-count bins.

    Same construction as ``scripts/diagnose_confound.py``, which is the impl the
    config's ``stratified_auc`` estimand names. Quantile edges rather than equal
    width, so a bin is not empty at the corners.
    """
    yy, xx = np.meshgrid(np.arange(grid, dtype=np.float64),
                         np.arange(grid, dtype=np.float64), indexing="ij")
    r = np.sqrt((yy - (grid - 1) / 2) ** 2 + (xx - (grid - 1) / 2) ** 2).ravel()
    edges = np.quantile(r, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, r, side="right")


def bag_strata(n: int, n_patch: int = N_PATCH, n_bins: int = N_RADIAL_BINS
               ) -> np.ndarray:
    """``slice x radial bin`` stratum id per patch of one bag.

    The estimand says *normalised* slice. Strata are formed inside a bag and the
    AUC is computed inside a stratum, so normalising the slice coordinate is a
    monotone relabel that cannot change which patches share a stratum -- the raw
    per-bag slice index gives an identical partition, and using it avoids
    inventing a binning the freeze does not specify.
    """
    n_slices = n // n_patch
    slice_id = np.repeat(np.arange(n_slices), n_patch)[:n]
    rbin = np.tile(radial_bins(n_bins=n_bins), n_slices)[:n]
    return slice_id * n_bins + rbin


def _auc(scores, target, strata, estimand: str) -> tuple[float, int]:
    """One bag's number under whichever estimand was asked for."""
    if int(target.sum()) in (0, len(target)):
        return float("nan"), 0
    if estimand == "auc":
        return float(roc_auc_score(target, scores)), int(len(target))
    d = stratified_auc_detail(scores, target, strata)
    return d["auc"], d["n_pairs"]


def analyse(tag, val_npz, test_npz, lung_path, uid_to_pat, estimand, reps, seed,
            lung_thresh):
    val_a, val_m = load(val_npz)
    test_a, test_m = load(test_npz)
    slot, _ = pick_lesion_slot(val_a, val_m)
    uids = [str(u) for u in np.load(test_npz, allow_pickle=True)["uids"]]

    # The pre-registered content-free reference: fit on val attention, frozen,
    # applied identically to every test bag.
    template = global_template(val_a, val_m, slot=slot)
    scored = template_scores(test_m, template)

    rows_in, rows_all, pats, kept = [], [], [], []
    dropped_no_lesion_in_lung = 0
    with h5py.File(lung_path, "r") as lf:
        for i, (s, m) in enumerate(zip(scored, test_m)):
            uid = uids[i]
            n = len(m)
            in_lung = in_lung_from_grid(load_lung_grid(lf, uid), lung_thresh,
                                        n_expected=n)
            target = (np.asarray(m) > 0).astype(np.int8)
            strata = bag_strata(n)

            a_all, _ = _auc(s[slot], target, strata, estimand)
            a_in, n_used = _auc(s[slot][in_lung], target[in_lung],
                                strata[in_lung], estimand)
            if not np.isfinite(a_in):
                dropped_no_lesion_in_lung += 1
            rows_all.append(a_all)
            rows_in.append(a_in)
            pats.append(uid_to_pat.get(uid, uid))
            kept.append(float(in_lung.mean()))

    boot_in = cluster_bootstrap(rows_in, pats, reps, seed)
    boot_all = cluster_bootstrap(rows_all, pats, reps, seed)
    return {
        "tag": tag,
        "frozen_slot": int(slot),
        "estimand": estimand,
        "lung_thresh": lung_thresh,
        "patch_rule": "a patch is in-lung when it contains any lung at all",
        "n_bags": len(rows_in),
        "mean_in_lung_fraction": float(np.mean(kept)),
        "n_bags_with_no_lesion_in_lung": dropped_no_lesion_in_lung,
        "in_lung": boot_in,
        "unrestricted": boot_all,
        # Deliberately no verdict. See the module docstring: H8's statement and
        # its declared estimand name two different numbers, and picking one after
        # seeing both is the error the amendment chain exists to prevent.
        "h8": {
            "threshold": 0.65,
            "outcome": None,
            "blocked_on": "H8.statement says 'in-lung fitted-template AUC' while "
                          "estimands.secondary declares in_lung_stratified_auc; "
                          "one amendment must pick one before a verdict is scored",
        },
    }


def report(r):
    def fmt(s):
        return "     --" if s["mean"] is None else \
            f"{s['mean']:.4f}  [{s['lo']:.4f}, {s['hi']:.4f}]"
    print(f"[{r['tag']}]  frozen_slot={r['frozen_slot']}  "
          f"estimand={r['estimand']}  bags={r['n_bags']}", flush=True)
    print(f"   mean in-lung fraction   {r['mean_in_lung_fraction']:.4f}", flush=True)
    print(f"   {'unrestricted':22s} {fmt(r['unrestricted'])}", flush=True)
    print(f"   {'in-lung':22s} {fmt(r['in_lung'])}", flush=True)
    if r["n_bags_with_no_lesion_in_lung"]:
        print(f"   {r['n_bags_with_no_lesion_in_lung']} bags had no lesion patch "
              f"inside the lung mask and score nan on the restricted axis",
              flush=True)
    print(f"   H8 (> {r['h8']['threshold']}): NOT SCORED -- "
          f"{r['h8']['blocked_on']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/nulls")
    ap.add_argument("--lung", default="data/lidc/lung_masks.h5")
    ap.add_argument("--meta", default="runs/_audit_meta.json")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="default: every tag with both a _val.npz and a _test.npz")
    ap.add_argument("--estimand", required=True, choices=["auc", "stratified_auc"],
                    help="no default: H8's statement and its declared estimand "
                         "name two different numbers and the freeze has not yet "
                         "picked one")
    ap.add_argument("--lung-thresh", type=float, default=DEFAULT_LUNG_THRESH)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/nulls/h8_in_lung.json")
    ap.add_argument("--role", choices=["exploratory", "confirmatory"],
                    default="exploratory")
    args = ap.parse_args()

    d = Path(args.dir)
    tags = args.tags or sorted(
        p.name[: -len("_test.npz")] for p in d.glob("*_test.npz")
        if (d / f"{p.name[:-len('_test.npz')]}_val.npz").exists()
    )
    uid_to_pat = json.loads(Path(args.meta).read_text())["pat"]

    results = []
    for tag in tags:
        r = analyse(tag, d / f"{tag}_val.npz", d / f"{tag}_test.npz", args.lung,
                    uid_to_pat, args.estimand, args.reps, args.seed,
                    args.lung_thresh)
        results.append(r)
        report(r)

    payload = prereg.load().stamp({
        "reps": args.reps, "analysis_role": args.role,
        "estimand": args.estimand, "lung_mask": args.lung, "results": results,
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}  ({args.role}, "
          f"prereg {payload['prereg']['prereg_hash']})", flush=True)


if __name__ == "__main__":
    main()
