#!/usr/bin/env python
"""H8: does the fitted template still score once the lung is all that is left?

H8 is two-sided and carries no falsifier -- it is reported either way, and if
restricting to lung removes the positional prior, that becomes the paper's
actionable recommendation instead of a defeat. It needs the lung store that
``scripts/lung_mask_lidc.py`` built for all 999 series and that, until
``slotmil/eval/lung.py``, nothing read.

**Two numbers, one threshold.** H8's statement named a plain AUC and the
estimand it declared ``depends_on`` named a stratified one, which is not the
same quantity; the amendment of 2026-08-15 (``7682347538e76fc8``) ruled that the
0.65 verdict attaches to ``in_lung_auc`` and that ``in_lung_stratified_auc`` is
reported beside it with no threshold. Both are computed here in one pass, so the
reported pair cannot drift, and the verdict reads only the one the config marks
``carries_verdict_for: H8``.

The stratified number is not a second opinion on the same question. A fitted
template is a *purely* positional scorer, and stratifying by (slice x radial
bin) removes exactly the ranking such a scorer lives on -- it collapses toward
0.5 by construction. It is reported because that collapse is informative about
the estimand, not because 0.65 could ever have meant it.

Restriction removes roughly three quarters of every bag -- the global in-lung
fraction under ``method: fill`` is 0.243 -- so the in-lung number is computed
over far fewer patches than the unrestricted one, and the unrestricted number is
reported beside it so the reader can see what restricting cost.

    python scripts/h8_in_lung.py --tags f32_seed0 --role exploratory
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


def analyse(tag, val_npz, test_npz, lung_path, uid_to_pat, reps, seed,
            lung_thresh):
    val_a, val_m = load(val_npz)
    test_a, test_m = load(test_npz)
    slot, _ = pick_lesion_slot(val_a, val_m)
    uids = [str(u) for u in np.load(test_npz, allow_pickle=True)["uids"]]

    # The pre-registered content-free reference: fit on val attention, frozen,
    # applied identically to every test bag.
    template = global_template(val_a, val_m, slot=slot)
    scored = template_scores(test_m, template)

    # Both estimands in one pass over the bags, so the reported pair cannot drift
    # apart into two runs of two different files.
    rows: dict[tuple[str, str], list[float]] = {
        (e, r): [] for e in ("auc", "stratified_auc") for r in ("in_lung", "all")
    }
    pats, kept = [], []
    no_lesion_in_lung = 0
    with h5py.File(lung_path, "r") as lf:
        for i, (s, m) in enumerate(zip(scored, test_m)):
            uid = uids[i]
            n = len(m)
            in_lung = in_lung_from_grid(load_lung_grid(lf, uid), lung_thresh,
                                        n_expected=n)
            target = (np.asarray(m) > 0).astype(np.int8)
            strata = bag_strata(n)

            # template_scores emits K=1, like centre_prior -- `slot` indexes the
            # ARM's attention and is used only to fit the template above. Reading
            # s[slot] here raised on the first dump whose frozen slot was not 0,
            # which is the loud version of the failure; the quiet version would
            # have been a dump whose slot happened to be 0 scoring correctly and
            # every other dump scoring a different scorer under the same name.
            for estimand in ("auc", "stratified_auc"):
                a_all, _ = _auc(s[0], target, strata, estimand)
                a_in, _ = _auc(s[0][in_lung], target[in_lung],
                               strata[in_lung], estimand)
                rows[(estimand, "all")].append(a_all)
                rows[(estimand, "in_lung")].append(a_in)
            if not np.isfinite(rows[("auc", "in_lung")][-1]):
                no_lesion_in_lung += 1
            pats.append(uid_to_pat.get(uid, uid))
            kept.append(float(in_lung.mean()))

    estimands = {
        e: {"in_lung": cluster_bootstrap(rows[(e, "in_lung")], pats, reps, seed),
            "unrestricted": cluster_bootstrap(rows[(e, "all")], pats, reps, seed)}
        for e in ("auc", "stratified_auc")
    }

    # The verdict reads in_lung_auc only -- estimands.secondary marks it
    # carries_verdict_for: H8. Two-sided: reported either way, no falsifier.
    verdict = estimands["auc"]["in_lung"]["mean"]
    return {
        "tag": tag,
        "frozen_slot": int(slot),
        "lung_thresh": lung_thresh,
        "patch_rule": "a patch is in-lung when it contains any lung at all",
        "n_bags": len(pats),
        "mean_in_lung_fraction": float(np.mean(kept)),
        "n_bags_with_no_lesion_in_lung": no_lesion_in_lung,
        "estimands": estimands,
        "h8": {
            "estimand": "in_lung_auc",
            "threshold": 0.65,
            "value": verdict,
            "exceeds": None if verdict is None else bool(verdict > 0.65),
            "falsifier": "none -- two-sided, reported either way",
            "reported_beside_the_threshold": "in_lung_stratified_auc",
        },
    }


def report(r):
    def fmt(s):
        return "     --" if s["mean"] is None else \
            f"{s['mean']:.4f}  [{s['lo']:.4f}, {s['hi']:.4f}]"
    print(f"[{r['tag']}]  frozen_slot={r['frozen_slot']}  bags={r['n_bags']}",
          flush=True)
    print(f"   mean in-lung fraction   {r['mean_in_lung_fraction']:.4f}", flush=True)
    print(f"   {'estimand':22s} {'unrestricted':24s} in-lung", flush=True)
    for name, e in r["estimands"].items():
        star = " *" if name == "auc" else ""
        print(f"   {name + star:22s} {fmt(e['unrestricted']):24s} "
              f"{fmt(e['in_lung'])}", flush=True)
    print("   * = in_lung_auc, the estimand H8's threshold attaches to", flush=True)
    if r["n_bags_with_no_lesion_in_lung"]:
        print(f"   {r['n_bags_with_no_lesion_in_lung']} bags had no lesion patch "
              f"inside the lung mask and score nan on the restricted axis",
              flush=True)
    h = r["h8"]
    print(f"   H8: in_lung_auc {h['value']:.4f} vs {h['threshold']} -> "
          f"{'EXCEEDS' if h['exceeds'] else 'BELOW'}  (two-sided, no falsifier)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/nulls")
    ap.add_argument("--lung", default="data/lidc/lung_masks.h5")
    ap.add_argument("--meta", default="runs/_audit_meta.json")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="default: every tag with both a _val.npz and a _test.npz")
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
                    uid_to_pat, args.reps, args.seed, args.lung_thresh)
        results.append(r)
        report(r)

    payload = prereg.load().stamp({
        "reps": args.reps, "analysis_role": args.role,
        "verdict_estimand": "in_lung_auc", "lung_mask": args.lung,
        "results": results,
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}  ({args.role}, "
          f"prereg {payload['prereg']['prereg_hash']})", flush=True)


if __name__ == "__main__":
    main()
