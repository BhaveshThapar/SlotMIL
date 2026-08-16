#!/usr/bin/env python
"""H7's numerator: the supervised patch probe as a prior-normalised skill.

The power gate reads "the supervised patch probe's prior-normalised skill exceeds
0.50 while every content-free baseline's is below 0.05", and until now nothing
computed the first half. ``scripts/probe_ceiling.py`` printed a bare patch AUC of
0.9102 over a subsampled test set; that number has no denominator attached, and
carrying it through the eight stored ``attention:inplane`` denominators in
``runs/nulls/template_family.json`` spans skill 0.4720-0.7224 -- straddling the
threshold. This driver computes the skill the pre-registration actually declares.

Protocol, from ``configs/prereg/isbi2027.yaml`` ``H7.probe_protocol``:

    fit_split       train           subsampling here is a fitting choice
    score_split     test            every patch of every bag, no subsample
    denominator     the probe's OWN in-plane template, fit on val

The denominator is the existing ``fitted_template`` rule -- a 256-number in-plane
map fit on validation that never reads a test image -- applied to a scorer nobody
had explicitly handed it to. Every arm already carries its own; borrowing another
arm's would make the verdict depend on which arm was borrowed from.

**This driver computes the probe half only.** The content-free half of the gate
reads the confirmatory attention dumps, which do not exist until the sweeps land
and ``null_collect_attn.py --role confirmatory --dtype float32`` has run. Both
halves must hold for H7 to clear; they are reported as they arrive, and the
``content_free`` block below records that the second half is outstanding rather
than leaving its absence to be inferred from a missing key.

    python scripts/probe_gate.py --cache data/lidc/features_dinov2_vitb14.h5 \
        --splits data/lidc/splits_confirmatory.json --role confirmatory
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from slotmil import prereg
from slotmil.eval.axes import per_bag_axes
from slotmil.eval.estimands import cluster_bootstrap, prior_normalised_skill
from slotmil.eval.nulls import global_template, template_scores
from slotmil.eval.probe import fit_probe, score_bags

AXES = (("flat_auc", 1), ("slice_auc", 2), ("within_slice_auc", 3))


def _scored(tag, rows, uids, uid_to_pat, reps, seed, **extra):
    """One scorer's three axes, bootstrapped over patients rather than bags.

    Identical shape to ``template_family.py``'s rows on purpose: the two files
    are read side by side and a reader must not have to check whether ``mean``
    means the same thing in both."""
    pats = [uid_to_pat.get(uids[r[0]], uids[r[0]]) for r in rows]
    out = {"scorer": tag, "n_bags_scored": len(rows), **extra}
    out.update({name: cluster_bootstrap([r[col] for r in rows], pats, reps, seed)
                for name, col in AXES})
    return out


def _ticker(label: str, total: int, every: int = 25):
    """Print every ``every`` series. The training split is ~41 GB of compressed
    reads; a driver silent for that long is indistinguishable from a hung one."""
    def tick(n, extra):
        if n % every == 0 or n == total:
            print(f"[probe] {label} {n}/{total}  ({extra:,})", flush=True)
    return tick


def analyse(cache_path, splits, uid_to_pat, neg_per_pos, seed, reps, max_fit_scans):
    with h5py.File(cache_path, "r") as f:
        n_fit = min(len(splits["train"]), max_fit_scans or len(splits["train"]))
        clf = fit_probe(f, splits["train"], neg_per_pos=neg_per_pos, seed=seed,
                        max_scans=max_fit_scans,
                        progress=_ticker("fit series", n_fit))
        print("[probe] fitted; scoring val", flush=True)
        val_a, val_m, _ = score_bags(
            clf, f, splits["val"], progress=_ticker("val bags", len(splits["val"])))
        print("[probe] scoring test", flush=True)
        test_a, test_m, test_uids = score_bags(
            clf, f, splits["test"], progress=_ticker("test bags", len(splits["test"])))
        print(f"[probe] bootstrapping ({reps} reps)", flush=True)

    # Fit on validation, freeze, apply identically to every test bag. It cannot
    # read the test image, which is exactly what makes it the honest denominator.
    template = global_template(val_a, val_m, slot=0)
    denom_a = template_scores(test_m, template)

    probe_rows = per_bag_axes(test_a, test_m, 0)
    denom_rows = per_bag_axes(denom_a, test_m, 0)

    scorers = [
        _scored("probe", probe_rows, test_uids, uid_to_pat, reps, seed,
                source="supervised_patch_probe"),
        _scored("probe:own_inplane_template", denom_rows, test_uids, uid_to_pat,
                reps, seed, source="content_free_fitted",
                pre_registered_denominator=True),
    ]
    by = {s["scorer"]: s for s in scorers}
    numer = by["probe"]["flat_auc"]["mean"]
    denom = by["probe:own_inplane_template"]["flat_auc"]["mean"]
    skill = prior_normalised_skill(numer, denom)

    return {
        "n_fit_series": len(splits["train"]),
        "n_val_bags": len(val_m),
        "neg_per_pos": neg_per_pos,
        "score_coverage": "every_patch_of_every_bag",
        "n_patches_scored": int(sum(len(m) for m in test_m)),
        "scorers": scorers,
        "prior_normalised_skill": {
            "denominator": "probe:own_inplane_template (H7.probe_denominator)",
            "auc_probe": numer,
            "auc_template": denom,
            "skill": skill,
        },
        "h7_probe_half": {
            "threshold": 0.50,
            "skill": skill,
            "clears": bool(skill > 0.50),
        },
        "content_free": {
            "set": ["chance", "roll_permutation", "entropy_matched_random",
                    "fitted_template"],
            "threshold": 0.05,
            "status": "not computed here -- needs the confirmatory attention "
                      "dumps; H7 clears only if this half holds too",
        },
    }


def _fmt(s):
    if s["mean"] is None:
        return "     --"
    return f"{s['mean']:.4f}  [{s['lo']:.4f}, {s['hi']:.4f}]"


def report(r):
    print(f"   {'scorer':30s} {'flat':24s} {'slice':24s} within", flush=True)
    for s in r["scorers"]:
        star = " *" if s.get("pre_registered_denominator") else ""
        print(f"   {s['scorer'] + star:30s} {_fmt(s['flat_auc']):24s} "
              f"{_fmt(s['slice_auc']):24s} {_fmt(s['within_slice_auc'])}", flush=True)
    print("   * = the probe's own denominator, fit on val", flush=True)
    k = r["prior_normalised_skill"]
    print(f"\n   patches scored: {r['n_patches_scored']:,} "
          f"(every patch of every bag)", flush=True)
    print(f"   skill = ({k['auc_probe']:.4f} - {k['auc_template']:.4f}) / "
          f"(1 - {k['auc_template']:.4f}) = {k['skill']:.4f}", flush=True)
    h = r["h7_probe_half"]
    print(f"   H7 probe half (> {h['threshold']}): "
          f"{'CLEARS' if h['clears'] else 'FAILS'}", flush=True)
    print(f"   H7 content-free half: {r['content_free']['status']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--meta", default="runs/_audit_meta.json")
    ap.add_argument("--neg-per-pos", type=int, default=20,
                    help="H7.probe_protocol.fit_negatives_per_positive")
    ap.add_argument("--seed", type=int, default=0,
                    help="H7.probe_protocol.fit_rng_seed")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--max-fit-scans", type=int, default=None,
                    help="probe_protocol declares fit_split: train and no scan "
                         "cap, so the default is the whole split; a cap would "
                         "have to be declared before the fit, not after")
    ap.add_argument("--out", default="runs/nulls/probe_gate.json")
    ap.add_argument("--role", choices=["exploratory", "confirmatory"],
                    default="exploratory",
                    help="confirmatory only for the seed-2027 splits")
    args = ap.parse_args()

    splits = json.loads(Path(args.splits).read_text())
    uid_to_pat = json.loads(Path(args.meta).read_text())["pat"]

    r = analyse(args.cache, splits, uid_to_pat, args.neg_per_pos, args.seed,
                args.reps, args.max_fit_scans)
    report(r)

    payload = prereg.load().stamp({
        "reps": args.reps, "analysis_role": args.role,
        "splits_file": args.splits, "splits_hash": splits.get("hash"),
        "cache": args.cache, **r,
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}  ({args.role}, "
          f"prereg {payload['prereg']['prereg_hash']})", flush=True)


if __name__ == "__main__":
    main()
