#!/usr/bin/env python
"""Driver for the factorised content-free family: fit on val, score on test.

``slotmil/eval/templates.py`` answers "which axis does a content-free baseline
need in order to score well on volumetric CT?" and had no driver, so the numbers
it produces existed only in a terminal. Anything that goes in the paper has to be
regenerable from the repository by someone who does not trust us, which means a
committed script and a stamped artefact.

What this writes, per attention dump, is nine scorers through one identical
decomposition (:func:`slotmil.eval.axes.per_bag_axes`, the same function
``axis_gate.py`` uses, so the rows are comparable across both files):

    trained            the arm's frozen lesion slot -- the number being audited
    mask-fitted        inplane / axial / separable / joint, fit on validation
                       LESION FREQUENCY: the oracle, the best any content-free
                       method could do here
    attention-fitted   the same four, fit on the arm's own average attention --
                       the pre-registered convention, whose `inplane` member IS
                       `reference_baselines.fitted_template`
    centre_prior       model-free geometry, no fitting at all

**Both fit sources are reported because reporting either alone is misleading.**
The attention-fitted axial template inherits the arm's axial blindness and scores
at chance; printing that as "the axial axis carries nothing" would be wrong,
since the centre prior reaches slice AUC ~0.60 on the same bags. The mask-fitted
axial template is what shows the axis genuinely carries signal that the arm did
not find. Conversely the attention-fitted `inplane` is the one the
pre-registration declares, so it is the one the skill denominator comes from --
and this script prints ``pre_registered_denominator`` next to it so that stays
visible rather than being a fact about a config file.

**The denominator is pinned, not selected.** ``estimands.prior_normalised_skill``
is computed against the declared reference only. The family maximum on test is
deliberately *not* used: the null battery already measured that best-of-k
selection on test buys +0.11 AUC from pure noise, and choosing the strongest of
four references after seeing their test scores is that error wearing a
decomposition. The other members are reported as references, never as the
denominator.

The paired ``separable - trained`` row is the one to read carefully. It is a
per-bag paired difference with a patient-level cluster bootstrap, and on the
discovery dumps its interval crosses zero -- so the defensible claim is that the
trained network is *statistically indistinguishable* from a reference that never
reads a test image, not that the reference beats it. The weaker sentence is the
one that survives a reviewer.

Discovery dumps in ``runs/nulls`` are exploratory by construction, which is the
default here. ``--role confirmatory`` is for dumps from the seed-2027 split and
nothing else.

    python scripts/template_family.py --tags f32_seed0 --reps 10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from null_battery import load, pick_lesion_slot
from null_decompose import centre_prior_scores

from slotmil import prereg
from slotmil.eval.axes import per_bag_axes
from slotmil.eval.estimands import (
    cluster_bootstrap,
    h10_outcome,
    prior_normalised_skill,
)
from slotmil.eval.templates import DEFAULT_DEPTH_BINS, fit_family, score_family

# The three columns of per_bag_axes, by the name they are reported under.
AXES = (("flat_auc", 1), ("slice_auc", 2), ("within_slice_auc", 3))

# Family members in the order the module's 2x2 table lists them, so the printed
# output and the docstring cannot drift apart.
MEMBERS = ("inplane", "axial", "separable", "joint")


def _patients(rows, uids, uid_to_pat):
    """Patient id per scored row. Bags, not rows, are the resampling unit's input
    -- the map is uid -> patient and a LIDC patient can contribute two series."""
    return [uid_to_pat.get(str(uids[r[0]]), str(uids[r[0]])) for r in rows]


def _boot(rows, pats, reps, seed):
    return {name: cluster_bootstrap([r[col] for r in rows], pats, reps, seed)
            for name, col in AXES}


def _scored(tag, rows, uids, uid_to_pat, reps, seed, **extra):
    pats = _patients(rows, uids, uid_to_pat)
    out = {"scorer": tag, "n_bags_scored": len(rows), **extra}
    out.update(_boot(rows, pats, reps, seed))
    return out


def _paired(rows_a, rows_b, uids, uid_to_pat, reps, seed, col=1):
    """Per-bag ``a - b`` on one axis, bootstrapped over patients.

    Aligned on the originating bag index rather than on position. The drop rules
    in ``per_bag_axes`` depend only on the mask, so every scorer here yields the
    same index set -- but relying on that silently would make a future scorer
    with its own drop rule produce a paired difference between different bags.
    """
    a = {r[0]: r[col] for r in rows_a}
    b = {r[0]: r[col] for r in rows_b}
    idx = sorted(a.keys() & b.keys())
    deltas = [a[i] - b[i] for i in idx]
    pats = [uid_to_pat.get(str(uids[i]), str(uids[i])) for i in idx]
    return cluster_bootstrap(deltas, pats, reps, seed)


def analyse(tag, val_npz, test_npz, uid_to_pat, reps, seed, n_bins):
    val_a, val_m = load(val_npz)
    test_a, test_m = load(test_npz)
    slot, _ = pick_lesion_slot(val_a, val_m)
    uids = np.load(test_npz, allow_pickle=True)["uids"]

    trained_rows = per_bag_axes(test_a, test_m, slot)
    scorers = [_scored("trained", trained_rows, uids, uid_to_pat, reps, seed,
                       frozen_slot=int(slot), source=None)]

    # Fit on validation only. Both sources, because each answers a question the
    # other cannot -- see the module docstring.
    rows_by = {}
    fits = {}
    for source in ("masks", "attention"):
        fam = fit_family(val_m, attns=val_a if source == "attention" else None,
                         slot=slot if source == "attention" else None,
                         source=source, n_bins=n_bins)
        fits[source] = {k: v.to_dict() for k, v in fam.items() if k != "source"}
        scores = score_family(test_m, fam)
        for member in MEMBERS:
            rows = per_bag_axes(scores[member], test_m, 0)
            rows_by[(source, member)] = rows
            scorers.append(_scored(
                f"{source}:{member}", rows, uids, uid_to_pat, reps, seed,
                source=source,
                # The one member the pre-registration names. Carried in the
                # artefact so a reader does not have to cross-reference the YAML
                # to know which of the eight is the declared reference.
                pre_registered_denominator=(source == "attention"
                                            and member == "inplane"),
            ))

    prior_rows = per_bag_axes(centre_prior_scores(test_m), test_m, 0)
    scorers.append(_scored("centre_prior", prior_rows, uids, uid_to_pat, reps,
                           seed, source="model_free"))

    # Paired against the trained arm, on the flat axis. Reported for every
    # mask-fitted member so the comparison cannot be read as cherry-picked to
    # separable, which is the member that happens to win.
    paired = {
        f"masks:{member} - trained": _paired(
            rows_by[("masks", member)], trained_rows, uids, uid_to_pat, reps, seed)
        for member in MEMBERS
    }

    by_name = {s["scorer"]: s for s in scorers}
    denom = by_name["attention:inplane"]["flat_auc"]["mean"]
    skill = {
        "denominator": "attention:inplane (reference_baselines.fitted_template)",
        "auc_template": denom,
        "trained": prior_normalised_skill(by_name["trained"]["flat_auc"]["mean"], denom),
        # Every other member measured against the SAME pinned denominator. A
        # reference scoring above the declared one shows up here as positive
        # skill, which is informative and is not licence to swap denominators.
        **{f"masks:{m}": prior_normalised_skill(
            by_name[f"masks:{m}"]["flat_auc"]["mean"], denom) for m in MEMBERS},
    }

    # H10, scored on the pinned member only. The other three paired rows stay
    # reported above; scoring whichever of them wins on test would be the
    # best-of-k error the denominator ruling already refuses.
    h10 = {"member": "separable", "axis": "flat",
           "outcome": h10_outcome(paired["masks:separable - trained"]),
           "falsifies": False}
    h10["falsifies"] = h10["outcome"] == "trained_wins"

    return {"tag": tag, "frozen_slot": int(slot), "n_depth_bins": n_bins,
            "fits": fits, "scorers": scorers, "paired_flat_auc": paired,
            "prior_normalised_skill": skill, "h10": h10}


def _fmt(s):
    if s["mean"] is None:
        return "     --"
    return f"{s['mean']:.4f}  [{s['lo']:.4f}, {s['hi']:.4f}]"


def report(r):
    print(f"[{r['tag']}]  frozen_slot={r['frozen_slot']}  bins={r['n_depth_bins']}",
          flush=True)
    print(f"   {'scorer':22s} {'flat':24s} {'slice':24s} within", flush=True)
    for s in r["scorers"]:
        star = " *" if s.get("pre_registered_denominator") else ""
        print(f"   {s['scorer'] + star:22s} {_fmt(s['flat_auc']):24s} "
              f"{_fmt(s['slice_auc']):24s} {_fmt(s['within_slice_auc'])}", flush=True)
    print("   * = the pre-registered skill denominator", flush=True)
    print("   paired, flat axis:", flush=True)
    for k, v in r["paired_flat_auc"].items():
        crosses = "" if v["lo"] is None else (
            "   (crosses zero)" if v["lo"] <= 0.0 <= v["hi"] else "")
        print(f"     {k:28s} {_fmt(v)}{crosses}", flush=True)
    print(f"   H10 ({r['h10']['member']}, {r['h10']['axis']}): "
          f"{r['h10']['outcome']}"
          f"{'  FALSIFIES' if r['h10']['falsifies'] else ''}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/nulls")
    ap.add_argument("--meta", default="runs/_audit_meta.json")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="default: every tag with both a _val.npz and a _test.npz")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-bins", type=int, default=DEFAULT_DEPTH_BINS,
                    help="depth bins. More bins makes the joint template "
                         "strictly stronger and the reported skill strictly "
                         "lower, so the conservative direction is up, not down.")
    ap.add_argument("--out", default="runs/nulls/template_family.json")
    ap.add_argument("--role", choices=["exploratory", "confirmatory"],
                    default="exploratory",
                    help="the dumps in runs/nulls are from the discovery split, "
                         "so anything scored from them is exploratory; pass "
                         "confirmatory only for seed-2027 dumps")
    args = ap.parse_args()

    d = Path(args.dir)
    tags = args.tags or sorted(
        p.name[: -len("_test.npz")] for p in d.glob("*_test.npz")
        if (d / f"{p.name[:-len('_test.npz')]}_val.npz").exists()
    )
    uid_to_pat = json.loads(Path(args.meta).read_text())["pat"]

    results = []
    for tag in tags:
        r = analyse(tag, d / f"{tag}_val.npz", d / f"{tag}_test.npz",
                    uid_to_pat, args.reps, args.seed, args.n_bins)
        results.append(r)
        report(r)

    payload = prereg.load().stamp({
        "reps": args.reps, "n_depth_bins": args.n_bins,
        "analysis_role": args.role, "results": results,
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}  ({args.role}, "
          f"prereg {payload['prereg']['prereg_hash']})", flush=True)


if __name__ == "__main__":
    main()
