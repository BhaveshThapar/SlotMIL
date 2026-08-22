#!/usr/bin/env python
"""H2 sensitivity: a paired, patient-clustered interval on the 0.0109 margin.

**This is not a pre-registered analysis and it does not change H2's verdict.**
The confirmatory H2 result is whatever ``scripts/prereg_verdict.py`` scores:
a comparison of means, per arm, against the centre prior's slice AUC. That
comparison passes on a margin of 0.0109 (0.6427 against Normal Guidance's
0.6318 on the condition that carries the family), stated on point estimates.
The marginal intervals in ``axis_gate.json`` overlap almost entirely -- but
marginal overlap is not a paired test, and a headline ordering that thin
deserves the paired interval stated next to it rather than left to a reviewer
to demand.

So this computes, per H2 tag, the per-bag paired difference

    delta = centre_prior_slice_auc(bag) - arm_slice_auc(bag)

on the slice axis, with the same patient-level cluster bootstrap every other
interval in the repository uses. Positive delta means the prior sits above the
arm. Two granularities are reported, because the repository already has a
precedent for each and they answer different sentences:

* per seed -- the H10 idiom: an interval per tag and a count of outcomes over
  the arm's seeds (``prior_wins`` / ``arm_wins`` / ``indistinguishable``);
* pooled per arm -- the arm's five seeds' per-bag deltas concatenated and
  bootstrapped with the patient as the cluster, so a patient appearing in all
  five seed-rows is resampled as one unit. This is the single inline number
  the paper can quote.

The per-seed interval is computed twice on purpose: once through
``template_family._paired`` (the tested alignment) and once from the local
delta rows that feed the pooled estimate. The two must agree exactly, or the
pooled number would be built on an alignment the tested path never saw --
refusal, not warning. The arm's and prior's marginal slice-AUC means are also
recomputed here and checked against the stored ``axis_gate.json`` rows, so
this artefact provably describes the same measurement the paper already cites.

    python scripts/h2_paired_slice.py \
        --dir runs/nulls_nodule_present_confirmatory \
        --out runs/nulls_nodule_present_confirmatory/h2_paired_slice.json \
        --condition nodule_present --role confirmatory
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from null_battery import load, pick_lesion_slot  # noqa: E402
from null_decompose import centre_prior_scores  # noqa: E402
from template_family import _paired  # noqa: E402

from slotmil import prereg  # noqa: E402
from slotmil.eval.axes import per_bag_axes  # noqa: E402
from slotmil.eval.estimands import cluster_bootstrap, h10_outcome  # noqa: E402
from slotmil.eval.verdict import arm_tag  # noqa: E402

SLICE_COL = 2  # per_bag_axes column: (index, flat, slice, within_slice, ...)

# h10_outcome is written for ``separable - trained``; here the first operand is
# the prior, so its labels translate rather than being reimplemented.
OUTCOME = {"oracle_wins": "prior_wins", "trained_wins": "arm_wins",
           "indistinguishable": "indistinguishable", None: None}


def delta_rows(prior_rows, trained_rows, uids, uid_to_pat):
    """Per-bag ``prior - arm`` slice-AUC deltas with their patient ids.

    The alignment mirrors ``template_family._paired`` -- by originating bag
    index, not by position -- and ``analyse_tag`` cross-checks the two against
    each other, so this copy cannot drift without refusing.
    """
    a = {r[0]: r[SLICE_COL] for r in prior_rows}
    b = {r[0]: r[SLICE_COL] for r in trained_rows}
    idx = sorted(a.keys() & b.keys())
    deltas = [a[i] - b[i] for i in idx]
    pats = [uid_to_pat.get(str(uids[i]), str(uids[i])) for i in idx]
    return deltas, pats


def _check(name, ours, stored, tol=1e-9):
    if stored is None or abs(ours - stored) <= tol:
        return
    raise SystemExit(
        f"{name}: recomputed mean {ours} disagrees with the stored "
        f"axis_gate.json value {stored}. The paired interval would describe a "
        "different measurement than the one the paper cites; refusing.")


def analyse_tag(tag, val_npz, test_npz, uid_to_pat, reps, seed, stored):
    """One tag's paired interval, plus the delta rows for pooling."""
    val_a, val_m = load(val_npz)
    test_a, test_m = load(test_npz)
    slot, _ = pick_lesion_slot(val_a, val_m)
    uids = np.load(test_npz, allow_pickle=True)["uids"]

    trained_rows = per_bag_axes(test_a, test_m, slot)
    prior_rows = per_bag_axes(centre_prior_scores(test_m), test_m, 0)

    ci = _paired(prior_rows, trained_rows, uids, uid_to_pat, reps, seed,
                 col=SLICE_COL)
    deltas, pats = delta_rows(prior_rows, trained_rows, uids, uid_to_pat)
    local = cluster_bootstrap(deltas, pats, reps, seed)
    if not all(local[k] == ci[k] for k in ("mean", "lo", "hi", "n")):
        raise SystemExit(
            f"{tag}: the local delta rows do not reproduce template_family."
            f"_paired ({local} vs {ci}). The pooled interval would be built on "
            "an untested alignment; refusing.")

    # Marginal means, checked against the stored artefact the paper cites.
    # Each scorer's rows carry their own indices: the drop rules in
    # per_bag_axes depend only on the mask, so the two sets coincide today,
    # but pairing the prior's values with the arm's patients would rely on
    # that silently.
    def marginal(rows):
        pats = [uid_to_pat.get(str(uids[r[0]]), str(uids[r[0]])) for r in rows]
        return cluster_bootstrap([r[SLICE_COL] for r in rows], pats, 0,
                                 seed)["mean"]

    arm_marg = marginal(trained_rows)
    prior_marg = marginal(prior_rows)
    if stored is not None:
        _check(f"{tag} slice_auc", arm_marg, stored.get(tag))
        _check(f"{tag} centre_prior slice_auc", prior_marg,
               stored.get("centre_prior"))

    return {
        "tag": tag,
        "frozen_slot": int(slot),
        "delta": ci,
        "outcome": OUTCOME[h10_outcome(ci)],
        "arm_slice_auc_mean": arm_marg,
        "centre_prior_slice_auc_mean": prior_marg,
    }, deltas, pats


def stored_slice_means(axis_gate_path: Path) -> dict[str, float] | None:
    """``{tag: slice_auc mean}`` from the stored artefact, if it exists."""
    if not axis_gate_path.exists():
        return None
    doc = json.loads(axis_gate_path.read_text())
    return {r["tag"]: r["slice_auc"]["mean"] for r in doc.get("results", [])
            if r.get("slice_auc")}


def analyse(pre, d: Path, uid_to_pat, seeds, reps, seed, stored):
    per_arm = {}
    for name in pre.arm_set("H2"):
        spec = pre.arm(name)["spec"]
        rows, all_deltas, all_pats = [], [], []
        for s in seeds:
            tag = arm_tag(spec, s)
            val, test = d / f"{tag}_val.npz", d / f"{tag}_test.npz"
            if not (val.exists() and test.exists()):
                continue
            row, deltas, pats = analyse_tag(tag, val, test, uid_to_pat, reps,
                                            seed, stored)
            rows.append(row)
            all_deltas.extend(deltas)
            all_pats.extend(pats)
        if not rows:
            continue
        pooled = cluster_bootstrap(all_deltas, all_pats, reps, seed)
        outcomes = [r["outcome"] for r in rows]
        per_arm[name] = {
            "per_seed": rows,
            "n_seeds": len(rows),
            "outcome_counts": {o: outcomes.count(o) for o in
                               ("prior_wins", "arm_wins", "indistinguishable")},
            "arm_slice_auc_mean_over_seeds": statistics.fmean(
                r["arm_slice_auc_mean"] for r in rows),
            "pooled": {**pooled, "outcome": OUTCOME[h10_outcome(pooled)],
                       "clusters": "patient (a patient's rows from every seed "
                                   "resample as one unit)"},
        }
    if not per_arm:
        raise SystemExit(f"no H2 arm dumps found in {d}")

    best = max(per_arm, key=lambda k: per_arm[k]["arm_slice_auc_mean_over_seeds"])
    return {
        "estimand": "paired_slice_auc_delta (centre_prior - arm)",
        "declared_in_prereg": False,
        "role": "sensitivity_outside_confirmatory_family",
        "note": (
            "Not a pre-registered analysis. The confirmatory H2 verdict is "
            "unchanged and is scored by scripts/prereg_verdict.py as a "
            "comparison of means. This file exists because that comparison "
            "passes on a 0.0109 point-estimate margin, and the paired "
            "patient-clustered interval is the honest thing to state next to "
            "an ordering that thin. Positive delta = prior above the arm."),
        "sign_convention": "delta = centre_prior - arm; prior_wins when lo > 0",
        "per_arm": per_arm,
        "best_arm_by_slice_auc": best,
        "best_arm_pooled_delta": per_arm[best]["pooled"],
    }


def _fmt(s):
    if s.get("mean") is None:
        return "     --"
    return f"{s['mean']:+.4f}  [{s['lo']:+.4f}, {s['hi']:+.4f}]"


def report(res, condition):
    print(f"\n== {condition} ==  (delta = centre_prior - arm, slice axis)",
          flush=True)
    for arm, v in sorted(res["per_arm"].items(),
                         key=lambda kv: -kv[1]["arm_slice_auc_mean_over_seeds"]):
        c = v["outcome_counts"]
        print(f"   {arm:28s} slice_auc {v['arm_slice_auc_mean_over_seeds']:.4f}"
              f"  pooled {_fmt(v['pooled'])} -> {v['pooled']['outcome']}"
              f"  (seeds: {c['prior_wins']}P/{c['arm_wins']}A/"
              f"{c['indistinguishable']}I)", flush=True)
    b = res["best_arm_by_slice_auc"]
    print(f"   best arm {b}: pooled {_fmt(res['best_arm_pooled_delta'])} -> "
          f"{res['best_arm_pooled_delta']['outcome']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="a nulls directory holding the *_val/_test.npz dumps")
    ap.add_argument("--meta", default="runs/_audit_meta.json")
    ap.add_argument("--out", required=True,
                    help="explicit; this script has no default output path on "
                         "purpose, so a forgotten flag cannot overwrite an artefact")
    ap.add_argument("--condition", default=None)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--role", choices=["exploratory", "confirmatory"],
                    default="exploratory")
    args = ap.parse_args()

    d = Path(args.dir)
    uid_to_pat = json.loads(Path(args.meta).read_text())["pat"]
    pre = prereg.load()
    stored = stored_slice_means(d / "axis_gate.json")
    if stored is None:
        print(f"[h2] note: {d}/axis_gate.json absent, marginal cross-check "
              "skipped", flush=True)

    res = analyse(pre, d, uid_to_pat, args.seeds, args.reps, args.seed, stored)
    condition = args.condition or d.name
    res.update({"analysis_role": args.role, "condition": condition,
                "reps": args.reps,
                "marginals_checked_against": None if stored is None
                else str(d / "axis_gate.json")})
    report(res, condition)

    payload = pre.stamp(res)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}  ({args.role}, prereg {payload['prereg']['prereg_hash']})",
          flush=True)


if __name__ == "__main__":
    main()
