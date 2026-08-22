#!/usr/bin/env python
"""Recommendation 1, practiced on ourselves: each arm's raw vs in-lung AUC.

**This is not a pre-registered analysis and it does not change H8's verdict.**
H8's threshold attaches to the attention-fitted denominator scored by
``scripts/h8_in_lung.py``; this file scores something that script deliberately
does not: **the arm's own frozen-slot attention**, unrestricted and restricted
to the lung, in one pass. The paper's Recommendation 1 says to report raw and
in-organ instance AUC side by side and read the gap as the dataset's positional
confound -- and until this artefact existed the paper only ever demonstrated
that convention on the template proxy, never on the arms it audits. A
recommendation the paper does not apply to itself is a reviewer finding; this
is the application.

The in-lung restriction is :func:`slotmil.eval.lung.restrict_to_lung` -- this
driver is its first consumer outside the test suite -- with the pre-registered
patch rule (``lung_thresh 0.0``, *strictly* greater than). Restriction drops
score and target together; an AUC over a restricted score population and an
unrestricted target would be a different quantity wearing the same name.

Two refusals tie this artefact to the confirmatory record:

* the frozen slot must equal the ``frozen_slot`` stored for the same tag in
  ``template_family.json`` -- scoring a different slot under the same tag name
  is the failure class ``h8_in_lung.py``'s slot comment documents;
* the per-tag raw mean must equal the stored ``trained`` scorer's
  ``flat_auc.mean`` -- the raw column already exists in the confirmatory
  artefact, and recomputing it here (so the raw/in-lung pair cannot drift into
  two runs of two files) is only honest if the recomputation lands on exactly
  the number the paper already cites.

    python scripts/arm_in_lung.py \
        --dir runs/nulls_nodule_present_confirmatory \
        --out runs/nulls_nodule_present_confirmatory/arm_in_lung.json \
        --condition nodule_present --role confirmatory
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from null_battery import load, pick_lesion_slot  # noqa: E402

from slotmil import prereg  # noqa: E402
from slotmil.eval.estimands import cluster_bootstrap  # noqa: E402
from slotmil.eval.lung import (  # noqa: E402
    DEFAULT_LUNG_THRESH,
    in_lung_from_grid,
    load_lung_grid,
    restrict_to_lung,
)
from slotmil.eval.verdict import arm_tag  # noqa: E402

ARM_SET_HYPOTHESIS = "H1"  # the eight learned-attention arms the paper scores


def _bag_auc(scores, target) -> float:
    if int(target.sum()) in (0, len(target)):
        return float("nan")
    return float(roc_auc_score(target, scores))


def stored_trained(tf_doc: dict) -> dict[str, dict]:
    """``{tag: {frozen_slot, flat_auc_mean}}`` from ``template_family.json``."""
    out = {}
    for row in tf_doc.get("results", []):
        trained = next((s for s in row.get("scorers", [])
                        if s.get("scorer") == "trained"), None)
        if trained is None:
            continue
        out[row["tag"]] = {"frozen_slot": row["frozen_slot"],
                           "flat_auc_mean": trained["flat_auc"]["mean"]}
    return out


def analyse_tag(tag, val_npz, test_npz, lung_path, uid_to_pat, reps, seed,
                lung_thresh, stored):
    val_a, val_m = load(val_npz)
    test_a, test_m = load(test_npz)
    slot, _ = pick_lesion_slot(val_a, val_m)
    uids = [str(u) for u in np.load(test_npz, allow_pickle=True)["uids"]]

    ref = stored.get(tag)
    if ref is None:
        raise SystemExit(
            f"{tag}: no trained row in template_family.json. The raw column "
            "would have no stored number to agree with; refusing.")
    if int(slot) != int(ref["frozen_slot"]):
        raise SystemExit(
            f"{tag}: pick_lesion_slot froze slot {slot} but template_family."
            f"json recorded {ref['frozen_slot']}. The two artefacts would "
            "score different slots under one tag; refusing.")

    raws, in_lungs, deltas, pats, kept = [], [], [], [], []
    no_lesion_in_lung = 0
    with h5py.File(lung_path, "r") as lf:
        for i, (a, m) in enumerate(zip(test_a, test_m)):
            uid = uids[i]
            n = len(m)
            in_lung = in_lung_from_grid(load_lung_grid(lf, uid), lung_thresh,
                                        n_expected=n)
            target = (np.asarray(m) > 0).astype(np.int8)

            raw = _bag_auc(a[slot], target)
            s_l, t_l = restrict_to_lung(a[slot], m, in_lung)
            inl = _bag_auc(s_l, t_l)

            raws.append(raw)
            in_lungs.append(inl)
            deltas.append(raw - inl)
            pats.append(uid_to_pat.get(uid, uid))
            kept.append(float(in_lung.mean()))
            # Same counting rule as h8_in_lung.py: only bags the RESTRICTION
            # emptied, not bags that were degenerate to begin with.
            if target.sum() > 0 and not (target.astype(bool) & in_lung).any():
                no_lesion_in_lung += 1

    raw_ci = cluster_bootstrap(raws, pats, reps, seed)
    if abs(raw_ci["mean"] - ref["flat_auc_mean"]) > 1e-9:
        raise SystemExit(
            f"{tag}: recomputed raw mean {raw_ci['mean']} disagrees with the "
            f"stored trained flat_auc.mean {ref['flat_auc_mean']}. The "
            "raw/in-lung pair would not describe the measurement the paper "
            "cites; refusing.")

    return {
        "tag": tag,
        "frozen_slot": int(slot),
        "lung_thresh": lung_thresh,
        "patch_rule": "a patch is in-lung when it contains any lung at all",
        "n_bags": len(pats),
        "mean_in_lung_fraction": float(np.mean(kept)),
        "n_bags_with_no_lesion_in_lung": no_lesion_in_lung,
        "raw_auc": raw_ci,
        "in_lung_auc": cluster_bootstrap(in_lungs, pats, reps, seed),
        "delta_auc": cluster_bootstrap(deltas, pats, reps, seed),
    }


def by_arm(pre, per_tag: dict[str, dict], seeds) -> dict[str, dict]:
    """Mean over seeds per arm, over the paper's scored arm set."""
    out = {}
    for name in pre.arm_set(ARM_SET_HYPOTHESIS):
        spec = pre.arm(name)["spec"]
        rows = [per_tag.get(arm_tag(spec, s)) for s in seeds]
        present = [r for r in rows if r is not None]
        if not present:
            continue
        out[name] = {
            "raw": statistics.fmean(r["raw_auc"]["mean"] for r in present),
            "in_lung": statistics.fmean(
                r["in_lung_auc"]["mean"] for r in present),
            "delta": statistics.fmean(
                r["delta_auc"]["mean"] for r in present),
            "n_seeds": len(present),
            "per_seed_raw": [None if r is None else r["raw_auc"]["mean"]
                             for r in rows],
            "per_seed_in_lung": [None if r is None
                                 else r["in_lung_auc"]["mean"] for r in rows],
        }
    return out


def _fmt(s):
    if s.get("mean") is None:
        return "     --"
    return f"{s['mean']:.4f}  [{s['lo']:.4f}, {s['hi']:.4f}]"


def report(res, condition):
    print(f"\n== {condition} ==  (the arm's own attention, frozen slot)",
          flush=True)
    for arm, v in sorted(res["per_arm"].items(), key=lambda kv: -kv[1]["raw"]):
        print(f"   {arm:28s} raw {v['raw']:.4f}   in-lung {v['in_lung']:.4f}"
              f"   delta {v['delta']:+.4f}   ({v['n_seeds']} seeds)",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="a nulls directory holding dumps and template_family.json")
    ap.add_argument("--lung", default="data/lidc/lung_masks.h5")
    ap.add_argument("--meta", default="runs/_audit_meta.json")
    ap.add_argument("--out", required=True,
                    help="explicit; this script has no default output path on "
                         "purpose, so a forgotten flag cannot overwrite an artefact")
    ap.add_argument("--condition", default=None)
    ap.add_argument("--lung-thresh", type=float, default=DEFAULT_LUNG_THRESH)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--role", choices=["exploratory", "confirmatory"],
                    default="exploratory")
    args = ap.parse_args()

    d = Path(args.dir)
    uid_to_pat = json.loads(Path(args.meta).read_text())["pat"]
    pre = prereg.load()
    tf = d / "template_family.json"
    if not tf.exists():
        raise SystemExit(f"{tf} is missing; the raw column has no stored "
                         "number to agree with")
    stored = stored_trained(json.loads(tf.read_text()))

    per_tag = {}
    for name in pre.arm_set(ARM_SET_HYPOTHESIS):
        spec = pre.arm(name)["spec"]
        for s in args.seeds:
            tag = arm_tag(spec, s)
            val, test = d / f"{tag}_val.npz", d / f"{tag}_test.npz"
            if not (val.exists() and test.exists()):
                continue
            r = analyse_tag(tag, val, test, args.lung, uid_to_pat, args.reps,
                            args.seed, args.lung_thresh, stored)
            per_tag[tag] = r
            print(f"[{tag}]  raw {_fmt(r['raw_auc'])}   "
                  f"in-lung {_fmt(r['in_lung_auc'])}", flush=True)
    if not per_tag:
        raise SystemExit(f"no arm dumps found in {d}")

    arms = by_arm(pre, per_tag, args.seeds)
    res = {
        "estimand": "raw_vs_in_lung_instance_auc (the arm's own attention)",
        "declared_in_prereg": False,
        "role": "sensitivity_outside_confirmatory_family",
        "note": (
            "Not a pre-registered analysis. H8's verdict attaches to the "
            "attention-fitted denominator and is unchanged; this artefact "
            "scores the arms' own frozen-slot attention, raw and restricted "
            "to the lung, which is Recommendation 1 applied to this paper's "
            "own arms."),
        "arm_set": ARM_SET_HYPOTHESIS,
        "per_arm": arms,
        "per_tag": per_tag,
        "raw_checked_against": str(tf),
    }
    condition = args.condition or d.name
    res.update({"analysis_role": args.role, "condition": condition,
                "reps": args.reps, "lung_mask": args.lung})
    report(res, condition)

    payload = pre.stamp(res)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}  ({args.role}, prereg {payload['prereg']['prereg_hash']})",
          flush=True)


if __name__ == "__main__":
    main()
