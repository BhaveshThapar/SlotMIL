#!/usr/bin/env python
"""Patient-specific skill: the estimand H4 is scored on, and H6 reads the same number.

``slotmil/eval/estimands.py::patient_specific_skill`` has existed, been unit-tested
and had no caller outside ``tests/`` since it was written. So has
``slotmil/eval/nulls.py::shuffle_masks_across_bags``, the derangement H4's own
``cross_patient`` block names as its ``impl`` -- ``scripts/null_battery.py`` holds a
separate discovery-era copy, and calling that one instead would compute a number
under a name the pre-registration assigned to a different function. This driver is
what makes the declared estimand exist as an artefact.

Four consumers, one artefact, deliberately:

* **H4** -- every arm's skill < 0.15 and the median across arms < 0.10.
* **H6** -- Normal Guidance must move slice AUC without moving *this*, and the
  hypothesis declares ``cross_patient: same_as_H4`` by reference rather than by
  copy so the two cannot drift into two numbers with one name.
* **the nested reference chain** -- ``cross_patient`` is the third link of
  ``estimands.decomposition``, between the fitted template and the full model.
* **the Holm family** -- ``cross_patient`` is one of its three references.

What the cross-patient arm changes is the *target*, not the score: the model, the
protocol and the frozen-slot selection are all untouched, and only whose lesion
masks the attention is scored against is swapped. Whatever survives that swap was
never about this patient.

Two operational points that are not free choices and are not hidden:

**The paired form is what the skill is computed from.** ``pairing:
per_bag_against_the_real_score`` is declared, so each bag's cross-patient score is
its own mean over the 100 derangements and the skill is the mean of per-bag
differences, bootstrapped over patients. ``aggregate: mean_over_derangements`` is
also reported, as the mean over derangements of each derangement's mean over bags.
The two agree except where a derangement handed some bag a degenerate donor mask
and ``per_bag_axes`` dropped it; the artefact carries both and the count of bags
that lost derangements, rather than reporting one number and calling it the other.

**H4 has no declared unit over seeds.** H5 had exactly this hole and ruled it as
the mean over seeds, arguing that "no ARM's skill" is the statement's own noun and
that a per-seed maximum takes five draws at the threshold instead of one. That
argument transfers verbatim, so the per-arm value used for the verdict is the mean
over that arm's seeds -- and every per-seed value is reported regardless, for the
same reason H5's ``report_per_seed`` is not decoration.

    python scripts/h4_cross_patient.py \\
        --dir runs/nulls_nodule_present_confirmatory \\
        --template-family runs/nulls_nodule_present_confirmatory/template_family.json \\
        --out runs/nulls_nodule_present_confirmatory/h4_cross_patient.json \\
        --role confirmatory
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from null_battery import load, pick_lesion_slot

from slotmil import prereg
from slotmil.eval.axes import per_bag_axes
from slotmil.eval.estimands import (
    cluster_bootstrap,
    patient_specific_skill,
    reference_chain,
)
from slotmil.eval.nulls import shuffle_masks_across_bags
from slotmil.eval.verdict import arm_tag

# per_bag_axes returns (index, flat, slice, within_slice, n_slices, n_lesion_slices).
# H4 declares `axis: flat`, which is column 1.
FLAT = 1


def analyse(tag, val_npz, test_npz, uid_to_pat, cp, reps, boot_seed,
            template_auc=None, progress=True):
    val_a, val_m = load(val_npz)
    test_a, test_m = load(test_npz)
    slot, _ = pick_lesion_slot(val_a, val_m)
    uids = np.load(test_npz, allow_pickle=True)["uids"]

    real = {r[0]: r[FLAT] for r in per_bag_axes(test_a, test_m, slot)}

    # One RNG for the whole derangement sequence, seeded from the config. Seeding
    # per derangement instead would make derangement d depend only on d, so two
    # runs with different n_derangements would share a prefix -- which sounds like
    # a feature and is how a "100-derangement" number becomes reproducible only at
    # 100.
    rng = np.random.default_rng(cp["rng_seed"])
    n_der = int(cp["n_derangements"])

    per_bag: dict[int, list[float]] = defaultdict(list)
    per_derangement: list[float] = []
    t0 = time.time()
    for d in range(n_der):
        deranged = shuffle_masks_across_bags(test_m, rng)
        rows = per_bag_axes(test_a, deranged, slot)
        flats = []
        for r in rows:
            per_bag[r[0]].append(r[FLAT])
            flats.append(r[FLAT])
        per_derangement.append(float(np.mean(flats)) if flats else float("nan"))
        if progress and (d + 1) % 20 == 0:
            rate = (time.time() - t0) / (d + 1)
            print(f"   [{tag}] derangement {d + 1}/{n_der}  "
                  f"{rate:.2f}s each, ~{rate * (n_der - d - 1):.0f}s left", flush=True)

    idx = sorted(set(real) & set(per_bag))
    dropped = sorted(set(real) - set(per_bag))
    cross = {i: float(np.mean(per_bag[i])) for i in idx}
    pats = [uid_to_pat.get(str(uids[i]), str(uids[i])) for i in idx]

    auc_real = cluster_bootstrap([real[i] for i in idx], pats, reps, boot_seed)
    auc_cross = cluster_bootstrap([cross[i] for i in idx], pats, reps, boot_seed)
    skill = cluster_bootstrap([real[i] - cross[i] for i in idx], pats, reps, boot_seed)

    # The mean of per-bag differences equals the difference of per-bag means over one
    # index set, so these two must agree to float noise. Checked rather than assumed:
    # a disagreement means the two paths saw different bags, which is the one way
    # this artefact could be quietly wrong.
    point = (None if auc_real["mean"] is None or auc_cross["mean"] is None
             else patient_specific_skill(auc_real["mean"], auc_cross["mean"]))
    consistent = (point is None or skill["mean"] is None
                  or abs(point - skill["mean"]) < 1e-9)

    chain = None
    if template_auc is not None and auc_real["mean"] is not None:
        chain = reference_chain(
            {"chance": 0.5, "fitted_template": template_auc,
             "cross_patient": auc_cross["mean"], "full": auc_real["mean"]},
            ["chance", "fitted_template", "cross_patient", "full"])

    return {
        "tag": tag,
        "frozen_slot": int(slot),
        "axis": cp["axis"],
        "n_derangements": n_der,
        "rng_seed": cp["rng_seed"],
        "impl": cp["impl"],
        "n_bags_scored": len(idx),
        "n_bags_without_any_derangement": len(dropped),
        "flat_auc_real": auc_real,
        "flat_auc_cross_patient": auc_cross,
        # The declared `aggregate: mean_over_derangements`, reported beside the
        # paired form the skill is actually computed from. See the module docstring.
        "flat_auc_cross_patient_mean_over_derangements": float(
            np.nanmean(per_derangement)) if per_derangement else None,
        "patient_specific_skill": skill,
        "patient_specific_skill_point": point,
        "paired_matches_difference_of_means": bool(consistent),
        "reference_chain": chain,
        "seconds": round(time.time() - t0, 1),
    }


def report(r):
    def fmt(s):
        return "     --" if s["mean"] is None else \
            f"{s['mean']:.4f}  [{s['lo']:.4f}, {s['hi']:.4f}]"
    print(f"[{r['tag']}]  frozen_slot={r['frozen_slot']}  "
          f"bags={r['n_bags_scored']}  {r['seconds']}s", flush=True)
    print(f"   flat AUC real          {fmt(r['flat_auc_real'])}", flush=True)
    print(f"   flat AUC cross-patient {fmt(r['flat_auc_cross_patient'])}"
          f"   (over-derangements mean "
          f"{r['flat_auc_cross_patient_mean_over_derangements']:.4f})", flush=True)
    print(f"   patient-specific skill {fmt(r['patient_specific_skill'])}", flush=True)
    if not r["paired_matches_difference_of_means"]:
        print("   WARNING: the paired mean and the difference of means disagree -- "
              "the two paths saw different bags", flush=True)
    if r["n_bags_without_any_derangement"]:
        print(f"   {r['n_bags_without_any_derangement']} bag(s) survived no "
              "derangement and are excluded from the skill", flush=True)
    if r["reference_chain"]:
        for row in r["reference_chain"]:
            print(f"     {row['from']:16s} -> {row['to']:16s} "
                  f"{row['delta']:+.4f}  ({row['frac']:+.1%} of the climb)", flush=True)


def _template_auc_by_tag(path):
    """The pre-registered fitted-template flat AUC per tag, from template_family.json.

    Read rather than recomputed so that the chain's second link and the skill
    denominator in ``template_family.json`` are the same number by construction and
    not by coincidence. The member is ``attention:inplane`` -- the one that file
    marks ``pre_registered_denominator``.
    """
    doc = json.loads(Path(path).read_text())
    out = {}
    for r in doc.get("results", []):
        by = {s["scorer"]: s for s in r.get("scorers", [])}
        row = by.get("attention:inplane")
        if row is not None:
            out[r["tag"]] = row["flat_auc"]["mean"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/nulls")
    ap.add_argument("--meta", default="runs/_audit_meta.json")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="default: every H4 arm_set arm x protocol.seeds present as a dump")
    ap.add_argument("--template-family", default=None,
                    help="template_family.json, for the fitted_template link of the "
                         "reference chain; the chain is omitted without it")
    ap.add_argument("--reps", type=int, default=None,
                    help="default: statistics.bootstrap.reps")
    ap.add_argument("--seed", type=int, default=None,
                    help="default: statistics.bootstrap.rng_seed")
    ap.add_argument("--out", default="runs/nulls/h4_cross_patient.json")
    ap.add_argument("--role", choices=["exploratory", "confirmatory"],
                    default="exploratory",
                    help="the dumps in runs/nulls are from the discovery split, so "
                         "anything scored from them is exploratory; pass confirmatory "
                         "only for seed-2027 dumps")
    args = ap.parse_args()

    pre = prereg.load()
    cp = pre.hypothesis("H4")["cross_patient"]
    reps = args.reps if args.reps is not None else pre.get("statistics.bootstrap.reps")
    boot_seed = (args.seed if args.seed is not None
                 else pre.get("statistics.bootstrap.rng_seed"))

    d = Path(args.dir)
    if args.tags:
        tags = list(args.tags)
    else:
        # H4's arm_set, not every dump on disk. mean and centre_gaussian are
        # excluded by scoring_class -- their attention is not fitted, so a
        # cross-patient swap has nothing to remove and the number would be
        # arithmetic reported as evidence.
        want = [arm_tag(pre.arm(name)["spec"], s)
                for name in pre.arm_set("H4")
                for s in pre.get("protocol.seeds")]
        tags = [t for t in want
                if (d / f"{t}_val.npz").exists() and (d / f"{t}_test.npz").exists()]
        missing = [t for t in want if t not in tags]
        if missing:
            print(f"[h4] {len(missing)} of {len(want)} declared tags have no dump "
                  f"in {d}: {missing}", flush=True)

    if not tags:
        raise SystemExit(f"no dumps to score in {d}")

    templates = _template_auc_by_tag(args.template_family) if args.template_family else {}
    uid_to_pat = json.loads(Path(args.meta).read_text())["pat"]

    print(f"[h4] {len(tags)} tags, {cp['n_derangements']} derangements each, "
          f"{reps} bootstrap reps", flush=True)
    results = []
    for tag in tags:
        r = analyse(tag, d / f"{tag}_val.npz", d / f"{tag}_test.npz", uid_to_pat,
                    cp, reps, boot_seed, templates.get(tag))
        results.append(r)
        report(r)

    payload = pre.stamp({
        "reps": reps, "analysis_role": args.role,
        "cross_patient": cp,
        "unit_over_seeds": "mean_over_seeds",
        "unit_over_seeds_note":
            "H4 declares no unit over seeds. Taken as the mean over seeds by the "
            "argument H5.unit_rationale makes for the identical hole -- 'no ARM's "
            "skill' is the statement's own noun, and a per-seed maximum takes five "
            "draws at the threshold instead of one. Per-seed values are reported "
            "regardless and the verdict runner recomputes the aggregate from them.",
        "results": results,
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}  ({args.role}, "
          f"prereg {payload['prereg']['prereg_hash']})", flush=True)


if __name__ == "__main__":
    main()
