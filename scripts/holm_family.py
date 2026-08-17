#!/usr/bin/env python
"""The declared Holm family: learned_attention arms x three content-free references.

``statistics.multiplicity: holm`` was declared at freeze time over a family the
config did not define, and the 2026-08-15 amendment fixed that by *enumerating*
one -- ``statistics.holm_family``, seven ``learned_attention`` arms against
``fitted_template``, ``centre_prior`` and ``cross_patient`` on the flat instance
axis, over the single condition carrying the hypothesis family. Its ``note`` says
why enumeration rather than computation: the family size sets every adjusted
p-value, so it must not be changeable by deciding later to report one more
comparison.

``classification.delong_test`` and ``classification.holm_reject`` have been
implemented and unit-tested since then with no caller outside ``tests/``. This is
the caller. Running it turned up two places where the enumerated family is not
computable exactly as written, and both are recorded in the artefact rather than
quietly resolved:

**1. ``cross_patient`` cannot be a paired DeLong comparison.** DeLong tests two
scores against *one* set of labels. ``cross_patient`` is a label-side null --
``reference_baselines`` gives its impl as ``shuffle_masks_across_bags``, and it
changes whose masks the attention is scored against while leaving the score
untouched. There is no second score vector to pair, so no z and no p exist for
these seven members. They are entered into the family with ``p = NaN``, which is
the behaviour ``holm_adjust`` was written for and documents: "NaN in, NaN out --
**and the NaN still counts toward the family size**". So the family stays at 21,
the surviving members keep their full multipliers, and a comparison that could not
be computed is never rejected. The patient-clustered interval on the paired per-bag
difference *is* well defined, and ``h4_cross_patient.py`` reports it; it is
referenced here rather than restated.

**2. the declared axis and DeLong measure two different things.**
``estimands.secondary`` gives ``flat_instance_auc`` the impl ``null_battery.score``,
which is the **mean of per-bag AUCs**. DeLong is defined for a single AUC over one
sample set and has no meaning applied to an average of per-bag AUCs. So the z and p
here are computed on the **pooled** flat instance AUC over the bags
``per_bag_axes`` keeps, and the declared mean-per-bag statistic is reported beside
every comparison. Both numbers are in the artefact under distinct names. Neither is
presented as the other.

**Seeds are not in the enumeration, and the family is corrected per seed.** Pooling
five seeds into one 105-member family would change the size the ``note`` pins;
pooling their instances into one comparison would duplicate every patch five times
and shrink the DeLong variance by a factor it has not earned. Each seed therefore
carries its own 21-member family, and all five are reported.

    python scripts/holm_family.py \\
        --dir runs/nulls_nodule_present_confirmatory \\
        --out runs/nulls_nodule_present_confirmatory/holm_family.json \\
        --role confirmatory
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
from slotmil.eval.classification import delong_test, holm_reject
from slotmil.eval.nulls import global_template, template_scores
from slotmil.eval.verdict import arm_tag

FLAT = 1

# cross_patient is a label-side null: it changes the target, not the score, so a
# paired test on two scores over one label set cannot express it. See the module
# docstring; the member stays in the family with p = NaN.
NOT_PAIRABLE = {
    "cross_patient": "a label-side null (nulls.shuffle_masks_across_bags changes "
                     "the target, not the score), so there is no second score "
                     "vector to pair and no DeLong z or p exists; the member stays "
                     "in the family at p = NaN and is never rejected, and the "
                     "well-defined paired interval is in h4_cross_patient.json",
}


def _pool(attns, masks, kept, slot):
    """Concatenate one scorer's instances over the kept bags, with their targets."""
    score = np.concatenate([attns[i][slot] for i in kept]).astype(np.float64)
    target = np.concatenate([(np.asarray(masks[i]) > 0).astype(np.int8) for i in kept])
    return target, score


def _mean_per_bag(attns, masks, slot):
    rows = per_bag_axes(attns, masks, slot)
    return (float(np.nanmean([r[FLAT] for r in rows])) if rows else None,
            [r[0] for r in rows])


def analyse_seed(seed, tags_by_arm, d, references):
    """One seed's 21-member family: every arm against every reference.

    No patient map and no bootstrap here: Holm corrects DeLong p-values, and
    DeLong's variance comes from its own structural components rather than from
    resampling. The patient-clustered intervals on these same AUCs live in
    ``axis_gate.json`` and ``h4_cross_patient.json``.
    """
    members = []
    for arm, tag in tags_by_arm.items():
        val_a, val_m = load(d / f"{tag}_val.npz")
        test_a, test_m = load(d / f"{tag}_test.npz")
        slot, _ = pick_lesion_slot(val_a, val_m)

        arm_mean, kept = _mean_per_bag(test_a, test_m, slot)
        y, s_arm = _pool(test_a, test_m, kept, slot)

        # Both references that CAN be paired are K=1 scorers over the real target.
        ref_scores = {
            "fitted_template": template_scores(
                test_m, global_template(val_a, val_m, slot)),
            "centre_prior": centre_prior_scores(test_m),
        }

        for ref in references:
            row = {"arm": arm, "tag": tag, "reference": ref, "seed": seed,
                   "arm_flat_instance_auc_mean_per_bag": arm_mean,
                   "n_bags_pooled": len(kept), "n_instances_pooled": int(y.size)}
            if ref in NOT_PAIRABLE:
                row.update({"p": float("nan"), "z": None, "delta": None,
                            "paired_delong_applicable": False,
                            "reason": NOT_PAIRABLE[ref]})
                members.append(row)
                continue

            ref_mean, ref_kept = _mean_per_bag(ref_scores[ref], test_m, 0)
            if ref_kept != kept:
                # The drop rules in per_bag_axes depend only on the mask, so every
                # scorer over one bag set yields the same indices. Checked rather
                # than assumed: a mismatch means the pooled vectors are not paired,
                # which DeLong would not notice.
                raise SystemExit(
                    f"{tag}/{ref}: the reference kept {len(ref_kept)} bags and the "
                    f"arm kept {len(kept)}; the pooled scores are not paired")
            _, s_ref = _pool(ref_scores[ref], test_m, kept, 0)
            t = delong_test(y, s_arm, s_ref)
            row.update({
                "pooled_flat_instance_auc_arm": t["auc_a"],
                "pooled_flat_instance_auc_reference": t["auc_b"],
                "reference_flat_instance_auc_mean_per_bag": ref_mean,
                "delta": t["delta"], "z": t["z"], "p": t["p"],
                "paired_delong_applicable": True,
            })
            members.append(row)
        del val_a, val_m, test_a, test_m, ref_scores

    # Holm is applied by the caller, once, over the whole seed's family -- not here
    # and not per arm. Correcting inside this loop would give each arm its own
    # three-member family and quietly triple the effective alpha.
    return members


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/nulls")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="default: protocol.seeds")
    ap.add_argument("--out", default="runs/nulls/holm_family.json")
    ap.add_argument("--role", choices=["exploratory", "confirmatory"],
                    default="exploratory")
    args = ap.parse_args()

    pre = prereg.load()
    fam = pre.get("statistics.holm_family")
    alpha = fam["alpha"]
    references = list(fam["references"])
    arms = pre.arm_set("H1")
    seeds = args.seeds if args.seeds else pre.get("protocol.seeds")
    d = Path(args.dir)

    expected = len(arms) * len(references)
    print(f"[holm] family of {len(arms)} arms x {len(references)} references = "
          f"{expected} members, corrected per seed, alpha={alpha}", flush=True)
    print(f"[holm] axis declared as {fam['axis']} (impl "
          f"{pre.estimand('flat_instance_auc')['impl']}, a mean of per-bag AUCs); "
          "DeLong is computed on the pooled form and both are reported", flush=True)

    per_seed = []
    for seed in seeds:
        tags_by_arm = {}
        for name in arms:
            tag = arm_tag(pre.arm(name)["spec"], seed)
            if (d / f"{tag}_val.npz").exists() and (d / f"{tag}_test.npz").exists():
                tags_by_arm[name] = tag
            else:
                print(f"[holm] seed {seed}: no dump for {name} ({tag})", flush=True)
        if not tags_by_arm:
            print(f"[holm] seed {seed}: no dumps, skipped", flush=True)
            continue

        members = analyse_seed(seed, tags_by_arm, d, references)
        holm = holm_reject([m["p"] for m in members], alpha=alpha)
        for m, padj, rej in zip(members, holm["p_adjusted"], holm["reject"]):
            m["p_adjusted"] = None if np.isnan(padj) else float(padj)
            m["reject"] = bool(rej)

        # The family size the note pins. A short family means a missing dump made
        # every surviving member easier to reject, which is the one direction a
        # multiplicity correction must never move -- so it is stated, not inferred.
        short = holm["n_family"] < expected

        # At pooled-instance scale the DeLong z is large enough that the two-sided
        # p underflows to exactly 0.0, and a family of zeros is rejected wholesale
        # whatever the multiplier. Counted and reported: "4/6 rejected" would
        # otherwise read as a correction that did work, when what happened is that
        # 7.4M paired instances leave no room for one not to.
        underflow = sum(1 for m in members if m["p"] == 0.0)
        print(f"[holm] seed {seed}: {holm['n_reject']}/{holm['n_family']} rejected "
              f"at alpha={alpha}"
              f"{'   FAMILY SHORT of ' + str(expected) if short else ''}", flush=True)
        if underflow:
            print(f"[holm] seed {seed}: {underflow} member(s) have p underflowed to "
                  "exactly 0 at pooled-instance scale -- the correction cannot "
                  "change those verdicts, and the patient-clustered intervals on "
                  "the declared mean-per-bag axis are the informative statistic",
                  flush=True)
        for m in members:
            padj = "     --" if m["p_adjusted"] is None else f"{m['p_adjusted']:.4g}"
            flag = "" if m["paired_delong_applicable"] else "   (not pairable)"
            print(f"     {m['arm']:18s} vs {m['reference']:16s} "
                  f"p_adj={padj:>10s} {'REJECT' if m['reject'] else '      '}{flag}",
                  flush=True)

        per_seed.append({
            "seed": seed, "n_family": holm["n_family"],
            "n_family_expected": expected, "family_short": bool(short),
            "n_reject": holm["n_reject"], "alpha": holm["alpha"],
            "n_p_underflowed_to_zero": underflow,
            "members": members,
        })

    payload = pre.stamp({
        "analysis_role": args.role,
        "holm_family": fam,
        "arms": arms,
        "references": references,
        "family_size_per_seed": expected,
        "correction_unit": "per_seed",
        "correction_unit_note":
            "Seeds are not in the enumerated scope. Pooling five seeds into one "
            "105-member family would change the size statistics.holm_family.note "
            "pins, and pooling their instances into one comparison would duplicate "
            "every patch five times and shrink the DeLong variance by a factor it "
            "has not earned. Each seed carries its own family; all are reported.",
        "delong_axis_note":
            "statistics.holm_family.axis is flat_instance_auc, whose declared impl "
            "(null_battery.score) is the mean of per-bag AUCs. DeLong is defined for "
            "a single AUC over one sample set, so z and p are computed on the pooled "
            "flat instance AUC over the bags per_bag_axes keeps, and the declared "
            "mean-per-bag statistic is reported beside every comparison under its "
            "own name.",
        "power_note":
            "A pooled comparison carries ~7.4M paired instances, at which the DeLong "
            "z is large enough that the two-sided p underflows to exactly 0. Every "
            "pairable member is then rejected regardless of the Holm multiplier, so "
            "the correction is real but not load-bearing here, and 'n of m rejected' "
            "must not be read as evidence that it discriminated. What discriminates "
            "at this sample size is the effect size and the patient-clustered "
            "interval on the declared mean-per-bag axis; both are reported, in this "
            "file and in axis_gate.json. n_p_underflowed_to_zero counts it per seed "
            "rather than leaving a reader to infer it from a column of zeros.",
        "not_pairable": NOT_PAIRABLE,
        "per_seed": per_seed,
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}  ({args.role}, "
          f"prereg {payload['prereg']['prereg_hash']})", flush=True)


if __name__ == "__main__":
    main()
