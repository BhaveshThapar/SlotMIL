#!/usr/bin/env python
"""H7's content-free half: the four members that must all score below 0.05.

``scripts/probe_gate.py`` computes the other half and says so in its own output --
``content_free.status`` reads "not computed here -- needs the confirmatory attention
dumps; H7 clears only if this half holds too". This is that half.

H7 is the power gate. If the supervised probe cannot separate from content-free
references on the proposed estimand, the estimand has no power and the paper must
not recommend it -- the gate blocks the constructive half. Which makes the
*construction* of the content-free set the load-bearing part, and
``H7.content_free_construction`` pins all four rather than leaving them to a script:

``chance``
    Analytic 0.5. Built here as a constant K=1 scorer pushed through the identical
    ``per_bag_axes`` path as everything else, so the 0.5 is measured by the same
    code rather than asserted beside it. Its own template is constant too, so the
    skill lands on exactly 0.0.
``roll_permutation``
    ``nulls.roll_masks`` rolls the **target**, per bag, in patches. The scorer is
    unmoved -- which the config states explicitly, because rolling the score
    instead gives the same AUC against a *different* denominator. So the arm's own
    template is fit against the TRUE validation masks, and both numerator and
    denominator are then evaluated against the rolled test target: the estimand is
    a ratio of two AUCs and they have to be over the same target for the ratio to
    mean anything.
``entropy_matched_random``
    ``nulls.random_attn`` at a scale from ``nulls.match_scale``, matched to the
    accompanying arm's own test-split ``nulls.slot_entropy``, with k from the arm's
    dump. **Undefined for a K=1 arm**: ``random_attn(k=1)`` softmaxes one slot and
    returns exactly 1.0 everywhere, so every instance ties and the AUC does not
    exist. That is the same objection ``content_free_construction`` raises against
    taking k from the probe. Measured K over the seven ``learned_attention`` arms is
    ``{1: abmil, gated_abmil, clam_sb, normal_guidance; 2: dsmil; 8: mh_abmil,
    slot:div=0.5}`` -- so four arms cannot carry the member and three can. Those
    four are reported as not computable, with the reason, and cannot falsify the
    gate. (``mh_abmil`` emits K=8, not 4: it is parameter-matched to slot attention
    at ``num_slots=8``. The ``4`` in ENGINEERING.md's ``[4, 45312, 2304]`` OOM note is a
    batch dimension.)
``fitted_template``
    ``nulls.global_template`` on the member's own validation attention, which is
    its own denominator and therefore scores exactly 0. Included, per the config,
    so the set is never empty -- which is also what keeps the gate from being
    vacuous when the member above is undefined.

Every member carries its **own** val-fitted in-plane denominator, per
``denominator_rule``. A shared denominator would make each member's verdict depend
on whose it was.

The three ``reported_beside_the_gate`` rows are read out of
``template_family.json`` rather than recomputed, so the table beside the gate and
the table in that file cannot disagree. They are reported whether or not H7 clears,
and they are deliberately *not* in the gate -- ``content_free_set_rationale``
excludes the mask-fitted family as fit to lesion labels and ``centre_prior`` as
axially informative against an in-plane denominator. The cost of those exclusions
is that skill against an in-plane denominator is an upper bound for any scorer
carrying axial structure, and it is paid in the paper.

    python scripts/h7_content_free.py \\
        --dir runs/nulls_nodule_present_confirmatory \\
        --template-family runs/nulls_nodule_present_confirmatory/template_family.json \\
        --out runs/nulls_nodule_present_confirmatory/h7_content_free.json \\
        --role confirmatory
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from null_battery import load, pick_lesion_slot

from slotmil import prereg
from slotmil.eval.axes import per_bag_axes
from slotmil.eval.estimands import cluster_bootstrap, prior_normalised_skill
from slotmil.eval.nulls import (
    global_template,
    match_scale,
    random_attn,
    roll_masks,
    slot_entropy,
    template_scores,
)
from slotmil.eval.verdict import arm_tag

FLAT = 1


def _flat(attns, masks, slot, uids, uid_to_pat, reps, seed):
    """Flat-axis cluster-bootstrapped AUC through the one shared scoring path."""
    rows = per_bag_axes(attns, masks, slot)
    pats = [uid_to_pat.get(str(uids[r[0]]), str(uids[r[0]])) for r in rows]
    return cluster_bootstrap([r[FLAT] for r in rows], pats, reps, seed), len(rows)


def _skill(numer, denom):
    if numer["mean"] is None or denom["mean"] is None:
        return None
    return prior_normalised_skill(numer["mean"], denom["mean"])


def _constant_like(masks):
    """A K=1 scorer that is the same everywhere: chance, built not asserted."""
    return [np.ones((1, len(m)), dtype=np.float32) for m in masks]


def analyse(tag, val_npz, test_npz, uid_to_pat, reps, seed, cf_seed):
    val_a, val_m = load(val_npz)
    test_a, test_m = load(test_npz)
    slot, _ = pick_lesion_slot(val_a, val_m)
    uids = np.load(test_npz, allow_pickle=True)["uids"]
    k = int(test_a[0].shape[0])

    def flat(attns, masks, s=0):
        return _flat(attns, masks, s, uids, uid_to_pat, reps, seed)

    members: dict[str, dict] = {}

    # ---- chance: analytic 0.5, measured through the shared path ---------------
    const_val, const_test = _constant_like(val_m), _constant_like(test_m)
    num, n = flat(const_test, test_m)
    den, _ = flat(template_scores(test_m, global_template(const_val, val_m, 0)), test_m)
    members["chance"] = {"auc": num, "auc_denominator": den, "skill": _skill(num, den),
                         "n_bags_scored": n, "construction": "analytic 0.5",
                         "computable": True}

    # ---- roll_permutation: roll the TARGET, leave the scorer alone ------------
    rolled = roll_masks(test_m, np.random.default_rng(cf_seed))
    template = global_template(val_a, val_m, slot)   # fit on the TRUE val masks
    num, n = flat(test_a, rolled, slot)
    den, _ = flat(template_scores(rolled, template), rolled)
    members["roll_permutation"] = {
        "auc": num, "auc_denominator": den, "skill": _skill(num, den),
        "n_bags_scored": n, "computable": True,
        "construction": "nulls.roll_masks on the target; the arm's attention is "
                        "unmoved and its template is fit on the true val masks, "
                        "with both numerator and denominator scored against the "
                        "rolled target",
    }

    # ---- entropy_matched_random: needs K > 1 ---------------------------------
    if k < 2:
        members["entropy_matched_random"] = {
            "auc": None, "auc_denominator": None, "skill": None, "computable": False,
            "reason": f"the accompanying arm emits K={k}; nulls.random_attn(k=1) "
                      "softmaxes one slot and returns exactly 1.0 everywhere, so "
                      "every instance ties and the AUC is undefined -- the same "
                      "objection H7.content_free_construction raises against "
                      "taking k from the probe",
            "construction": "nulls.random_attn at nulls.match_scale, k and entropy "
                            "from the accompanying arm's dump",
        }
    else:
        target_h = slot_entropy(test_a)
        scale = match_scale(target_h, k)
        rng = np.random.default_rng(cf_seed)
        rand_val = random_attn([len(m) for m in val_m], k, scale, rng)
        rand_test = random_attn([len(m) for m in test_m], k, scale, rng)
        num, n = flat(rand_test, test_m)
        den, _ = flat(template_scores(test_m, global_template(rand_val, val_m, 0)),
                      test_m)
        members["entropy_matched_random"] = {
            "auc": num, "auc_denominator": den, "skill": _skill(num, den),
            "n_bags_scored": n, "computable": True, "k": k,
            "target_slot_entropy": target_h, "matched_scale": scale,
            "achieved_slot_entropy": slot_entropy(rand_test),
            "max_possible_entropy": float(np.log(k)),
            "construction": "nulls.random_attn at nulls.match_scale, k and entropy "
                            "target from the accompanying arm's dump",
        }

    # ---- fitted_template: its own denominator, so exactly 0 ------------------
    scored = template_scores(test_m, template)
    num, n = flat(scored, test_m)
    members["fitted_template"] = {
        "auc": num, "auc_denominator": num, "skill": _skill(num, num),
        "n_bags_scored": n, "computable": True,
        "construction": "nulls.global_template on the member's own validation "
                        "attention, which is its own denominator and therefore "
                        "scores exactly 0; included so the set is never empty",
    }

    return {"tag": tag, "frozen_slot": int(slot), "k": k, "members": members}


def report(r):
    def fmt(s):
        return "     --" if s is None or s["mean"] is None else \
            f"{s['mean']:.4f}  [{s['lo']:.4f}, {s['hi']:.4f}]"
    print(f"[{r['tag']}]  frozen_slot={r['frozen_slot']}  K={r['k']}", flush=True)
    print(f"   {'member':24s} {'auc':24s} {'own denominator':24s} skill", flush=True)
    for name, m in r["members"].items():
        skill = "  not computable" if m["skill"] is None else f"{m['skill']:+.4f}"
        print(f"   {name:24s} {fmt(m['auc']):24s} "
              f"{fmt(m['auc_denominator']):24s} {skill}", flush=True)
        if not m["computable"]:
            print(f"      {m['reason']}", flush=True)


def _beside_the_gate(path, members):
    """The reported_beside_the_gate rows, from template_family.json.

    Read rather than recomputed: these rows also appear in that file, and two
    computations of one number under one name is how they come to disagree.
    """
    doc = json.loads(Path(path).read_text())
    out = {}
    for r in doc.get("results", []):
        by = {s["scorer"]: s for s in r.get("scorers", [])}
        denom = by.get("attention:inplane")
        row = {}
        for member in members:
            scorer = by.get(member)
            if scorer is None or denom is None:
                continue
            row[member] = {
                "flat_auc": scorer["flat_auc"],
                "skill_against_the_pinned_denominator": (
                    None if scorer["flat_auc"]["mean"] is None
                    or denom["flat_auc"]["mean"] is None
                    else prior_normalised_skill(scorer["flat_auc"]["mean"],
                                                denom["flat_auc"]["mean"])),
            }
        if row:
            out[r["tag"]] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/nulls")
    ap.add_argument("--meta", default="runs/_audit_meta.json")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="default: every H1 arm_set arm x protocol.seeds present as a dump")
    ap.add_argument("--template-family", default=None,
                    help="template_family.json, for the reported_beside_the_gate rows")
    ap.add_argument("--reps", type=int, default=None,
                    help="default: statistics.bootstrap.reps")
    ap.add_argument("--seed", type=int, default=None,
                    help="default: statistics.bootstrap.rng_seed; also the RNG for "
                         "the roll and the random-attention members")
    ap.add_argument("--out", default="runs/nulls/h7_content_free.json")
    ap.add_argument("--role", choices=["exploratory", "confirmatory"],
                    default="exploratory")
    args = ap.parse_args()

    pre = prereg.load()
    h7 = pre.hypothesis("H7")
    reps = args.reps if args.reps is not None else pre.get("statistics.bootstrap.reps")
    boot_seed = (args.seed if args.seed is not None
                 else pre.get("statistics.bootstrap.rng_seed"))

    d = Path(args.dir)
    if args.tags:
        tags = list(args.tags)
    else:
        want = [arm_tag(pre.arm(name)["spec"], s)
                for name in pre.arm_set("H1")
                for s in pre.get("protocol.seeds")]
        tags = [t for t in want
                if (d / f"{t}_val.npz").exists() and (d / f"{t}_test.npz").exists()]
        missing = [t for t in want if t not in tags]
        if missing:
            print(f"[h7] {len(missing)} of {len(want)} declared tags have no dump "
                  f"in {d}: {missing}", flush=True)
    if not tags:
        raise SystemExit(f"no dumps to score in {d}")

    uid_to_pat = json.loads(Path(args.meta).read_text())["pat"]
    results = []
    for tag in tags:
        r = analyse(tag, d / f"{tag}_val.npz", d / f"{tag}_test.npz", uid_to_pat,
                    reps, boot_seed, boot_seed)
        results.append(r)
        report(r)

    # The gate reads the worst computed member over every tag: "every content-free
    # baseline's is below 0.05" is a statement about the set, not about an average
    # over it.
    worst = None
    for r in results:
        for name, m in r["members"].items():
            if m["skill"] is not None and (worst is None or m["skill"] > worst["skill"]):
                worst = {"skill": m["skill"], "member": name, "tag": r["tag"]}
    threshold = h7["threshold"]["content_free"]

    payload = pre.stamp({
        "reps": reps, "analysis_role": args.role,
        "content_free_set": h7["content_free_set"],
        "denominator_rule": h7["content_free_construction"]["denominator_rule"],
        "h7_content_free_half": {
            "threshold": threshold,
            "worst_computed": worst,
            "clears": None if worst is None else bool(worst["skill"] < threshold),
            "not_computable": sorted({
                f"{r['tag']}:{name}" for r in results
                for name, m in r["members"].items() if not m["computable"]}),
        },
        "reported_beside_the_gate": (
            _beside_the_gate(args.template_family, h7["reported_beside_the_gate"])
            if args.template_family else None),
        "reported_beside_the_gate_rule": h7["reported_beside_the_gate_rule"],
        "results": results,
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))

    h = payload["h7_content_free_half"]
    if worst is not None:
        print(f"\n[h7] worst computed content-free skill {worst['skill']:+.4f} "
              f"({worst['member']} on {worst['tag']}) vs < {threshold}: "
              f"{'CLEARS' if h['clears'] else 'FAILS'}", flush=True)
    if h["not_computable"]:
        print(f"[h7] {len(h['not_computable'])} member-tag pairs not computable "
              "(K=1 arms cannot carry entropy_matched_random)", flush=True)
    print(f"wrote {args.out}  ({args.role}, "
          f"prereg {payload['prereg']['prereg_hash']})", flush=True)


if __name__ == "__main__":
    main()
