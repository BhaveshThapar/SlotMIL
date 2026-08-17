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

# How many times each STOCHASTIC content-free member is drawn per tag.
#
# It used to be once. `np.random.default_rng(cf_seed)` was constructed fresh per
# tag with cf_seed fixed at 0, so every tag in a condition shared one realisation
# -- roll_permutation's offsets were literally identical across all 35 tags --
# and the spread across tags was arm-to-arm variation with the draw held
# constant. There was therefore no estimate of draw-to-draw variance anywhere,
# which matters because H7's gate is a MAXIMUM over tags: a maximum over one
# realisation cannot distinguish a member that really sits above 0.05 from a
# member whose one draw happened to. Neither a max nor a mean is well founded on
# a single realisation; the fix is to draw enough times to summarise, not to
# change what the gate does with the summary.
#
# 30 because the protocol already uses ">= 30" for the one other place it makes a
# distributional claim (the untrained-init floor, reported as median and
# 5th-95th percentile). Reusing a number this document already committed to is a
# better justification than a fresh one chosen while looking at a result.
CONTENT_FREE_DRAWS = 30


def _flat(attns, masks, slot, uids, uid_to_pat, reps, seed):
    """Flat-axis cluster-bootstrapped AUC through the one shared scoring path."""
    rows = per_bag_axes(attns, masks, slot)
    pats = [uid_to_pat.get(str(uids[r[0]]), str(uids[r[0]])) for r in rows]
    return cluster_bootstrap([r[FLAT] for r in rows], pats, reps, seed), len(rows)


def _skill(numer, denom):
    if numer["mean"] is None or denom["mean"] is None:
        return None
    return prior_normalised_skill(numer["mean"], denom["mean"])


def _summarise_draws(skills):
    """Collapse one member's per-draw skills at one tag into the gate's input.

    The gate keeps its pre-registered aggregation -- a maximum over tags, so that
    no arm can hide a badly-behaved one inside an average. What changes is only
    how a single tag's value is estimated: the MEAN over draws, which is the
    member's expected skill for that arm rather than whichever draw came up. The
    full distribution is carried alongside, because the whole point of drawing 30
    times is that a reader can see the spread the single-draw form hid.
    """
    vals = [s for s in skills if s is not None]
    if not vals:
        return None, {"draws": len(skills), "computable": 0}
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), {
        "draws": len(skills), "computable": int(a.size),
        "per_draw": [float(v) for v in a],
        "mean": float(a.mean()), "median": float(np.median(a)),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()), "max": float(a.max()),
        # Draw 0 is seeded exactly as the single-realisation form was, so the
        # published number stays locatable inside its own distribution rather
        # than being silently replaced by one computed a different way.
        "single_draw": float(a[0]),
    }


def _constant_like(masks):
    """A K=1 scorer that is the same everywhere: chance, built not asserted."""
    return [np.ones((1, len(m)), dtype=np.float32) for m in masks]


def analyse(tag, val_npz, test_npz, uid_to_pat, reps, seed, cf_seed,
            cf_draws=CONTENT_FREE_DRAWS):
    val_a, val_m = load(val_npz)
    test_a, test_m = load(test_npz)
    slot, _ = pick_lesion_slot(val_a, val_m)
    uids = np.load(test_npz, allow_pickle=True)["uids"]
    k = int(test_a[0].shape[0])

    def flat(attns, masks, s=0, reps=reps):
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
    # Draw d is seeded cf_seed + d, so draw 0 reproduces the single-realisation
    # number exactly and the remaining draws are independent streams (numpy runs
    # an integer seed through SeedSequence, so consecutive seeds are not
    # correlated). Only draw 0 pays for the bootstrap: reps=0 returns the point
    # estimate, and an interval on one draw of a null is not a reported quantity.
    template = global_template(val_a, val_m, slot)   # fit on the TRUE val masks
    roll_skills, roll_num, roll_den, n = [], None, None, 0
    for d in range(cf_draws):
        rolled = roll_masks(test_m, np.random.default_rng(cf_seed + d))
        num, n = flat(test_a, rolled, slot, reps=reps if d == 0 else 0)
        den, _ = flat(template_scores(rolled, template), rolled,
                      reps=reps if d == 0 else 0)
        roll_skills.append(_skill(num, den))
        if d == 0:
            roll_num, roll_den = num, den
    skill, spread = _summarise_draws(roll_skills)
    members["roll_permutation"] = {
        "auc": roll_num, "auc_denominator": roll_den, "skill": skill,
        "n_bags_scored": n, "computable": skill is not None,
        "draw_distribution": spread,
        "construction": "nulls.roll_masks on the target; the arm's attention is "
                        "unmoved and its template is fit on the true val masks, "
                        "with both numerator and denominator scored against the "
                        f"rolled target. Drawn {cf_draws} times, seeds "
                        f"{cf_seed}..{cf_seed + cf_draws - 1}; the reported skill "
                        "is the mean over draws. auc and auc_denominator are "
                        "draw 0's, the only draw carrying an interval.",
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
        # match_scale is deterministic in (target_h, k), so it is fit once and
        # every draw shares it: the member is "random attention AT the arm's
        # entropy", and refitting the scale per draw would let the entropy match
        # itself wander between draws.
        target_h = slot_entropy(test_a)
        scale = match_scale(target_h, k)
        rand_skills, rand_num, rand_den, achieved = [], None, None, []
        for d in range(cf_draws):
            rng = np.random.default_rng(cf_seed + d)
            rand_val = random_attn([len(m) for m in val_m], k, scale, rng)
            rand_test = random_attn([len(m) for m in test_m], k, scale, rng)
            num, n = flat(rand_test, test_m, reps=reps if d == 0 else 0)
            den, _ = flat(template_scores(test_m, global_template(rand_val, val_m, 0)),
                          test_m, reps=reps if d == 0 else 0)
            rand_skills.append(_skill(num, den))
            achieved.append(slot_entropy(rand_test))
            if d == 0:
                rand_num, rand_den = num, den
        skill, spread = _summarise_draws(rand_skills)
        members["entropy_matched_random"] = {
            "auc": rand_num, "auc_denominator": rand_den, "skill": skill,
            "n_bags_scored": n, "computable": skill is not None, "k": k,
            "target_slot_entropy": target_h, "matched_scale": scale,
            "achieved_slot_entropy": float(np.mean(achieved)),
            "achieved_slot_entropy_per_draw": [float(v) for v in achieved],
            "max_possible_entropy": float(np.log(k)),
            "draw_distribution": spread,
            "construction": "nulls.random_attn at nulls.match_scale, k and entropy "
                            "target from the accompanying arm's dump. Drawn "
                            f"{cf_draws} times, seeds {cf_seed}.."
                            f"{cf_seed + cf_draws - 1}; the reported skill is the "
                            "mean over draws. The scale is fit once and shared, "
                            "so the entropy match does not wander between draws.",
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
    ap.add_argument("--cf-draws", type=int, default=None,
                    help="draws per tag for the two stochastic content-free "
                         "members; default: hypotheses.H7.content_free_draws if "
                         "declared, else CONTENT_FREE_DRAWS")
    ap.add_argument("--out", default="runs/nulls/h7_content_free.json")
    ap.add_argument("--role", choices=["exploratory", "confirmatory"],
                    default="exploratory")
    args = ap.parse_args()

    pre = prereg.load()
    h7 = pre.hypothesis("H7")
    reps = args.reps if args.reps is not None else pre.get("statistics.bootstrap.reps")
    boot_seed = (args.seed if args.seed is not None
                 else pre.get("statistics.bootstrap.rng_seed"))
    # Declared if the config declares it, taken if not -- and either way the
    # number and which of the two it was land in the artefact, so a reader never
    # has to open this file to find out how many draws stand behind a member.
    declared_draws = h7.get("content_free_draws")
    cf_draws = (args.cf_draws if args.cf_draws is not None
                else declared_draws if declared_draws is not None
                else CONTENT_FREE_DRAWS)
    draws_basis = ("cli" if args.cf_draws is not None
                   else "declared" if declared_draws is not None else "taken")

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
                    reps, boot_seed, boot_seed, cf_draws)
        results.append(r)
        report(r)

    # The gate reads the worst computed member over every tag: "every content-free
    # baseline's is below 0.05" is a statement about the set, not about an average
    # over it. Unchanged by the replication -- each tag's value is now a mean over
    # draws rather than one draw, but what the gate does with those values is the
    # rule as written.
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
        "content_free_draws": cf_draws,
        "content_free_draws_basis": draws_basis,
        "content_free_draws_note":
            "The two stochastic members are drawn cf_draws times per tag and each "
            "tag's reported skill is the mean over its draws. Before this, every "
            "tag in a condition shared ONE realisation -- default_rng(0) rebuilt "
            "per tag -- so there was no draw-to-draw variance estimate anywhere "
            "and a maximum over tags could not tell a member that sits above the "
            "threshold from a member whose single draw did. The aggregation the "
            "gate applies to the per-tag values is unchanged.",
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
