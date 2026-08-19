#!/usr/bin/env python
"""H5 sensitivity: the same estimand with the denominator floored at chance.

**This is not a pre-registered analysis and it does not change H5's verdict.**
The confirmatory H5 result is whatever ``scripts/prereg_verdict.py`` scores
against the estimand declared in ``configs/prereg/isbi2027.yaml``. This script
exists because that estimand has a property worth reporting rather than
defending.

The pre-registered denominator is ``attention:inplane`` -- an in-plane template
fit on validation *attention*, pinned as ``reference_baselines.fitted_template``.
Fit to the supervised probe's output that same rule scores 0.836; fit to a MIL
arm's attention its median over the learned-attention dumps is **0.4208**, and it
sits below 0.5 for 24 of 40 tags on the condition that carries the family. That
is the mechanism the paper reports -- the denominator adapts to the arm, and it
collapses below chance for MIL arms precisely because their attention carries no
in-plane structure to fit -- but it has an arithmetic consequence:
``(AUC - AUC_t) / (1 - AUC_t)`` is then not "the fraction of achievable gain over
a prior" as the phrase suggests, and a sub-chance denominator inflates the
denominator term ``1 - AUC_t`` above 1, which *shrinks* the reported skill and
makes H5 -- a hypothesis that passes by staying **below** a bar -- easier to pass.

So the caveat cuts against us, and the honest thing is to compute the
conservative direction rather than only disclose it. Flooring the denominator at
0.5 asks: if no content-free reference is allowed to score worse than chance,
does any arm's prior-normalised skill clear 0.30?

Aggregation is deliberately identical to ``slotmil.eval.verdict.h5_verdict`` --
floor per tag, mean over the arm's seeds, maximum over the arm set returned by
``Prereg.arm_set("H5")`` -- so the floored and published numbers are the same
statistic under two denominators and not two statistics. The threshold is read
from the frozen config rather than retyped; reading never rehashes it.

    python scripts/h5_floored_denominator.py \
        --dir runs/nulls_nodule_present_confirmatory \
        --out runs/nulls_nodule_present_confirmatory/h5_floored_denominator.json \
        --condition nodule_present --role confirmatory
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slotmil import prereg  # noqa: E402
from slotmil.eval.estimands import prior_normalised_skill  # noqa: E402
from slotmil.eval.verdict import arm_tag  # noqa: E402

DEFAULT_FLOOR = 0.5


def _scorer(row: dict, name: str) -> dict | None:
    for s in row.get("scorers", []):
        if s.get("scorer") == name:
            return s
    return None


def per_tag_skills(doc: dict, floor: float) -> dict[str, dict]:
    """``{tag: {auc, auc_template, published, floored}}`` for every result row.

    Reads ``prior_normalised_skill.auc_template`` and cross-checks it against the
    ``attention:inplane`` scorer's ``flat_auc.mean``. They are the same number by
    construction (``scripts/template_family.py`` builds the first from the
    second), and asserting it here is what stops this script from silently
    normalising against the *probe's* template, which follows the identical rule
    and scores 0.836 rather than 0.42.
    """
    out = {}
    for row in doc.get("results", []):
        tag = row["tag"]
        pns = row.get("prior_normalised_skill")
        trained = _scorer(row, "trained")
        inplane = _scorer(row, "attention:inplane")
        if pns is None or trained is None or inplane is None:
            continue
        auc = trained["flat_auc"]["mean"]
        den = pns["auc_template"]
        den_check = inplane["flat_auc"]["mean"]
        if abs(den - den_check) > 1e-12:
            raise SystemExit(
                f"{tag}: prior_normalised_skill.auc_template ({den}) is not the "
                f"attention:inplane scorer's flat_auc.mean ({den_check}). The "
                "denominator this script floors is not the one that was scored; "
                "refusing to report a sensitivity for a different estimand.")
        out[tag] = {
            "auc": auc,
            "auc_template": den,
            "template_below_floor": bool(den < floor),
            "published": pns["trained"],
            "floored": prior_normalised_skill(auc, max(den, floor)),
        }
    return out


def by_arm(pre, per_tag: dict[str, dict], seeds: list[int]) -> dict[str, dict]:
    """Mean over seeds per arm, over ``arm_set("H5")``.

    Mirrors ``prereg_verdict._by_arm``: tags are built from the arm *spec*, not
    the arm name, because ``slot:div=0.5`` has name ``slot_div0.5`` and dump
    directory ``slot_div=0.5``.
    """
    out = {}
    for name in pre.arm_set("H5"):
        spec = pre.arm(name)["spec"]
        rows = [per_tag.get(arm_tag(spec, s)) for s in seeds]
        present = [r for r in rows if r is not None]
        if not present:
            continue
        out[name] = {
            "published": statistics.fmean(r["published"] for r in present),
            "floored": statistics.fmean(r["floored"] for r in present),
            "auc_template_mean": statistics.fmean(r["auc_template"] for r in present),
            "n_seeds": len(present),
            "per_seed_published": [None if r is None else r["published"] for r in rows],
            "per_seed_floored": [None if r is None else r["floored"] for r in rows],
        }
    return out


def analyse(doc: dict, pre, seeds: list[int], floor: float) -> dict:
    per_tag = per_tag_skills(doc, floor)
    arms = by_arm(pre, per_tag, seeds)
    if not arms:
        raise SystemExit("no H5 arms present in the artefact")

    threshold = float(pre.hypothesis("H5")["threshold"])
    max_pub = max(a["published"] for a in arms.values())
    max_flo = max(a["floored"] for a in arms.values())
    # restrict the denominator distribution to the H5 arm set, matching the verdict:
    # over *all* 50 tags the median is exactly 0.5000, because `mean` and
    # `centre_gaussian` return 0.5 by construction and are not scored by H5.
    h5_tags = {arm_tag(pre.arm(n)["spec"], s) for n in pre.arm_set("H5") for s in seeds}
    dens = sorted(r["auc_template"] for t, r in per_tag.items() if t in h5_tags)

    return {
        "estimand": "prior_normalised_skill_floored",
        "declared_in_prereg": False,
        "role": "sensitivity_outside_confirmatory_family",
        "note": (
            "Not a pre-registered analysis. The confirmatory H5 verdict is "
            "unchanged and is scored by scripts/prereg_verdict.py against the "
            "unfloored estimand declared in configs/prereg/isbi2027.yaml. This "
            "file exists because the declared denominator sits below chance for "
            "most learned-attention dumps, which makes the published estimand "
            "easier to pass; flooring it is the conservative direction."),
        "floor": floor,
        "threshold_read_from_prereg": threshold,
        "aggregation": "floor per tag, mean over seeds, max over arm_set(H5)",
        "denominator": "attention:inplane (reference_baselines.fitted_template)",
        "denominator_distribution": {
            "n_tags": len(dens),
            "median": statistics.median(dens) if dens else None,
            "min": dens[0] if dens else None,
            "max": dens[-1] if dens else None,
            "n_below_floor": sum(1 for d in dens if d < floor),
        },
        "per_arm": arms,
        "max_published": max_pub,
        "max_floored": max_flo,
        "arms_over_threshold_floored": sorted(
            k for k, v in arms.items() if v["floored"] > threshold),
        "verdict_would_flip": bool(max_flo > threshold),
        "per_tag": per_tag,
    }


def report(res: dict, condition: str) -> None:
    d = res["denominator_distribution"]
    print(f"\n== {condition} ==", flush=True)
    print(f"   denominator over {d['n_tags']} H5 tags: median {d['median']:.4f}, "
          f"{d['n_below_floor']}/{d['n_tags']} below {res['floor']}", flush=True)
    for arm, v in sorted(res["per_arm"].items(), key=lambda kv: -kv[1]["published"]):
        print(f"      {arm:28s} published={v['published']:+.4f}  "
              f"floored={v['floored']:+.4f}  denom={v['auc_template_mean']:.4f}",
              flush=True)
    t = res["threshold_read_from_prereg"]
    print(f"   max published {res['max_published']:+.4f} -> "
          f"max floored {res['max_floored']:+.4f}  (bar {t}) -> "
          f"{'VERDICT WOULD FLIP' if res['verdict_would_flip'] else 'H5 PASS survives'}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="a nulls directory holding template_family.json")
    ap.add_argument("--out", required=True,
                    help="explicit; this script has no default output path on "
                         "purpose, so a forgotten flag cannot overwrite an artefact")
    ap.add_argument("--condition", default=None)
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--role", choices=["exploratory", "confirmatory"],
                    default="exploratory")
    args = ap.parse_args()

    src = Path(args.dir) / "template_family.json"
    doc = json.loads(src.read_text())
    pre = prereg.load()

    res = analyse(doc, pre, args.seeds, args.floor)
    condition = args.condition or Path(args.dir).name
    res.update({"analysis_role": args.role, "condition": condition,
                "source": str(src),
                "source_prereg_hash": doc.get("prereg", {}).get("prereg_hash")})
    report(res, condition)

    payload = pre.stamp(res)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}  ({args.role}, prereg {payload['prereg']['prereg_hash']})",
          flush=True)


if __name__ == "__main__":
    main()
