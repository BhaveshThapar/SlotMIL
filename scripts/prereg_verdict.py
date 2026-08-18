#!/usr/bin/env python
"""Score the ten pre-registered hypotheses from stamped artefacts. Blinded.

``scripts/final_verdict.py`` is a discovery-era table over ``runs/control_*.npz``
with three conditions hardcoded. It is not this. Until this file existed, exactly
one of the ten declared falsifiers -- H10's, through ``estimands.h10_outcome`` --
was expressed anywhere a machine could read, so the other nine were a threshold in
a YAML file, a number in an artefact, and a person to compare them. A
pre-registration scored that way keeps the paperwork and loses the point: the
comparison happens after the numbers are visible, and no record survives of the
rule that was applied.

What this reads, and what it refuses:

* **Only stamped artefact JSON**, never a ``.npz``. Every number here was written
  by a driver that recorded its own provenance; recomputing one would create a
  second value under one name.
* **Every input is placed on the amendment chain first**, through
  ``prereg.amendment_chain`` and ``prereg.classify_hash``. A ``superseded`` or
  ``unknown`` stamp makes every hypothesis reading that artefact **VOID**, not
  failed -- the config the number was computed under is not the config the
  threshold comes from.
* **``analysis_role`` must be ``confirmatory``.** Pointed at ``runs/nulls``, whose
  dumps are from the discovery split, this refuses to score rather than producing a
  table that looks confirmatory. That is worth being able to demonstrate, so it is
  a real code path rather than a warning.
* **Thresholds come from ``configs/prereg/isbi2027.yaml``**, never from the
  artefact. Some drivers print their own copy of a bound (``probe_gate.py`` carries
  0.50, ``h8_in_lung.py`` carries 0.65); those copies are cross-checked against the
  config and a disagreement is VOID, because one of the two is then stale.
* **Arms stay blinded.** Names are printed as ``ARM-XXXXXX`` from
  ``prereg.blind_key()`` and there is no ``--unblind`` flag. Revealing the mapping
  is a deliberate, dated act through ``scripts/prereg_unblind.py``, which appends to
  ``AMENDMENTS.md``. Anyone with filesystem access can read the key directly and
  this does not stop them -- ``PREREGISTRATION.md`` says as much -- but running the
  scorer must not be the thing that reveals it. Reference baselines are never
  blinded; the estimands need to know which one is the template.

Two units are taken rather than read, because the config does not state them, and
both are recorded in the output:

* **H1 and H4 declare no unit over seeds.** H5 had the identical hole and ruled it
  as the mean over seeds, arguing that "no ARM's skill" is the statement's own noun
  and that a per-seed maximum takes five draws at the threshold instead of one.
  That argument transfers unchanged, so per-arm values here are means over that
  arm's seeds, with every per-seed value reported beside them.
* **H9 declares no aggregation over arms and seeds.** Scored on every matched
  ``(arm, seed)`` pair, which is the strictest available reading -- one pair failing
  to order fails the hypothesis -- and flagged as taken rather than declared.

    python scripts/prereg_verdict.py \\
        --dir runs/nulls_nodule_present_confirmatory \\
        --untrained-floor runs/untrained_floor.json \\
        --out runs/nulls_nodule_present_confirmatory/prereg_verdict.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from slotmil import prereg
from slotmil.eval.verdict import (
    FAIL,
    NOT_RUN,
    PASS,
    REPORTED,
    VOID,
    arm_tag,
    h1_verdict,
    h2_verdict,
    h3_verdict,
    h4_verdict,
    h5_verdict,
    h6_verdict,
    h7_verdict,
    h8_verdict,
    h9_verdict,
    h10_verdict,
)


# ------------------------------------------------------------------ provenance
class Artefacts:
    """Loaded artefacts plus the reason each one may not be scored.

    The gate runs once per file, at load, rather than per hypothesis. A hypothesis
    then either has its inputs or has a reason, and cannot accidentally be scored
    from a file that failed the gate three hypotheses ago.
    """

    # An input can be unavailable two ways, and they do not mean the same thing.
    # ABSENT -- the file was never written, so the hypothesis has no evidence and is
    # NOT_RUN. UNTRUSTED -- the file exists but its number was computed under a
    # different config, or outside the confirmatory scope, or from a tree that is not
    # in git. That is a hypothesis whose falsifier cannot be trusted, which is VOID.
    # Collapsing the two would let a superseded artefact read as "not yet run" and
    # quietly invite a re-run that is not what is needed.
    ABSENT = "absent"
    UNTRUSTED = "untrusted"

    def __init__(self, pre, current, chain, require_role="confirmatory"):
        self.pre = pre
        self.current = current
        self.chain = chain
        self.require_role = require_role
        self.docs: dict[str, dict] = {}
        self.problems: dict[str, str] = {}
        self.kinds: dict[str, str] = {}
        self.lineage: dict[str, str] = {}

    def _reject(self, key: str, kind: str, reason: str) -> None:
        self.kinds[key] = kind
        self.problems[key] = reason

    def add(self, key: str, path: str | Path | None) -> None:
        if path is None:
            return self._reject(key, self.ABSENT, "not supplied")
        p = Path(path)
        if not p.exists():
            return self._reject(key, self.ABSENT, f"missing: {p}")
        doc = json.loads(p.read_text())
        stamp = doc.get("prereg", {})
        stamped = stamp.get("prereg_hash")
        if stamped is None:
            return self._reject(key, self.UNTRUSTED,
                                f"{p} carries no prereg stamp")
        lin = prereg.classify_hash(stamped, self.current, self.chain)
        self.lineage[key] = lin.status
        if lin.status != "current":
            return self._reject(key, self.UNTRUSTED,
                                f"{p} is stamped {stamped}, which is {lin.status} "
                                f"against the current {self.current}; the threshold "
                                "and the number come from two different configs")
        role = doc.get("analysis_role")
        if role != self.require_role:
            return self._reject(key, self.UNTRUSTED,
                                f"{p} has analysis_role={role!r}, "
                                f"not {self.require_role!r}")
        if stamp.get("git_dirty"):
            return self._reject(key, self.UNTRUSTED,
                                f"{p} was stamped from a dirty tree (commit "
                                f"{stamp.get('git_commit')}), so the code that "
                                "produced it is not recoverable from git")
        self.docs[key] = doc
        return None

    def need(self, *keys: str) -> tuple[dict | None, dict | None]:
        """All of ``keys``, or the first input that is unusable and why."""
        for k in keys:
            if k not in self.docs:
                return None, {
                    "key": k,
                    "kind": self.kinds.get(k, self.UNTRUSTED),
                    "reason": f"{k}: {self.problems.get(k, 'unavailable')}",
                }
        return {k: self.docs[k] for k in keys}, None

    def group(self, prefix: str) -> tuple[dict[str, dict], dict | None]:
        """Every input whose key starts with ``prefix``, or the first unusable one.

        H3 is the case: two untrained floors exist, one per pooling, and both are
        declared reference floors. An untrusted member of the group voids the
        hypothesis rather than being skipped -- skipping it would let the group
        shrink to whichever floor happened to be clean.
        """
        keys = sorted(k for k in set(self.docs) | set(self.problems)
                      if k.startswith(prefix))
        untrusted = [k for k in keys if self.kinds.get(k) == self.UNTRUSTED]
        if untrusted:
            k = untrusted[0]
            return {}, {"key": k, "kind": self.UNTRUSTED,
                        "reason": f"{k}: {self.problems[k]}"}
        present = {k: self.docs[k] for k in keys if k in self.docs}
        if not present:
            return {}, {"key": prefix, "kind": self.ABSENT,
                        "reason": f"{prefix}: no artefact supplied"}
        return present, None


def _void(hid: str, reason: str) -> dict:
    return {"id": hid, "outcome": VOID, "reason": reason}


def _unavailable(hid: str, problem: dict) -> dict:
    """NOT_RUN for a missing input, VOID for one that cannot be trusted."""
    outcome = NOT_RUN if problem["kind"] == Artefacts.ABSENT else VOID
    return {"id": hid, "outcome": outcome, "reason": problem["reason"],
            "unusable_input": problem["key"]}


# -------------------------------------------------------------------- helpers
def _rows(doc, key="results"):
    return {r["tag"]: r for r in doc.get(key, [])}


def _mean(vals):
    finite = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def _by_arm(pre, hid, per_tag, seeds):
    """``{arm: (mean_over_seeds, [per-seed])}`` over the hypothesis's arm set.

    Tags are built from the arm *spec*, not the name: ``slot:div=0.5`` has name
    ``slot_div0.5`` and dump directory ``slot_div=0.5``, so a tag from the name
    looks for a file that is not there.
    """
    out = {}
    for name in pre.arm_set(hid):
        spec = pre.arm(name)["spec"]
        vals = [per_tag.get(arm_tag(spec, s)) for s in seeds]
        present = [v for v in vals if v is not None]
        if present:
            out[name] = (_mean(present), vals)
    return out


def _gap(row):
    """``|flat - within_slice|``, H1's statistic, from one axis_gate row."""
    f, w = row.get("flat_auc", {}), row.get("within_slice_auc", {})
    if f.get("mean") is None or w.get("mean") is None:
        return None
    return abs(f["mean"] - w["mean"])


def _scorer_gaps(doc, scorer):
    """``{tag: gap}`` for one named scorer across every template_family row."""
    out = {}
    for r in doc.get("results", []):
        row = next((s for s in r.get("scorers", []) if s["scorer"] == scorer), None)
        g = None if row is None else _gap(row)
        if g is not None:
            out[r["tag"]] = g
    return out


# ----------------------------------------------------------------- hypotheses
def score_h1(pre, art, seeds):
    docs, why = art.need("axis_gate", "template_family")
    if why:
        return _unavailable("H1", why)
    h = pre.hypothesis("H1")
    rows = _rows(docs["axis_gate"])
    per_tag = {t: _gap(r) for t, r in rows.items()}
    by = _by_arm(pre, "H1", per_tag, seeds)
    gaps = {a: m for a, (m, _) in by.items() if m is not None}
    per_seed = {a: v for a, (_, v) in by.items()}

    # Controls: the same |flat - within| statistic on references that carry axial
    # information by construction. masks:* come from the template family, the centre
    # prior from whichever file holds it.
    controls: dict[str, float] = {}
    for member in h["positive_controls"]["members"]:
        if member.startswith("masks:"):
            vals = _scorer_gaps(docs["template_family"], member)
        elif member in rows:
            vals = {member: per_tag.get(member)}
        else:
            vals = _scorer_gaps(docs["template_family"], member)
        m = _mean(list(vals.values()))
        if m is not None:
            controls[member] = m
    absent = [m for m in h["positive_controls"]["members"] if m not in controls]
    if absent:
        return _void("H1", f"positive control(s) absent from the artefacts: {absent}")

    tf = h["tie_floor_check"]
    tie_tags = [arm_tag(pre.arm(tf["arm"])["spec"], s) for s in seeds]
    tie = _mean([per_tag.get(t) for t in tie_tags])

    v = h1_verdict(gaps, h["threshold"], controls, tie, tf["expected_gap"])
    v["per_seed"] = per_seed
    v["unit_over_seeds"] = "mean_over_seeds (undeclared for H1; see the module docstring)"
    return v


def score_h2(pre, art, seeds):
    docs, why = art.need("axis_gate")
    if why:
        return _unavailable("H2", why)
    rows = _rows(docs["axis_gate"])
    prior = rows.get("centre_prior", {}).get("slice_auc", {}).get("mean")
    per_tag = {t: r.get("slice_auc", {}).get("mean") for t, r in rows.items()}
    by = _by_arm(pre, "H2", per_tag, seeds)
    v = h2_verdict(prior, {a: m for a, (m, _) in by.items()})
    v["per_seed"] = {a: s for a, (_, s) in by.items()}
    v["excluded_arms"] = pre.hypothesis("H2").get("excluded_arms")
    return v


def score_h3(pre, art):
    docs, why = art.group("untrained_floor")
    if why:
        return _unavailable("H3", why)
    fleet = next((b for b in pre.get("reference_baselines")
                  if b["name"] == "untrained_fleet"), {})
    required = int(fleet.get("n_inits", 30))

    # One floor per pooling, and the verdict takes the WEAKEST of them. H3 says 0.5
    # is not the floor; a floor that clears on one architecture and not another has
    # not established that, and picking the architecture that clears after seeing
    # both is the best-of-k error the denominator ruling already refuses.
    floors = {}
    for doc in docs.values():
        f = doc.get("floor", {})
        floors[doc.get("pooling", "?")] = {"p95": f.get("p95"),
                                           "n_inits": int(f.get("n_inits", 0)),
                                           "median": f.get("median"),
                                           "p5": f.get("p5")}
    usable = [v for v in floors.values() if v["p95"] is not None]
    if not usable:
        return _void("H3", "no floor artefact carries a 95th percentile")
    weakest = min(usable, key=lambda v: v["p95"])
    v = h3_verdict(weakest["p95"], pre.hypothesis("H3")["threshold"],
                   min(f["n_inits"] for f in usable), required)
    v["per_pooling"] = floors
    v["aggregation"] = "weakest_floor_over_poolings"
    return v


def score_h4(pre, art, seeds):
    docs, why = art.need("h4_cross_patient")
    if why:
        return _unavailable("H4", why)
    per_tag = {t: r.get("patient_specific_skill", {}).get("mean")
               for t, r in _rows(docs["h4_cross_patient"]).items()}
    by = _by_arm(pre, "H4", per_tag, seeds)
    thr = pre.hypothesis("H4")["threshold"]
    v = h4_verdict({a: m for a, (m, _) in by.items()}, thr["per_arm"], thr["median"])
    v["per_seed"] = {a: s for a, (_, s) in by.items()}
    v["unit_over_seeds"] = "mean_over_seeds (undeclared for H4; see the module docstring)"
    return v


def score_h5(pre, art, seeds):
    docs, why = art.need("template_family")
    if why:
        return _unavailable("H5", why)
    per_tag = {t: r.get("prior_normalised_skill", {}).get("trained")
               for t, r in _rows(docs["template_family"]).items()}
    by = _by_arm(pre, "H5", per_tag, seeds)
    return h5_verdict({a: m for a, (m, _) in by.items()},
                      pre.hypothesis("H5")["threshold"],
                      {a: [x for x in s if x is not None] for a, (_, s) in by.items()})


def score_h6(pre, art, seeds):
    docs, why = art.need("axis_gate", "h4_cross_patient")
    if why:
        return _unavailable("H6", why)
    cmp_ = pre.hypothesis("H6")["base_arm_comparison"]
    slice_rows = _rows(docs["axis_gate"])
    skill_rows = _rows(docs["h4_cross_patient"])

    def over_seeds(spec, rows, get):
        return _mean([get(rows.get(arm_tag(spec, s), {})) for s in seeds])

    arm_spec = cmp_["arm"]
    base_spec = pre.arm(cmp_["base"])["spec"]

    def sl(r):
        return r.get("slice_auc", {}).get("mean")

    def sk(r):
        return r.get("patient_specific_skill", {}).get("mean")

    v = h6_verdict(over_seeds(base_spec, slice_rows, sl),
                   over_seeds(arm_spec, slice_rows, sl),
                   over_seeds(base_spec, skill_rows, sk),
                   over_seeds(arm_spec, skill_rows, sk),
                   pre.hypothesis("H6")["threshold"])
    v["comparison"] = cmp_
    return v


def score_h7(pre, art):
    docs, why = art.need("probe_gate", "h7_content_free")
    if why:
        return _unavailable("H7", why)
    thr = pre.hypothesis("H7")["threshold"]
    probe_doc = docs["probe_gate"]
    stated = probe_doc.get("h7_probe_half", {}).get("threshold")
    if stated is not None and stated != thr["probe"]:
        return _void("H7", f"probe_gate.json carries threshold {stated} but the "
                           f"config declares {thr['probe']}; one is stale")
    probe = probe_doc.get("prior_normalised_skill", {}).get("skill")

    # "every content-free baseline's is below 0.05" is a statement about the set, so
    # each member is summarised by its WORST value over the arms it accompanies, not
    # by an average that a good arm could hide a bad one inside. Declared since the
    # 2026-08-17 amendment as `content_free_unit`; it lived only in this comment
    # before that, which is why it was the one chosen unit absent from
    # `undeclared_units_taken` below.
    h7 = pre.hypothesis("H7")
    per_tag: dict[str, list[float]] = {m: [] for m in h7["content_free_set"]}
    members: dict[str, float | None] = {m: None for m in h7["content_free_set"]}
    for r in docs["h7_content_free"].get("results", []):
        for name, m in r.get("members", {}).items():
            if m.get("skill") is None:
                continue
            cur = members.get(name)
            members[name] = m["skill"] if cur is None else max(cur, m["skill"])
            if name in per_tag:
                per_tag[name].append(m["skill"])

    v = h7_verdict(probe, thr["probe"], members, thr["content_free"])

    # The verdict attaches to the max, and the other two aggregations are reported
    # beside it for the same reason H8 publishes the number its threshold does not
    # attach to: an exclusion a reader cannot check is indistinguishable from a
    # number that was never computed. This one is worth checking -- the choice
    # between max and the others decided H7's outcome in all three conditions on
    # the single-draw artefacts.
    v["aggregation"] = h7.get("content_free_unit", "max_over_tags")
    v["aggregation_sensitivity"] = {
        name: ({"n_tags": len(vals),
                "mean": float(np.mean(vals)), "median": float(np.median(vals)),
                "max": float(np.max(vals)), "min": float(np.min(vals))}
               if vals else {"n_tags": 0})
        for name, vals in per_tag.items()
    }
    v["aggregation_sensitivity_note"] = (
        "Reported, not applied. The verdict uses the declared unit above. These are "
        "the same per-tag values under the two aggregations the unit was chosen "
        "against, so a reader can see what the choice was worth rather than take it "
        "on trust."
    )
    # How many draws stand behind each per-tag value, carried up from the driver so
    # the verdict is readable without opening the artefact it was computed from.
    v["content_free_draws"] = docs["h7_content_free"].get("content_free_draws")
    v["draw_spread"] = {
        name: max(
            (r["members"][name]["draw_distribution"]["sd"]
             for r in docs["h7_content_free"].get("results", [])
             if r.get("members", {}).get(name, {}).get("draw_distribution", {}).get("sd")
             is not None),
            default=None)
        for name in h7["content_free_set"]
    }
    v["draw_spread_note"] = (
        "Worst per-tag draw-to-draw standard deviation per member. Before the "
        "2026-08-17 amendment every tag shared one realisation, so this was not "
        "estimable and a maximum over tags could not distinguish a member above the "
        "threshold from a member whose single draw was."
    )
    return v


def score_h8(pre, art, seeds):
    docs, why = art.need("h8_in_lung")
    if why:
        return _unavailable("H8", why)
    h = pre.hypothesis("H8")
    rows = _rows(docs["h8_in_lung"])
    stated = next((r.get("h8", {}).get("threshold") for r in rows.values()), None)
    if stated is not None and stated != h["threshold"]:
        return _void("H8", f"h8_in_lung.json carries threshold {stated} but the "
                           f"config declares {h['threshold']}; one is stale")
    per_tag = {t: r.get("estimands", {}).get("auc", {}).get("in_lung", {}).get("mean")
               for t, r in rows.items()}
    by = _by_arm(pre, "H8", per_tag, seeds)
    v = h8_verdict(_mean([m for m, _ in by.values()]), h["threshold"])
    v["per_arm"] = {a: m for a, (m, _) in by.items()}
    v["per_seed"] = {a: s for a, (_, s) in by.items()}
    v["reported_beside_the_threshold"] = {
        t: r.get("estimands", {}).get("stratified_auc", {}).get("in_lung", {}).get("mean")
        for t, r in rows.items()}
    return v


def _template_ci(doc):
    """``{tag: (fitted-template flat AUC ci, trained skill)}`` from a template family."""
    out = {}
    for r in doc.get("results", []):
        row = next((s for s in r.get("scorers", [])
                    if s.get("pre_registered_denominator")), None)
        if row is not None:
            out[r["tag"]] = (row["flat_auc"],
                             r.get("prior_normalised_skill", {}).get("trained"))
    return out


def _h9_power(tag, mos_ci, lidc_ci):
    """What separation this pair's intervals would have needed to order.

    H9 is interval-scored and its falsifier counts *indistinguishable* as a
    failure, so a FAIL can mean two different things: the orderings are really
    absent, or the intervals were too wide for any ordering to show. Those are
    not the same finding and the paper must not report the second as the first.

    The gate is ``mosmed.hi < lidc.lo``, so the point estimates have to be
    separated by at least the two facing half-widths summed. Reporting the
    required separation beside the observed one turns "FAIL" into a number a
    reader can attribute. MosMed contributes 22 test clusters against LIDC's
    150, and the config already flags 38 clusters as wide enough that most
    outcomes land indistinguishable by default -- so this is an expected
    hazard being measured, not a surprise being explained away.
    """
    if any(c is None or c.get("lo") is None or c.get("hi") is None
           for c in (mos_ci, lidc_ci)):
        return {"tag": tag, "computable": False}
    observed = lidc_ci["mean"] - mos_ci["mean"]
    required = (lidc_ci["mean"] - lidc_ci["lo"]) + (mos_ci["hi"] - mos_ci["mean"])
    return {
        "tag": tag, "computable": True,
        "mosmed": {"mean": mos_ci["mean"], "lo": mos_ci["lo"], "hi": mos_ci["hi"],
                   "width": mos_ci["hi"] - mos_ci["lo"],
                   "n_clusters": mos_ci.get("n_clusters")},
        "lidc": {"mean": lidc_ci["mean"], "lo": lidc_ci["lo"], "hi": lidc_ci["hi"],
                 "width": lidc_ci["hi"] - lidc_ci["lo"],
                 "n_clusters": lidc_ci.get("n_clusters")},
        "observed_separation": observed,
        "required_separation": required,
        "shortfall": required - observed,
        "orders": bool(mos_ci["hi"] < lidc_ci["lo"]),
    }


def score_h9(pre, art):
    docs, why = art.need("template_family", "mosmed_template_family")
    if why:
        return _unavailable("H9", why)
    lidc = _template_ci(docs["template_family"])
    mos = _template_ci(docs["mosmed_template_family"])
    shared = sorted(set(lidc) & set(mos))
    if not shared:
        return {"id": "H9", "outcome": NOT_RUN,
                "reason": "no (arm, seed) tag appears in both template families"}

    per_pair = [h9_verdict(mos[t][0], lidc[t][0], mos[t][1], lidc[t][1])
                for t in shared]
    failed = [t for t, v in zip(shared, per_pair) if v["outcome"] != PASS]
    outcome = PASS if not failed else FAIL
    if any(v["outcome"] == VOID for v in per_pair):
        outcome = VOID
    power = [_h9_power(t, mos[t][0], lidc[t][0]) for t in shared]
    usable = [p for p in power if p["computable"]]
    # Reported whichever way H9 lands. A PASS with a tiny margin and a FAIL that
    # was arithmetically unreachable are both things a reader should be able to
    # see without re-deriving them from the per-tag intervals.
    summary = {"n_tags": len(power), "n_computable": len(usable)}
    if usable:
        short = [p["shortfall"] for p in usable]
        summary.update({
            "median_mosmed_ci_width": float(np.median([p["mosmed"]["width"]
                                                       for p in usable])),
            "median_lidc_ci_width": float(np.median([p["lidc"]["width"]
                                                     for p in usable])),
            "median_required_separation": float(np.median([p["required_separation"]
                                                           for p in usable])),
            "median_observed_separation": float(np.median([p["observed_separation"]
                                                           for p in usable])),
            "median_shortfall": float(np.median(short)),
            "n_ordering": sum(1 for p in usable if p["orders"]),
            "mosmed_n_clusters": usable[0]["mosmed"]["n_clusters"],
            "lidc_n_clusters": usable[0]["lidc"]["n_clusters"],
        })

    return {
        "id": "H9", "outcome": outcome,
        "reason": (f"{len(shared) - len(failed)}/{len(shared)} matched tags order "
                   "MosMed's template AUC below LIDC's with higher skill"),
        "matched_tags": shared, "tags_not_ordering": failed,
        "per_tag": dict(zip(shared, per_pair)),
        "aggregation": "every_matched_tag",
        "aggregation_note":
            "H9 declares no aggregation over arms and seeds. Taken as the strictest "
            "available reading -- one matched pair failing to order fails the "
            "hypothesis -- and flagged as taken rather than declared.",
        "power": summary,
        "power_per_tag": {p["tag"]: p for p in power},
        "power_note":
            "H9's falsifier counts 'indistinguishable' as a failure, so a FAIL can "
            "mean the ordering is absent OR that the intervals were too wide for any "
            "ordering to show. `required_separation` is what the point estimates "
            "would have had to differ by for the intervals to clear -- the two facing "
            "half-widths summed -- and `observed_separation` is what they do differ "
            "by. A FAIL whose median shortfall is positive across nearly every tag is "
            "a statement about 22 MosMed test clusters, not about stereotypy. "
            "Reported whichever way the hypothesis lands, and it changes no verdict.",
    }


def score_h10(pre, art, seeds):
    docs, why = art.need("template_family")
    if why:
        return _unavailable("H10", why)
    h = pre.hypothesis("H10")
    rows = _rows(docs["template_family"])
    tags = [arm_tag(h["arm"], s) for s in seeds]
    outcomes = [rows.get(t, {}).get("h10", {}).get("outcome") for t in tags]
    v = h10_verdict(outcomes)
    v["arm"] = h["arm"]
    v["member"] = h["member"]
    v["tags"] = tags
    return v


# --------------------------------------------------------------------- output
def blind_substitutions(pre, key, seeds):
    """Every string that identifies an arm, mapped to its blind code.

    An arm reaches this file under three spellings and blinding one of them is not
    blinding: the ``name`` (``slot_div0.5``), the ``spec`` (``slot:div=0.5``, which
    H6's ``base_arm_comparison`` and H10's ``arm`` are written as), and the dump
    ``tag`` (``slot_div=0.5_seed3``, which every artefact keys its rows by). A tag
    carries the spec verbatim, so a verdict table keyed by tag is unblinded no
    matter what the code column says.

    Arms declared ``blind: false`` map to themselves. ``centre_gaussian`` is one:
    the estimands need to know which arm is a fixed geometric rule, and
    ``PREREGISTRATION.md`` blinds model arms rather than references.
    """
    subs: dict[str, str] = {}
    for arm in pre.arms(status="implemented"):
        code = key.codes.get(arm["name"], arm["name"])
        subs[arm["name"]] = code
        subs[arm["spec"]] = code
        for s in seeds:
            subs[arm_tag(arm["spec"], s)] = f"{code}_seed{s}"
    return subs


def _blind(subs, obj):
    """Recursively substitute arm names, specs and tags, in keys and in values.

    Reason strings are *not* rewritten by pattern -- they are written without arm
    names in the first place (see ``h4_verdict``), because substituting inside prose
    would silently depend on formatting.
    """
    if isinstance(obj, dict):
        return {(subs.get(k, k) if isinstance(k, str) else k): _blind(subs, v)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_blind(subs, x) for x in obj]
    if isinstance(obj, str):
        return subs.get(obj, obj)
    return obj


SYMBOL = {PASS: "PASS", FAIL: "FAIL", VOID: "VOID", NOT_RUN: "----",
          REPORTED: "RPT "}


def infer_condition(pre, d: Path) -> str | None:
    """The declared condition whose name appears in the artefact directory.

    Longest match wins so that ``balanced_presence`` is not read as a prefix of
    something shorter. Returns None rather than guessing when nothing matches.
    """
    names = sorted((c["name"] for c in pre.get("conditions")), key=len, reverse=True)
    return next((n for n in names if n in d.name), None)


def report(verdicts, condition, carries):
    print("\n" + "=" * 78, flush=True)
    print("PRE-REGISTERED VERDICTS  (arms blinded; "
          "unblind through scripts/prereg_unblind.py)", flush=True)
    print(f"condition: {condition}", flush=True)
    if not carries:
        # Exactly one condition carries the family, ruled 2026-08-15 before any
        # confirmatory number existed. A table computed on another condition is a
        # consistency check; presenting it as the pre-registered verdict would be
        # the best-of-three selection that ruling exists to prevent.
        print("NOT the hypothesis-family condition -- this table is a consistency",
              flush=True)
        print("check, not the pre-registered verdict. See "
              "conditions[].carries_hypothesis_family.", flush=True)
    print("=" * 78, flush=True)
    for v in verdicts:
        print(f"  {v['id']:4s} {SYMBOL.get(v['outcome'], v['outcome']):5s} "
              f"{v['reason']}", flush=True)
    tally: dict[str, int] = {}
    for v in verdicts:
        tally[v["outcome"]] = tally.get(v["outcome"], 0) + 1
    print("-" * 78, flush=True)
    print("  " + "  ".join(f"{k}={n}" for k, n in sorted(tally.items())), flush=True)
    if tally.get(VOID):
        print("  VOID is not FAIL and not PASS: the falsifier could not be trusted.",
              flush=True)
    if tally.get(NOT_RUN):
        print("  NOT_RUN hypotheses have no artefact. They are not satisfied.",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/nulls",
                    help="directory holding axis_gate.json, template_family.json, "
                         "h4_cross_patient.json, h7_content_free.json, "
                         "h8_in_lung.json and probe_gate.json")
    ap.add_argument("--axis-gate", default=None)
    ap.add_argument("--template-family", default=None)
    ap.add_argument("--h4-cross-patient", default=None)
    ap.add_argument("--h7-content-free", default=None)
    ap.add_argument("--h8-in-lung", default=None)
    ap.add_argument("--probe-gate", default=None)
    # The rehash4 pair is the current, clean-tree stamping of the two floors. The
    # bare runs/untrained_floor.json is stamped 20bdd93b781d950d from a dirty tree
    # and is two amendments behind; defaulting to it would void H3 on every run.
    ap.add_argument("--untrained-floor", nargs="*",
                    default=["runs/untrained_floor_rehash4.json",
                             "runs/untrained_floor_mh_rehash4.json"],
                    help="one floor artefact per pooling; the verdict takes the "
                         "weakest 95th percentile among them")
    ap.add_argument("--mosmed-template-family", default=None,
                    help="MosMed's template_family.json; H9 is NOT_RUN without it")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="default: protocol.seeds")
    ap.add_argument("--condition", default=None,
                    help="default: inferred from --dir. Only the condition named by "
                         "hypothesis_family_condition carries the pre-registered "
                         "verdict; the others are labelled consistency checks")
    ap.add_argument("--out", default=None,
                    help="default: <dir>/prereg_verdict.json")
    args = ap.parse_args()

    pre = prereg.load()
    chain = prereg.amendment_chain()
    seeds = args.seeds if args.seeds else pre.get("protocol.seeds")
    d = Path(args.dir)

    art = Artefacts(pre, pre.hash, chain)
    for key, override, default in (
        ("axis_gate", args.axis_gate, d / "axis_gate.json"),
        ("template_family", args.template_family, d / "template_family.json"),
        ("h4_cross_patient", args.h4_cross_patient, d / "h4_cross_patient.json"),
        ("h7_content_free", args.h7_content_free, d / "h7_content_free.json"),
        ("h8_in_lung", args.h8_in_lung, d / "h8_in_lung.json"),
        ("probe_gate", args.probe_gate, d / "probe_gate.json"),
        ("mosmed_template_family", args.mosmed_template_family, None),
    ):
        art.add(key, override if override is not None else default)

    # One key per floor, so an untrusted floor voids H3 instead of being dropped
    # from a group that then reads as complete.
    for i, path in enumerate(args.untrained_floor or [None]):
        art.add(f"untrained_floor[{i}]", path)

    print(f"[verdict] prereg {pre.hash}, {len(chain)} amendment entries", flush=True)
    for key in sorted(set(art.docs) | set(art.problems)):
        state = (f"ok ({art.lineage.get(key, '?')})" if key in art.docs
                 else art.problems[key])
        print(f"   {key:24s} {state}", flush=True)

    verdicts = [
        score_h1(pre, art, seeds),
        score_h2(pre, art, seeds),
        score_h3(pre, art),
        score_h4(pre, art, seeds),
        score_h5(pre, art, seeds),
        score_h6(pre, art, seeds),
        score_h7(pre, art),
        score_h8(pre, art, seeds),
        score_h9(pre, art),
        score_h10(pre, art, seeds),
    ]

    condition = args.condition or infer_condition(pre, d)
    carries = condition == pre.get("hypothesis_family_condition")

    subs = blind_substitutions(pre, pre.blind_key(), seeds)
    blinded = [_blind(subs, v) for v in verdicts]
    report(blinded, condition, carries)

    payload = pre.stamp({
        "analysis_role": "confirmatory",
        "condition_dir": str(d),
        "condition": condition,
        "carries_hypothesis_family": carries,
        "carries_hypothesis_family_note":
            "hypothesis_family_condition names exactly one condition, ruled "
            "2026-08-15 before any confirmatory number existed. A verdict table "
            "computed on any other condition is a consistency check and must not be "
            "reported as the pre-registered verdict -- choosing which condition "
            "leads after seeing three tables is the best-of-k error the template "
            "denominator ruling already refuses.",
        "seeds": list(seeds),
        "blinded": True,
        "blinding_note":
            "Arm names are the codes from blinding.key_path. Unblinding is a "
            "separate dated act through scripts/prereg_unblind.py, which appends to "
            "AMENDMENTS.md. Procedural, not cryptographic -- the key is on the same "
            "disk, as PREREGISTRATION.md states.",
        "inputs": {k: {"lineage": art.lineage.get(k),
                       "kind": art.kinds.get(k),
                       "problem": art.problems.get(k)}
                   for k in sorted(set(art.docs) | set(art.problems))},
        # H7 is deliberately absent: its unit was the one chosen unit documented
        # solely in source comments -- neither declared in the config nor recorded
        # here as taken -- and the 2026-08-17 amendment declared it as
        # `content_free_unit` rather than adding it to this block. A unit that the
        # config states is not a unit this file took.
        "undeclared_units_taken": {
            "H1": "mean_over_seeds", "H4": "mean_over_seeds",
            "H9": "every_matched_tag",
        },
        "verdicts": blinded,
    })
    out = Path(args.out) if args.out else d / "prereg_verdict.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}  (prereg {payload['prereg']['prereg_hash']})", flush=True)


if __name__ == "__main__":
    main()
