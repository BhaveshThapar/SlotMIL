"""The ten pre-registered falsifiers, as code rather than as arithmetic done by eye.

``PREREGISTRATION.md`` states a numeric way to fail for every hypothesis, and until
this module existed exactly one of them -- H10, through
:func:`slotmil.eval.estimands.h10_outcome` -- was expressed anywhere a machine
could read. Everything else was a threshold in a YAML file, a number in a stamped
artefact, and a person to compare them. That gap is not cosmetic. A
pre-registration whose verdicts are read off by hand gives up most of what it was
for: the comparison happens *after* the numbers are visible, by someone who has
seen them, with no record of the rule that was applied.

So each falsifier below takes already-extracted numbers and returns a verdict.
No file reading, no thresholds of its own -- every bound is passed in by
:mod:`scripts.prereg_verdict`, which reads it from
``configs/prereg/isbi2027.yaml``. Nothing here retypes a pre-registered constant,
because a literal copied into Python is a second source of truth that goes stale
without saying so.

Four outcomes, and the distinctions between them all cost something to lose:

``PASS`` / ``FAIL``
    The falsifier was evaluated and did or did not fire.
``VOID``
    The falsifier could not be *trusted*, as opposed to not being met -- H1's
    positive controls came in under threshold, or the tie floor moved, or an input
    artefact is off the amendment chain. A void hypothesis is not a supported one,
    and reporting it as either PASS or FAIL would be a claim the evidence does not
    carry.
``NOT_RUN``
    No artefact. Distinct from PASS on purpose: a hypothesis with no evidence must
    never read as satisfied, which is the failure mode of every scoring script that
    iterates over the results it happens to find.

``statistics.verdict_basis`` is ``point_estimate``, so the falsifiers here compare
point estimates and the patient-clustered interval is reported beside them without
changing anything. H9 and H10 are the two exceptions, and they are exceptions
because their own text states interval rules -- not because an interval reading was
preferred for them after the fact.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = [
    "PASS",
    "FAIL",
    "VOID",
    "NOT_RUN",
    "REPORTED",
    "arm_tag",
    "majority",
    "h1_verdict",
    "h2_verdict",
    "h3_verdict",
    "h4_verdict",
    "h5_verdict",
    "h6_verdict",
    "h7_verdict",
    "h8_verdict",
    "h9_verdict",
    "h10_verdict",
]

PASS = "PASS"
FAIL = "FAIL"
VOID = "VOID"
NOT_RUN = "NOT_RUN"
REPORTED = "REPORTED"

# H1's tie floor is an exact algebraic claim -- `mean` returns 1/n, so both axes
# tie at 0.5 and the gap is identically zero. The tolerance is for float32 dump
# round-trip, not for a real deviation: anything above this means the tie handling
# changed underneath the estimand, which is what the check is for.
TIE_TOL = 1e-9


def arm_tag(spec: str, seed: int) -> str:
    """Dump tag for one arm spec and seed.

    ``scripts/train_cached.py`` writes run directories as the spec with ``:``
    replaced by ``_``, and ``confirmatory_collect_array.sbatch`` tags dumps
    ``<that>_seed<N>``. Derived here from the *spec* rather than the arm ``name``
    because the two differ where it matters: ``slot:div=0.5`` has name
    ``slot_div0.5`` and directory ``slot_div=0.5``, so building a tag from the name
    silently looks for a dump that does not exist.
    """
    return f"{spec.replace(':', '_')}_seed{seed}"


def majority(n: int) -> int:
    """Smallest count that is a majority of ``n``.

    Spelled out because H1 and H10 both turn on it and "a majority of arms" is the
    kind of phrase that becomes ``> n / 2`` in one place and ``>= n / 2`` in
    another. Over the 7-arm ``learned_attention`` set this is 4, which is the same
    four the amendment's arithmetic identifies over the 9-arm set.
    """
    return n // 2 + 1


def _verdict(hid: str, outcome: str, reason: str, **detail: Any) -> dict:
    return {"id": hid, "outcome": outcome, "reason": reason, **detail}


def h1_verdict(gaps: dict[str, float], threshold: float,
               controls: dict[str, float], tie_gap: float | None,
               tie_expected: float) -> dict:
    """``|flat - within_slice| < threshold`` for every arm; majority failing falsifies.

    Two gates run before the arms are counted, and both produce VOID rather than a
    verdict. The ``positive_controls`` carry axial information by construction and
    therefore *must* exceed the threshold -- if a reference that is known to be
    axially informative comes in looking in-plane, the instrument is broken and the
    arms it measured cannot be read. The ``tie_floor_check`` asserts ``mean``'s gap
    is identically zero; a nonzero value means the tie handling moved, which would
    shift every gap in the table by an unknown amount.
    """
    if tie_gap is None:
        return _verdict("H1", VOID, "tie_floor_check arm has no gap in the artefact",
                        tie_expected=tie_expected)
    if abs(tie_gap - tie_expected) > TIE_TOL:
        return _verdict("H1", VOID,
                        f"tie floor moved: expected {tie_expected}, got {tie_gap:.3e} -- "
                        "the tie handling changed underneath the estimand",
                        tie_gap=tie_gap, tie_expected=tie_expected)

    failed_controls = {k: v for k, v in controls.items() if not v > threshold}
    if not controls:
        return _verdict("H1", VOID, "no positive controls present in the artefact")
    if failed_controls:
        return _verdict("H1", VOID,
                        "positive control(s) did not exceed the threshold, so the "
                        f"harness cannot detect axial content: {failed_controls}",
                        controls=controls, threshold=threshold)

    if not gaps:
        return _verdict("H1", NOT_RUN, "no arms present in the artefact")
    over = sorted(k for k, v in gaps.items() if not v < threshold)
    need = majority(len(gaps))
    outcome = FAIL if len(over) >= need else PASS
    return _verdict(
        "H1", outcome,
        f"{len(over)}/{len(gaps)} arms have |flat - within_slice| >= {threshold}; "
        f"a majority is {need}",
        gaps=gaps, threshold=threshold, arms_over_threshold=over,
        n_arms=len(gaps), majority_needed=need, controls=controls, tie_gap=tie_gap,
    )


def h2_verdict(centre_prior_slice: float | None,
               arm_slice: dict[str, float]) -> dict:
    """The centre prior's slice AUC exceeds every trained arm's.

    ``arm_slice`` must already be restricted to ``H2.arm_set``
    (``learned_attention``): ``centre_gaussian`` ranks slices by ``-|z|`` exactly as
    the reference does and would tie it, and a tie is not "exceeds", so an arm that
    *is* a centre prior would otherwise falsify a hypothesis about centre priors.
    """
    if centre_prior_slice is None:
        return _verdict("H2", NOT_RUN, "no centre_prior row in the artefact")
    if not arm_slice:
        return _verdict("H2", NOT_RUN, "no arms present in the artefact")
    beat = sorted(k for k, v in arm_slice.items() if v >= centre_prior_slice)
    outcome = FAIL if beat else PASS
    return _verdict(
        "H2", outcome,
        f"{len(beat)}/{len(arm_slice)} arms match or beat the centre prior's "
        f"slice AUC of {centre_prior_slice:.4f}",
        centre_prior_slice_auc=centre_prior_slice, arm_slice_auc=arm_slice,
        arms_at_or_above=beat,
    )


def h3_verdict(p95: float | None, threshold: float, n_inits: int,
               n_required: int) -> dict:
    """Over at least ``n_required`` untrained inits, the 95th-percentile AUC > threshold."""
    if p95 is None:
        return _verdict("H3", NOT_RUN, "no floor in the artefact")
    if n_inits < n_required:
        return _verdict("H3", VOID,
                        f"{n_inits} inits is below the pre-registered {n_required}; "
                        "a floor from fewer is not the declared estimand",
                        n_inits=n_inits, n_required=n_required)
    outcome = PASS if p95 > threshold else FAIL
    return _verdict("H3", outcome,
                    f"95th-percentile untrained instance AUC {p95:.4f} vs {threshold}",
                    p95=p95, threshold=threshold, n_inits=n_inits)


def h4_verdict(skills: dict[str, float], per_arm: float, median_thr: float) -> dict:
    """Every arm's patient-specific skill < ``per_arm``, and the median < ``median_thr``.

    Both halves falsify independently -- the statement is a conjunction, so either
    an arm over the per-arm bound or a median over the median bound is enough.
    """
    if not skills:
        return _verdict("H4", NOT_RUN, "no arms present in the artefact")
    vals = sorted(skills.values())
    n = len(vals)
    med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    over = sorted(k for k, v in skills.items() if v >= per_arm)
    median_over = med >= median_thr
    outcome = FAIL if (over or median_over) else PASS
    # Arm names stay out of the reason string and live in ``arms_over_threshold``.
    # scripts/prereg_verdict.py blinds structured fields; it cannot blind a name
    # already formatted into prose, so a name in a reason is a blinding leak.
    reasons = []
    if over:
        reasons.append(f"{len(over)} arm(s) at or above {per_arm}")
    if median_over:
        reasons.append(f"median {med:.4f} is at or above {median_thr}")
    return _verdict("H4", outcome,
                    "; ".join(reasons) or
                    f"every arm below {per_arm} and median {med:.4f} below {median_thr}",
                    skills=skills, median=med, threshold_per_arm=per_arm,
                    threshold_median=median_thr, arms_over_threshold=over)


def h5_verdict(mean_skill: dict[str, float], threshold: float,
               per_seed: dict[str, list[float]] | None = None) -> dict:
    """No arm's prior-normalised skill, as the mean over its seeds, exceeds ``threshold``.

    ``per_seed`` is carried through and reported whether or not it falsifies:
    ``report_per_seed: true`` exists so the unit ruling cannot become a way of not
    publishing a seed that sits above the bar.
    """
    if not mean_skill:
        return _verdict("H5", NOT_RUN, "no arms present in the artefact")
    over = sorted(k for k, v in mean_skill.items() if v > threshold)
    outcome = FAIL if over else PASS
    detail: dict[str, Any] = {"mean_over_seeds": mean_skill, "threshold": threshold,
                              "arms_over_threshold": over, "unit": "mean_over_seeds"}
    if per_seed is not None:
        detail["per_seed"] = per_seed
        detail["seeds_over_threshold"] = {
            arm: [v for v in vals if v > threshold]
            for arm, vals in per_seed.items()
            if any(v > threshold for v in vals)
        }
    return _verdict("H5", outcome,
                    f"{len(over)}/{len(mean_skill)} arms exceed {threshold} "
                    "on the mean over seeds",
                    **detail)


def h6_verdict(slice_base: float | None, slice_arm: float | None,
               skill_base: float | None, skill_arm: float | None,
               threshold: float) -> dict:
    """Normal Guidance raises slice AUC but raises patient-specific skill by < ``threshold``.

    The falsifier is one-sided and names only the skill: "skill rises >= 0.02 --
    prior injection would then be adding real information and the critique fails."
    So the verdict attaches to the skill delta alone. Whether the slice AUC actually
    rose is reported beside it and does not falsify; if it did not rise, the
    statement's antecedent simply did not hold, and inventing a second falsifier
    here would be tightening a pre-registered rule after the fact.
    """
    if skill_base is None or skill_arm is None:
        return _verdict("H6", NOT_RUN,
                        "patient-specific skill missing for the arm or its base")
    d_skill = skill_arm - skill_base
    d_slice = (None if slice_base is None or slice_arm is None
               else slice_arm - slice_base)
    outcome = FAIL if d_skill >= threshold else PASS
    return _verdict(
        "H6", outcome,
        f"patient-specific skill moved {d_skill:+.4f} against a falsifier at "
        f">= {threshold}",
        delta_patient_specific_skill=d_skill, threshold=threshold,
        skill_base=skill_base, skill_arm=skill_arm,
        delta_slice_auc=d_slice, slice_base=slice_base, slice_arm=slice_arm,
        slice_rose=None if d_slice is None else bool(d_slice > 0),
        note="the falsifier names the skill only; the slice column is reported beside it",
    )


def h7_verdict(probe_skill: float | None, probe_threshold: float,
               content_free: dict[str, float | None],
               content_free_threshold: float) -> dict:
    """The probe's skill > ``probe_threshold`` while every content-free member's < the other.

    A member that is not computable is carried as ``None`` and cannot falsify --
    but if *no* member is computable the gate has nothing to gate with and the
    result is VOID, not a clearance. ``entropy_matched_random`` is the member this
    matters for: ``nulls.random_attn(k=1)`` returns exactly 1.0 everywhere, so it is
    undefined for the five K=1 arms and defined for the two K>1 arms, which is the
    same reason ``H7.content_free_construction`` refuses to take k from the probe.
    """
    if probe_skill is None:
        return _verdict("H7", NOT_RUN, "no probe skill in the artefact")
    computed = {k: v for k, v in content_free.items() if v is not None}
    uncomputable = sorted(k for k, v in content_free.items() if v is None)
    if not content_free:
        return _verdict("H7", NOT_RUN, "no content-free members in the artefact")
    if not computed:
        return _verdict("H7", VOID,
                        "no content-free member was computable, so the gate has "
                        "nothing to test against",
                        uncomputable=uncomputable)

    probe_clears = probe_skill > probe_threshold
    over = sorted(k for k, v in computed.items() if not v < content_free_threshold)
    outcome = PASS if (probe_clears and not over) else FAIL
    reasons = []
    if not probe_clears:
        reasons.append(f"probe skill {probe_skill:.4f} does not exceed {probe_threshold}")
    if over:
        reasons.append(f"content-free member(s) at or above "
                       f"{content_free_threshold}: {over}")
    return _verdict(
        "H7", outcome,
        "; ".join(reasons) or
        f"probe {probe_skill:.4f} > {probe_threshold} and every computed "
        f"content-free member < {content_free_threshold}",
        probe_skill=probe_skill, probe_threshold=probe_threshold,
        content_free=content_free, content_free_threshold=content_free_threshold,
        members_over_threshold=over, uncomputable=uncomputable,
    )


def h8_verdict(value: float | None, threshold: float) -> dict:
    """In-lung fitted-template AUC against ``threshold``. Two-sided: no falsifier.

    Returns ``REPORTED``, not PASS or FAIL. H8's declared falsifier is "none --
    reported either way. If restricting to lung removes the prior, that becomes the
    paper's actionable recommendation instead", so scoring it as a pass or a failure
    would invent a verdict the pre-registration deliberately declines to state.
    """
    if value is None:
        return _verdict("H8", NOT_RUN, "no in_lung_auc in the artefact")
    return _verdict("H8", REPORTED,
                    f"in_lung_auc {value:.4f} "
                    f"{'exceeds' if value > threshold else 'is below'} {threshold} "
                    "(two-sided, no falsifier)",
                    value=value, threshold=threshold, exceeds=bool(value > threshold),
                    estimand="in_lung_auc")


def h9_verdict(mosmed_auc: dict | None, lidc_auc: dict | None,
               mosmed_skill: float | None = None,
               lidc_skill: float | None = None) -> dict:
    """MosMed's fitted-template AUC is lower than LIDC's, with higher skill.

    Interval-scored, unlike the point-estimate hypotheses, because its own falsifier
    says so: "ordering reverses **or is indistinguishable**". Overlapping intervals
    are therefore a failure and not a null result -- the hypothesis asserts a
    detectable ordering, so failing to detect one falsifies it. ``mosmed_auc`` and
    ``lidc_auc`` are :func:`slotmil.eval.estimands.cluster_bootstrap` results.
    """
    if mosmed_auc is None or lidc_auc is None:
        return _verdict("H9", NOT_RUN,
                        "MosMed or LIDC fitted-template AUC is missing")
    for label, ci in (("mosmed", mosmed_auc), ("lidc", lidc_auc)):
        if ci.get("lo") is None or ci.get("hi") is None:
            return _verdict("H9", VOID, f"{label} interval is undefined", **{label: ci})

    lower = mosmed_auc["hi"] < lidc_auc["lo"]
    reversed_ = lidc_auc["hi"] < mosmed_auc["lo"]
    # An absent skill pair cannot contradict the ordering, so it does not veto it;
    # the AUC interval is the half the statement leads with.
    skill_ok = (mosmed_skill is None or lidc_skill is None
                or mosmed_skill > lidc_skill)
    if lower and skill_ok:
        outcome, why = PASS, "MosMed's template AUC interval sits below LIDC's"
    elif reversed_:
        outcome, why = FAIL, "the ordering reverses -- LIDC's template AUC is lower"
    elif lower:
        outcome, why = FAIL, ("the AUC ordering holds but the skill ordering does "
                              "not, and the statement asserts both")
    else:
        outcome, why = FAIL, "the intervals overlap: the ordering is indistinguishable"
    return _verdict("H9", outcome, why, mosmed_template_auc=mosmed_auc,
                    lidc_template_auc=lidc_auc, mosmed_skill=mosmed_skill,
                    lidc_skill=lidc_skill, basis="interval")


def h10_verdict(outcomes: Sequence[str | None]) -> dict:
    """Per seed, ``separable - trained`` scores as one of three outcomes; only one falsifies.

    ``trained_wins`` taking a majority of seeds withdraws the claim. An oracle win
    is *stronger* than the claim and an indistinguishable interval is the claim, so
    neither counts against it -- H10 was drafted as a two-outcome test that would
    have recorded an oracle win as a failure, and this is the corrected form.
    """
    scored = [o for o in outcomes if o is not None]
    if not scored:
        return _verdict("H10", NOT_RUN, "no seed produced a scoreable interval")
    tally = {k: scored.count(k) for k in
             ("oracle_wins", "indistinguishable", "trained_wins")}
    need = majority(len(scored))
    outcome = FAIL if tally["trained_wins"] >= need else PASS
    return _verdict("H10", outcome,
                    f"trained_wins on {tally['trained_wins']}/{len(scored)} seeds; "
                    f"a majority is {need}",
                    per_seed=list(outcomes), tally=tally, n_scored=len(scored),
                    majority_needed=need, undefined_seeds=len(outcomes) - len(scored))
