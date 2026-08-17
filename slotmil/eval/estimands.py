"""The two numbers the paper asks people to report, and the machinery under them.

A localisation AUC of 0.84 sounds like localisation. On LIDC it decomposes into a
0.4965 metric floor, a 0.2488 population position prior, and 0.0971 that is
actually about the patient -- so roughly 28% of the above-floor score carries the
information the number is read as carrying. The point of this module is to make
that decomposition routine rather than heroic.

Both primary estimands are **imports, not inventions**, and the docstrings say so
where a reader will see them:

* :func:`prior_normalised_skill` is information gain over a baseline model
  (Kummerer, Wallis & Bethge, PNAS 2015) expressed on the AUC scale.
* :func:`patient_specific_skill` is shuffled AUC (Borji et al. ICCV 2013;
  Bylinskii et al. TPAMI 2019), the saliency field's standard centre-bias
  control, with the shuffle done across patients.

What is ours is the construction: a *fitted* content-free template as the
reference rather than an unfit average mask or a fixed Gaussian, and a
cross-patient control at the level of the naming step.

The gate that matters is power, not scepticism. A metric that scores zero for
everything is not a fix, so :func:`prior_normalised_skill` is only recommendable
if it also recovers the supervised patch probe's 0.91 -- that is hypothesis H7,
and ``tests/test_estimands.py`` pins both halves of it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "prior_normalised_skill",
    "patient_specific_skill",
    "reference_chain",
    "cluster_bootstrap",
    "h10_outcome",
    "stratified_auc",
]


def h10_outcome(ci: dict) -> str | None:
    """Score one seed's paired interval against H10's three outcomes.

    ``ci`` is a :func:`cluster_bootstrap` result for the paired per-bag
    ``separable - trained`` difference. Returns ``oracle_wins`` when the
    interval sits entirely above zero, ``trained_wins`` entirely below, and
    ``indistinguishable`` when it contains zero; ``None`` when the interval is
    undefined.

    Written as code rather than left to a reader's eye, for the reason H5's unit
    was a hole: a pre-registered verdict that exists only as an interval someone
    interprets can be interpreted differently later. H10 was in fact drafted
    with two outcomes and mis-scored its own evidence -- an oracle win is
    *stronger* than the claim, not a failure of it -- so the three are
    enumerated here and only ``trained_wins`` falsifies.
    """
    if ci.get("lo") is None or ci.get("hi") is None:
        return None
    if ci["lo"] > 0.0:
        return "oracle_wins"
    if ci["hi"] < 0.0:
        return "trained_wins"
    return "indistinguishable"


def prior_normalised_skill(auc: float, auc_prior: float) -> float:
    """Fraction of the headroom above a content-free reference that a method claims.

    ``(auc - auc_prior) / (1 - auc_prior)``. Zero when the method merely matches
    the prior, one at a perfect score, negative when the prior wins -- which
    happens, and is the whole point.

    ``auc_prior`` should be the *strongest* content-free reference available, not
    the most convenient. A fitted in-plane template beats both Arun et al.'s unfit
    average mask and Harvey et al.'s fixed sigma=1 slice Gaussian, so it shrinks
    the denominator and lowers the reported skill. That is the honest direction:
    a weak reference flatters every method measured against it.

    Prior art: structurally this is Kummerer, Wallis & Bethge (PNAS 2015),
    "Information-theoretic model comparison unifies saliency metrics", which
    defines information gain over a baseline model. Kept on the AUC scale here so
    it reads against published tables.
    """
    if not 0.0 <= auc_prior < 1.0:
        raise ValueError(f"auc_prior must be in [0, 1), got {auc_prior}")
    return (auc - auc_prior) / (1.0 - auc_prior)


def patient_specific_skill(auc_real: float, auc_cross_patient: float) -> float:
    """How much of a score survives when the target comes from a different patient.

    ``auc_real - auc_cross_patient``. The cross-patient arm keeps the model, the
    protocol and the selection step identical and swaps only *whose* lesion masks
    the assignment is fit against. Whatever score survives that swap was never
    about this patient -- it is population anatomy, and on LIDC it is most of the
    number.

    Prior art: this is shuffled AUC (Borji et al., ICCV 2013; Bylinskii et al.,
    TPAMI 2019), where negatives are drawn from other images so that a centre bias
    scores exactly chance. The construction here shuffles at the patient level,
    which is the unit that matters clinically and the unit the bootstrap
    resamples.
    """
    return auc_real - auc_cross_patient


def reference_chain(links: dict[str, float], order: list[str]) -> list[dict]:
    """Adjacent differences along a nested chain of reference models.

    Reported as a chain -- chance -> fitted template -> cross-patient -> full --
    rather than as an additive three-term sum, because AUC is not additive and a
    reader is right to object to a decomposition that pretends otherwise. Each
    step is the gain from *relaxing one constraint*, which is well defined
    whatever the metric's algebra.

    Returns one row per step with ``from``, ``to``, ``delta`` and ``frac`` (the
    step's share of the total climb), plus a final ``total`` row.
    """
    missing = [k for k in order if k not in links]
    if missing:
        raise KeyError(f"chain references undefined links: {missing}")
    if len(order) < 2:
        raise ValueError("a chain needs at least two links")

    total = links[order[-1]] - links[order[0]]
    rows = []
    for a, b in zip(order, order[1:]):
        d = links[b] - links[a]
        rows.append({
            "from": a,
            "to": b,
            "delta": float(d),
            "frac": float(d / total) if total else float("nan"),
        })
    rows.append({"from": order[0], "to": order[-1], "delta": float(total),
                 "frac": 1.0 if total else float("nan")})
    return rows


def cluster_bootstrap(values, clusters, reps: int = 10000, seed: int = 0,
                      alpha: float = 0.05) -> dict:
    """Mean and percentile CI, resampling *clusters* rather than rows.

    Patients, not bags. A LIDC patient can contribute more than one series -- 999
    series come from 991 patients -- and series from one patient share anatomy, so
    a row-level bootstrap treats correlated observations as independent and
    reports an interval that is too narrow. The effect is small here but it is
    free to do correctly, and "we resampled the wrong unit" is an avoidable
    reviewer finding.

    Non-finite values are dropped before resampling, so a bag that was degenerate
    for one axis does not poison the others.

    ``reps=0`` returns the point estimate with ``lo``/``hi`` as ``None`` and does
    no resampling. That exists for replicated nulls: H7's content-free members are
    now drawn many times per tag, the per-draw values are summarised before the
    verdict reads them, and an interval on a single draw of a null is not a
    quantity anything reports. It also keeps the "drop non-finite, then take the
    mean over rows" rule in one place -- a second implementation of it in a driver
    would be a copy that can go stale against this one.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    if values.shape[0] != clusters.shape[0]:
        raise ValueError(
            f"values and clusters disagree: {values.shape[0]} vs {clusters.shape[0]}"
        )

    ok = np.isfinite(values)
    values, clusters = values[ok], clusters[ok]
    if values.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0, "n_clusters": 0}

    uniq = np.unique(clusters)
    if reps <= 0:
        return {"mean": float(values.mean()), "lo": None, "hi": None,
                "n": int(values.size), "n_clusters": int(uniq.size)}

    index = {c: np.flatnonzero(clusters == c) for c in uniq}
    rng = np.random.default_rng(seed)
    boot = np.empty(reps)
    for r in range(reps):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        boot[r] = values[np.concatenate([index[c] for c in pick])].mean()

    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(values.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(values.size),
        "n_clusters": int(uniq.size),
    }


def stratified_auc(scores, labels, strata) -> float:
    """Mantel-Haenszel AUC: concordant pairs counted only *within* strata.

    Comparing a positive against a negative from the same stratum removes any
    ranking a method gets for free from the stratifying variable. With strata of
    (normalised slice x radial bin), a score that is purely positional collapses
    toward 0.5, which is exactly what the paper needs to demonstrate.

    Strata containing only positives or only negatives contribute no pairs and are
    skipped, so a fine stratification quietly discards data -- check the returned
    pair count in :func:`stratified_auc_detail` before reading a value that used
    few strata.
    """
    return stratified_auc_detail(scores, labels, strata)["auc"]


def stratified_auc_detail(scores, labels, strata) -> dict:
    """:func:`stratified_auc` plus the bookkeeping needed to trust it."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    strata = np.asarray(strata)
    if not (scores.shape == labels.shape == strata.shape):
        raise ValueError(
            f"shape mismatch: scores {scores.shape}, labels {labels.shape}, "
            f"strata {strata.shape}"
        )

    concordant = 0.0
    pairs = 0.0
    used = 0
    for s in np.unique(strata):
        m = strata == s
        pos = scores[m & labels]
        neg = scores[m & ~labels]
        if pos.size == 0 or neg.size == 0:
            continue
        used += 1
        neg_sorted = np.sort(neg)
        lo = np.searchsorted(neg_sorted, pos, side="left")
        hi = np.searchsorted(neg_sorted, pos, side="right")
        # Midpoint of the two insertion points credits ties with half a pair,
        # matching the standard tie-corrected AUC rather than silently favouring
        # whichever side the sort happened to place them.
        concordant += float(((lo + hi) / 2.0).sum())
        pairs += pos.size * neg.size

    return {
        "auc": float(concordant / pairs) if pairs else float("nan"),
        "n_pairs": int(pairs),
        "n_strata_used": used,
        "n_strata_total": int(np.unique(strata).size),
    }
