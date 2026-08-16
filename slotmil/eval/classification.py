"""Bag-level classification metrics with the statistics ISBI/MICCAI expect.

plan.md line 116 asks for >=5 seeds, paired significance testing (DeLong for AUC)
and bootstrap CIs. Reporting a bare point estimate against the 0.879 MedMNIST
reference would not survive review.

Multiplicity lives here too: nine pre-registered hypotheses (H1-H9) are read at
alpha = 0.05, so an uncorrected family would run roughly a 37% chance of at least
one spurious rejection under the global null. `configs/prereg/isbi2027.yaml:265`
declares `multiplicity: holm`; :func:`holm_adjust` is that declaration.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, multilabel: bool = False
) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if multilabel:
        y_pred = (y_score > 0.5).astype(int)
        return {
            "auc": roc_auc_score(y_true, y_score, average="macro"),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "acc": accuracy_score(y_true.ravel(), y_pred.ravel()),
        }

    n_classes = y_score.shape[1] if y_score.ndim > 1 else 2
    if n_classes == 2:
        pos = y_score[:, 1] if y_score.ndim > 1 else y_score
        auc = roc_auc_score(y_true, pos)
        y_pred = (pos > 0.5).astype(int)
    else:
        auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
        y_pred = y_score.argmax(axis=1)

    return {
        "auc": auc,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "acc": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
    }


def _auc_variance_components(y_true: np.ndarray, y_score: np.ndarray):
    """DeLong structural components (V10, V01) for one model."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        raise ValueError("DeLong needs both classes present")

    # V10[i] = fraction of negatives that positive i beats (ties count half).
    v10 = np.array([(np.sum(p > neg) + 0.5 * np.sum(p == neg)) / n for p in pos])
    v01 = np.array([(np.sum(pos > q) + 0.5 * np.sum(pos == q)) / m for q in neg])
    auc = v10.mean()
    return auc, v10, v01


def delong_test(
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray
) -> dict:
    """Paired DeLong test for two AUCs on the same samples.

    Returns the AUC difference, z statistic and two-sided p-value. Use for
    "does SlotMIL beat Gated-Attention MIL?" -- the models are evaluated on
    identical bags, so the paired form is the correct one.
    """
    y_true = np.asarray(y_true)
    auc_a, v10_a, v01_a = _auc_variance_components(y_true, np.asarray(score_a))
    auc_b, v10_b, v01_b = _auc_variance_components(y_true, np.asarray(score_b))
    m, n = len(v10_a), len(v01_a)

    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    s = s10 / m + s01 / n

    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return {"auc_a": auc_a, "auc_b": auc_b, "delta": auc_a - auc_b, "z": 0.0, "p": 1.0}

    z = (auc_a - auc_b) / np.sqrt(var)
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "delta": auc_a - auc_b,
        "z": float(z),
        "p": float(2 * (1 - stats.norm.cdf(abs(z)))),
    }


def holm_adjust(pvalues) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values for a family of tests.

    The pre-registration declares `multiplicity: holm` over the nine-hypothesis
    family H1-H9 at alpha = 0.05 and nothing implemented it. Hand-rolled for the
    same reason :func:`delong_test` is: `pyproject.toml` declares no dependencies
    at all, and a scavenger node that has to resolve statsmodels at job start is a
    failure mode, not an inconvenience.

    Sort ascending, scale the i-th smallest (0-indexed) by ``n - i``, take the
    running maximum, clip at 1.0, restore the caller's order. The two middle steps
    are the whole job. Without the running maximum the output can be non-monotone:
    p = [0.03, 0.04] scales to [0.06, 0.04], which reads as "the larger raw p is
    the more significant" and lets a caller reject a hypothesis whose predecessor
    in the step-down order was not rejected. Enforcing monotonicity is exactly what
    makes ``p_adjusted <= alpha`` equivalent to walking the step-down rule by hand,
    which is why :func:`holm_reject` needs no loop.

    NaN in, NaN out -- **and the NaN still counts toward the family size**. A
    hypothesis that could not be computed (a degenerate stratum, a bag with one
    class) must not be free: dropping it shrinks n and makes every surviving
    hypothesis easier to reject, the one direction a multiplicity correction must
    never move. NaN sorts last, so the finite p-values keep multipliers n, n-1,
    ... -- the smallest still pays full Bonferroni -- and the uncomputable
    hypothesis is simply never rejected. Reporting it as NaN rather than 1.0 keeps
    "we could not test this" distinguishable from "we tested it and it failed".

    n = 1 is the identity, since the only multiplier is ``1 - 0``. p-values outside
    [0, 1] raise: 1.7 is an upstream bug, and clipping it here would hide it.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"holm_adjust needs a 1-D family, got shape {p.shape}")
    if p.size == 0:
        return p.copy()

    # NaN is exempted from the range check, not from the family -- see above.
    bad = ~np.isnan(p) & ((p < 0.0) | (p > 1.0))
    if bad.any():
        raise ValueError(f"p-values outside [0, 1]: {p[bad].tolist()}")

    n = p.size
    order = np.argsort(p)  # numpy sorts NaN to the end, which is the behaviour above
    scaled = p[order] * (n - np.arange(n))
    adjusted = np.minimum(np.maximum.accumulate(scaled), 1.0)

    out = np.empty(n, dtype=float)
    # Inverse permutation. `adjusted[order]` is the classic wrong answer: it sorts
    # a second time instead of un-sorting, and happens to agree only when `order`
    # is its own inverse -- so it passes on sorted and reversed inputs and silently
    # mislabels every hypothesis on any other.
    out[order] = adjusted
    return out


def holm_reject(pvalues, alpha: float = 0.05) -> dict:
    """:func:`holm_adjust` plus the decision, so alpha never meets a raw p-value.

    Returning the adjusted vector without the rejections invites the one mistake
    the correction exists to prevent -- reading `p` off `delong_test` and comparing
    it to 0.05 directly. Callers should take ``reject`` from here rather than
    thresholding anything themselves.

    ``alpha`` defaults to the pre-registered 0.05. NaN adjusted p-values compare
    False, so a hypothesis that could not be computed is never rejected.
    """
    p_adjusted = holm_adjust(pvalues)
    reject = p_adjusted <= alpha
    return {
        "p_adjusted": p_adjusted,
        "reject": reject,
        "alpha": float(alpha),
        "n_family": int(p_adjusted.size),
        "n_reject": int(reject.sum()),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "auc",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI, resampling bags (not instances)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)

    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            vals.append(classification_metrics(y_true[idx], y_score[idx])[metric])
        except ValueError:
            continue

    arr = np.asarray(vals, dtype=float)
    return {
        "mean": float(arr.mean()),
        "lo": float(np.percentile(arr, 100 * alpha / 2)),
        "hi": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        "n_boot": int(arr.size),
    }


def aggregate_seeds(runs: list[dict], keys: list[str] | None = None) -> dict:
    """mean +/- std across seeds, the format the results tables report."""
    keys = keys or [k for k, v in runs[0].items() if isinstance(v, (int, float))]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in runs if k in r], dtype=float)
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "n": len(v), "values": v.tolist()}
    return out
