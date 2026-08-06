"""Bag-level classification metrics with the statistics ISBI/MICCAI expect.

plan.md line 116 asks for >=5 seeds, paired significance testing (DeLong for AUC)
and bootstrap CIs. Reporting a bare point estimate against the 0.879 MedMNIST
reference would not survive review.
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

    vals = np.array(vals)
    return {
        "mean": float(vals.mean()),
        "lo": float(np.percentile(vals, 100 * alpha / 2)),
        "hi": float(np.percentile(vals, 100 * (1 - alpha / 2))),
        "n_boot": len(vals),
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
