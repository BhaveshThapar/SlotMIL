"""Slot-to-radiological-finding alignment -- the money experiment (plan.md line 135).

This is the moat. Slot-MIL (arXiv:2311.17466) already claimed slot attention as
MIL pooling for WSI, and INSIGHT (arXiv:2412.02012) already claimed interpretable
CT MIL aggregation. What neither did is measure, against real lesion masks,
whether the learned latents bind to named radiological findings. That is what
this module computes.

The critical methodological point is in :func:`fit_slot_assignment` /
:func:`apply_slot_assignment`: the slot -> finding mapping is fit on validation
and frozen before test is touched. Reviewer objection #3 is "you post-hoc named
the slots", and the only answer is that naming happened before the test set was
read. Fitting the Hungarian assignment on test would be exactly the error being
defended against, so the two steps are separate functions and the assignment is
an explicit argument rather than an internal default.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score


def slot_finding_affinity(
    slot_attn: list[np.ndarray], finding_masks: list[np.ndarray]
) -> np.ndarray:
    """Mean affinity between each slot and each finding class.

    Args:
        slot_attn: per-bag ``[K, N]`` slot attention (already mask-zeroed).
        finding_masks: per-bag ``[F, N]`` binary finding masks on the same grid.

    Returns:
        ``[K, F]`` affinity. Entry (k, f) is the average share of slot k's
        attention mass that lands inside finding f.
    """
    if len(slot_attn) != len(finding_masks):
        raise ValueError("slot_attn and finding_masks must align bag-for-bag")

    total = None
    count = 0
    for attn, masks in zip(slot_attn, finding_masks):
        if masks.sum() == 0:  # no annotated finding in this bag
            continue
        a = attn / attn.sum(axis=-1, keepdims=True).clip(1e-8)
        aff = a @ masks.T  # K, F
        total = aff if total is None else total + aff
        count += 1

    if count == 0:
        raise ValueError("no bags with annotated findings; cannot compute affinity")
    return total / count


def fit_slot_assignment(affinity: np.ndarray) -> dict[int, int]:
    """Hungarian slot -> finding assignment, maximising total affinity.

    **Fit on validation only.** The returned mapping is the frozen naming that
    gets applied to test.
    """
    row, col = linear_sum_assignment(-affinity)
    return {int(r): int(c) for r, c in zip(row, col)}


def apply_slot_assignment(
    affinity: np.ndarray, assignment: dict[int, int]
) -> dict:
    """Score a *held-out* affinity matrix under a previously frozen assignment."""
    scores = [affinity[k, f] for k, f in assignment.items()]
    n_slots, n_findings = affinity.shape
    chance = affinity.mean()
    return {
        "mean_assigned_affinity": float(np.mean(scores)),
        "per_finding": {int(f): float(affinity[k, f]) for k, f in assignment.items()},
        "chance_affinity": float(chance),
        "lift_over_chance": float(np.mean(scores) / max(chance, 1e-8)),
        "n_slots": n_slots,
        "n_findings": n_findings,
    }


def slot_purity(
    slot_attn: list[np.ndarray], finding_labels: list[np.ndarray]
) -> dict:
    """Purity and NMI of the hard slot assignment of instances vs finding labels.

    Each instance is assigned to its argmax slot; that clustering is then scored
    against the finding label of the instance. plan.md line 54 asks for exactly
    this, reported per-head for multi-head ABMIL and per-slot for SlotMIL, so the
    two are directly comparable.
    """
    cluster, truth = [], []
    for attn, labels in zip(slot_attn, finding_labels):
        cluster.append(attn.argmax(axis=0))
        truth.append(labels)

    cluster = np.concatenate(cluster)
    truth = np.concatenate(truth)

    # Purity: for each slot, the frequency of its most common finding label.
    purity_num = 0
    for k in np.unique(cluster):
        sel = truth[cluster == k]
        if sel.size:
            purity_num += np.bincount(sel.astype(int)).max()

    return {
        "purity": float(purity_num / max(len(truth), 1)),
        "nmi": float(normalized_mutual_info_score(truth, cluster)),
        "n_instances": int(len(truth)),
        "n_active_slots": int(len(np.unique(cluster))),
    }


def slot_consistency(
    slot_attn: list[np.ndarray], finding_masks: list[np.ndarray]
) -> dict:
    """Does slot index k bind the same finding across bags, above chance?

    plan.md line 56 asks for a permutation/consistency test. For each bag, the
    dominant finding of each slot is recorded; consistency is the modal
    agreement rate across bags. Chance is 1/n_findings.
    """
    per_bag = []
    for attn, masks in zip(slot_attn, finding_masks):
        if masks.sum() == 0:
            continue
        a = attn / attn.sum(axis=-1, keepdims=True).clip(1e-8)
        per_bag.append((a @ masks.T).argmax(axis=1))  # K

    if not per_bag:
        raise ValueError("no annotated bags for consistency test")

    arr = np.stack(per_bag)  # B, K
    n_findings = max(int(arr.max()) + 1, 2)

    rates = []
    for k in range(arr.shape[1]):
        counts = np.bincount(arr[:, k], minlength=n_findings)
        rates.append(counts.max() / counts.sum())

    chance = 1.0 / n_findings
    return {
        "mean_consistency": float(np.mean(rates)),
        "per_slot": [float(r) for r in rates],
        "chance": chance,
        "lift_over_chance": float(np.mean(rates) / chance),
        "n_bags": int(arr.shape[0]),
    }


def head_redundancy(slot_attn: list[np.ndarray]) -> dict:
    """Mean pairwise cosine between slot/head attention maps.

    The direct test of reviewer objection #1: if multi-head ABMIL's heads are
    redundant (high similarity) where SlotMIL's slots are not, competition rather
    than capacity is producing the specialisation.
    """
    sims = []
    for attn in slot_attn:
        a = attn / np.linalg.norm(attn, axis=-1, keepdims=True).clip(1e-8)
        s = a @ a.T
        k = s.shape[0]
        if k < 2:
            continue
        sims.append(s[~np.eye(k, dtype=bool)].mean())
    if not sims:
        return {"mean_pairwise_cosine": 0.0, "n_bags": 0}
    return {"mean_pairwise_cosine": float(np.mean(sims)), "n_bags": len(sims)}
