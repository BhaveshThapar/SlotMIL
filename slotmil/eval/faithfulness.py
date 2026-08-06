"""Deletion / insertion faithfulness curves (plan.md line 112).

Attention weights being interpretable is a claim, not a given -- a slot can look
convincing on an overlay while contributing nothing to the decision. These curves
test whether removing what a slot attends to actually moves the prediction.

Deletion: progressively drop the instances a slot attends to most; a faithful
explanation makes the score fall fast (low AUC is good).
Insertion: start from an empty bag and add those instances first; a faithful
explanation makes the score rise fast (high AUC is good).
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def _bag_score(model, feats, pad_mask, target_class: int, device) -> float:
    out = model(feats.to(device), pad_mask.to(device))
    return torch.softmax(out["logits"], dim=-1)[0, target_class].item()


@torch.no_grad()
def deletion_curve(
    model,
    feats: torch.Tensor,  # 1, N, D
    attn: torch.Tensor,  # K, N
    target_class: int,
    device: str = "cuda",
    steps: int = 20,
    slot: int | None = None,
) -> dict:
    """Drop the top-attended instances in ``steps`` increments.

    Instances are removed by clearing the pad mask rather than zeroing features,
    so the model sees a genuinely shorter bag instead of a bag full of zero
    vectors that it may still attend to.
    """
    model.eval()
    n = feats.shape[1]
    score = attn.max(dim=0).values if slot is None else attn[slot]
    order = torch.argsort(score, descending=True)

    mask = torch.ones(1, n, dtype=torch.bool)
    scores = [_bag_score(model, feats, mask, target_class, device)]
    per_step = max(1, n // steps)

    for s in range(steps):
        drop = order[s * per_step : (s + 1) * per_step]
        mask[0, drop] = False
        if mask.sum() == 0:  # keep at least one instance or pooling is undefined
            mask[0, order[-1]] = True
        scores.append(_bag_score(model, feats, mask, target_class, device))

    frac = np.linspace(0, 1, len(scores))
    return {"scores": scores, "fraction": frac.tolist(), "auc": float(np.trapz(scores, frac))}


@torch.no_grad()
def insertion_curve(
    model,
    feats: torch.Tensor,
    attn: torch.Tensor,
    target_class: int,
    device: str = "cuda",
    steps: int = 20,
    slot: int | None = None,
) -> dict:
    model.eval()
    n = feats.shape[1]
    score = attn.max(dim=0).values if slot is None else attn[slot]
    order = torch.argsort(score, descending=True)

    mask = torch.zeros(1, n, dtype=torch.bool)
    mask[0, order[0]] = True  # pooling needs a non-empty bag
    scores = [_bag_score(model, feats, mask, target_class, device)]
    per_step = max(1, n // steps)

    for s in range(steps):
        add = order[s * per_step : (s + 1) * per_step]
        mask[0, add] = True
        scores.append(_bag_score(model, feats, mask, target_class, device))

    frac = np.linspace(0, 1, len(scores))
    return {"scores": scores, "fraction": frac.tolist(), "auc": float(np.trapz(scores, frac))}


def faithfulness_summary(deletions: list[dict], insertions: list[dict]) -> dict:
    """Lower deletion AUC and higher insertion AUC both mean more faithful."""
    d = np.array([x["auc"] for x in deletions])
    i = np.array([x["auc"] for x in insertions])
    return {
        "deletion_auc": float(d.mean()),
        "deletion_auc_std": float(d.std()),
        "insertion_auc": float(i.mean()),
        "insertion_auc_std": float(i.std()),
        "insertion_minus_deletion": float(i.mean() - d.mean()),
        "n_bags": len(d),
    }
