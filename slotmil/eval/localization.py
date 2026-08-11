"""Localisation metrics: slot attention masks vs annotated lesion masks.

Slot attention gives ``[K, N]`` over instances; reshaping to the patch grid and
upsampling to voxels turns each slot into a 3D heatmap that can be scored against
LIDC nodule masks or MosMed GGO/consolidation masks (plan.md line 106).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


def attn_to_volume(
    attn: torch.Tensor,
    n_slices: int,
    grid_h: int,
    grid_w: int,
    out_hw: tuple[int, int],
) -> torch.Tensor:
    """``[K, N]`` slot attention -> ``[K, n_slices, H, W]`` voxel heatmaps.

    Upsamples in-plane only. Slices are not interpolated across z because the
    instance axis is genuinely discrete -- inventing values between cached slices
    would fabricate localisation evidence.
    """
    k = attn.shape[0]
    expected = n_slices * grid_h * grid_w
    if attn.shape[-1] < expected:
        raise ValueError(
            f"attention has {attn.shape[-1]} instances, need {expected} "
            f"({n_slices} slices x {grid_h}x{grid_w} grid)"
        )
    x = attn[:, :expected].reshape(k * n_slices, 1, grid_h, grid_w)
    x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
    return x.reshape(k, n_slices, *out_hw)


def normalize_heatmap(h: np.ndarray) -> np.ndarray:
    lo, hi = h.min(), h.max()
    return (h - lo) / (hi - lo) if hi > lo else np.zeros_like(h)


def dice_iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict:
    p = (normalize_heatmap(pred) > threshold).astype(bool)
    t = target.astype(bool)
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    denom = p.sum() + t.sum()
    return {
        "dice": float(2 * inter / denom) if denom else 0.0,
        "iou": float(inter / union) if union else 0.0,
    }


def best_slot_dice(
    slot_heatmaps: np.ndarray, target: np.ndarray, thresholds: tuple[float, ...] = (0.3, 0.5, 0.7)
) -> dict:
    """Dice of the best-matching slot, over a small threshold sweep.

    Reported as "best slot" because SlotMIL is not asked to put *every* slot on
    the lesion -- the claim is that *some* slot binds it. Which slot that is gets
    checked separately for consistency by ``eval.alignment``.
    """
    best = {"dice": 0.0, "iou": 0.0, "slot": -1, "threshold": None}
    for k in range(slot_heatmaps.shape[0]):
        for th in thresholds:
            m = dice_iou(slot_heatmaps[k], target, threshold=th)
            if m["dice"] > best["dice"]:
                best = {**m, "slot": k, "threshold": th}
    return best


def pointing_game(heatmap: np.ndarray, target: np.ndarray) -> bool:
    """Does the heatmap's argmax fall inside the annotated region?

    Threshold-free, which makes it the more robust headline localisation number.
    """
    if target.sum() == 0:
        return False
    return bool(target.reshape(-1)[int(np.argmax(heatmap))])


def instance_auc(
    attn: np.ndarray, instance_labels: np.ndarray, slot: int | None = None
) -> float:
    """AUC of slot attention as an instance-level lesion detector.

    ``attn``: ``[K, N]``. ``slot`` selects which slot's attention is the score,
    and should be the lesion slot from the **validation-fitted, frozen** Hungarian
    assignment.

    ``slot=None`` falls back to max-over-slots, which is **invalid for slot
    attention and must not be used for reporting**. Slot attention normalises over
    the slot axis, so ``sum_k attn[k, n] == 1`` for every instance; the max over
    that axis therefore measures how confidently an instance is bound to *some*
    slot -- assignment confidence -- not lesion saliency. A background patch
    confidently bound to the background slot scores high.

    This distinction is not academic. On the LIDC test split with the frozen
    assignment, the same checkpoint gives:

        max-over-slots      0.4885   (looks like pure chance)
        frozen lesion slot  0.8423   (93% of the 0.9102 supervised ceiling)

    Reporting the former as "SlotMIL does not localise" inverted the actual
    result. The fallback is retained only so old runs remain reproducible.
    """
    if len(np.unique(instance_labels)) < 2:
        return float("nan")
    score = attn.max(axis=0) if slot is None else attn[slot]
    return float(roc_auc_score(instance_labels, score))


def evaluate_localization(
    slot_attn: list[np.ndarray],
    masks: list[np.ndarray],
    n_slices: list[int],
    grid: int,
    lesion_slot: int | None = None,
) -> dict:
    """Aggregate localisation over a split.

    ``masks`` are patch-grid targets ``[N]`` matching the flattened instance axis.
    ``lesion_slot`` is the slot named by the validation-fitted frozen assignment;
    pass it, or instance AUC falls back to the invalid max-over-slots statistic
    (see :func:`instance_auc`).

    Note on Dice and pointing game at this prevalence: lesion patches are ~0.07%
    of the LIDC grid. ``best_slot_dice`` min-max normalises then thresholds, so
    the predicted region is orders of magnitude larger than the target and Dice
    is driven to ~0 regardless of how well the attention *ranks* instances.
    Pointing game asks a single argmax over ~40,000 patches to land inside a
    ~30-patch target. Neither statistic separates "no signal" from "good ranking
    under extreme class imbalance", so both are reported for completeness but
    **average precision against prevalence is the honest low-prevalence measure**.
    """
    dices, ious, points, aucs, aps, prevs = [], [], [], [], [], []

    for attn, mask, ns in zip(slot_attn, masks, n_slices):
        if mask.sum() == 0:
            continue
        heat = attn.reshape(attn.shape[0], ns, grid, grid)
        tgt = mask.reshape(ns, grid, grid)
        binary = (mask > 0).astype(int)

        best = best_slot_dice(heat, tgt)
        dices.append(best["dice"])
        ious.append(best["iou"])
        points.append(pointing_game(heat[best["slot"]], tgt))

        a = instance_auc(attn, binary, slot=lesion_slot)
        if not np.isnan(a):
            aucs.append(a)
        if binary.sum() and binary.sum() < len(binary):
            score = attn[lesion_slot] if lesion_slot is not None else attn.max(axis=0)
            aps.append(average_precision_score(binary, score))
            prevs.append(binary.mean())

    if not dices:
        return {"n_bags": 0}
    out = {
        "dice": float(np.mean(dices)),
        "dice_std_across_bags": float(np.std(dices)),
        "iou": float(np.mean(ious)),
        "pointing_game": float(np.mean(points)),
        "instance_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "instance_auc_std_across_bags": float(np.std(aucs)) if aucs else float("nan"),
        "lesion_slot": lesion_slot,
        "n_bags": len(dices),
    }
    if aps:
        out["avg_precision"] = float(np.mean(aps))
        out["prevalence"] = float(np.mean(prevs))
        out["ap_lift_over_prevalence"] = float(np.mean(aps) / max(np.mean(prevs), 1e-12))
    return out
