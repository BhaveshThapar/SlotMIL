"""Localisation metrics: slot attention masks vs annotated lesion masks.

Slot attention gives ``[K, N]`` over instances; reshaping to the patch grid and
upsampling to voxels turns each slot into a 3D heatmap that can be scored against
LIDC nodule masks or MosMed GGO/consolidation masks (plan.md line 106).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


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


def instance_auc(attn: np.ndarray, instance_labels: np.ndarray) -> float:
    """AUC of slot attention as an instance-level lesion detector.

    ``attn``: ``[K, N]``; the per-instance score is the max over slots.
    """
    if len(np.unique(instance_labels)) < 2:
        return float("nan")
    return float(roc_auc_score(instance_labels, attn.max(axis=0)))


def evaluate_localization(
    slot_attn: list[np.ndarray],
    masks: list[np.ndarray],
    n_slices: list[int],
    grid: int,
) -> dict:
    """Aggregate localisation over a split.

    ``masks`` are patch-grid targets ``[N]`` matching the flattened instance axis.
    """
    dices, ious, points, aucs = [], [], [], []

    for attn, mask, ns in zip(slot_attn, masks, n_slices):
        if mask.sum() == 0:
            continue
        heat = attn.reshape(attn.shape[0], ns, grid, grid)
        tgt = mask.reshape(ns, grid, grid)

        best = best_slot_dice(heat, tgt)
        dices.append(best["dice"])
        ious.append(best["iou"])
        points.append(pointing_game(heat[best["slot"]], tgt))
        a = instance_auc(attn, (mask > 0).astype(int))
        if not np.isnan(a):
            aucs.append(a)

    if not dices:
        return {"n_bags": 0}
    return {
        "dice": float(np.mean(dices)),
        "dice_std": float(np.std(dices)),
        "iou": float(np.mean(ious)),
        "pointing_game": float(np.mean(points)),
        "instance_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "n_bags": len(dices),
    }
