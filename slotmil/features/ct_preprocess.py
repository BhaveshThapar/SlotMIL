"""CT volume preprocessing: HU windowing, lung-region slice selection, resizing.

Slice selection is what keeps the feature cache inside budget. LIDC is 243,958
slices across 1,018 series; caching every one at ViT-B/14 patch resolution would
be ~92 GB. Restricting to slices that actually contain lung drops that by roughly
40% with no loss for a nodule-localisation task, since a slice with no lung
cannot contain a lung nodule.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

# Standard lung window: air at -1000 HU, soft tissue/vessels up to ~400 HU.
HU_MIN, HU_MAX = -1000.0, 400.0


def window_hu(volume: np.ndarray, hu_min: float = HU_MIN, hu_max: float = HU_MAX) -> np.ndarray:
    """Clip to a HU window and scale to [0, 1]."""
    v = np.clip(volume.astype(np.float32), hu_min, hu_max)
    return (v - hu_min) / (hu_max - hu_min)


def _cleared_air_per_slice(volume_hu: np.ndarray, air_thresh: float) -> np.ndarray:
    """Air voxels that are not connected to the slice border.

    The border-connected component is the outside world and the scanner table,
    not lung. Extracted so the evaluation mask can reuse it to repair slices
    that :func:`lung_mask_3d`'s 3D closing erodes away.
    """
    air = volume_hu < air_thresh
    cleared = np.zeros_like(air)
    for z in range(air.shape[0]):
        lab, n = ndimage.label(air[z])
        if n == 0:
            continue
        border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
        border.discard(0)
        keep = np.isin(lab, list(set(range(1, n + 1)) - border))
        cleared[z] = keep
    return cleared


def lung_mask_3d(volume_hu: np.ndarray, air_thresh: float = -320.0) -> np.ndarray:
    """Coarse binary lung mask from raw HU.

    Threshold for air, drop everything connected to the volume border (that is
    the outside world and the scanner table, not lung), then keep the two largest
    remaining components. Deliberately crude -- it only has to decide which slices
    to keep, not to be a segmentation result.
    """
    cleared = _cleared_air_per_slice(volume_hu, air_thresh)

    lab, n = ndimage.label(cleared)
    if n == 0:
        return cleared
    sizes = ndimage.sum(cleared, lab, range(1, n + 1))
    keep_labels = np.argsort(sizes)[::-1][:2] + 1
    mask = np.isin(lab, keep_labels)

    return ndimage.binary_closing(mask, structure=np.ones((3, 3, 3)))


def _disk(radius: int) -> np.ndarray:
    r = int(radius)
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (y * y + x * x) <= r * r


def lung_mask_for_evaluation(
    volume_hu: np.ndarray,
    air_thresh: float = -320.0,
    method: str = "fill_hull",
    closing_radius: int = 8,
    min_component_frac: float = 0.001,
) -> np.ndarray:
    """Lung mask suitable for *restricting an evaluation region*.

    Deliberately separate from :func:`lung_mask_3d`, which only has to decide
    which slices to keep. The difference is load-bearing and easy to get wrong:

    **An air threshold excludes the nodules themselves.** A nodule is soft
    tissue, not air, so it fails ``volume < -320``. Solitary nodules sitting in
    parenchyma survive as holes that get filled, but juxtapleural nodules (fused
    with the chest wall) and juxtavascular ones (fused with a vessel) are
    connected to non-lung structure and fall *outside* an air mask entirely.
    Restricting a localisation metric to that mask would delete precisely the
    targets the metric is meant to score -- turning "does attention find the
    nodule inside the lung?" into "does attention find the nodule in a region
    the nodule was removed from". The control would not be conservative; it
    would be inverted.

    So the mask is grown until it contains the lesions, and the containment
    fraction is measured rather than assumed -- see ``scripts/lung_mask_lidc.py``,
    which reports it per series and gates on it.

    ``method``:

    ``air``
        :func:`lung_mask_3d` unchanged. Kept as the negative control that
        demonstrates the problem, not as a candidate.
    ``fill``
        plus per-slice hole filling. Recovers nodules fully enclosed by
        parenchyma.
    ``fill_close``
        plus morphological closing with ``closing_radius``. Bridges small
        pleural indentations.
    ``fill_hull``
        plus a per-slice convex hull of each lung. Recovers juxtapleural
        nodules, at the cost of including some mediastinum -- over-inclusion is
        the safe direction here, since it only weakens the restriction rather
        than biasing it toward the answer we want.

    Returns a boolean ``(S, H, W)`` array.
    """
    if method not in ("air", "fill", "fill_close", "fill_hull"):
        raise ValueError(
            f"method must be air|fill|fill_close|fill_hull, got {method!r}"
        )

    mask = lung_mask_3d(volume_hu, air_thresh=air_thresh)
    if method == "air":
        return mask

    # lung_mask_3d closes with a 3x3x3 element, and binary erosion reads outside
    # the volume as background, so the first and last slices come back empty.
    # Harmless when the mask only has to pick a slice range -- fatal here, since
    # an empty slice contains no lung patches and would silently drop every
    # lesion patch on it from the evaluation region.
    empty = ~mask.reshape(mask.shape[0], -1).any(axis=1)
    if empty.any():
        mask[empty] = _cleared_air_per_slice(volume_hu, air_thresh)[empty]

    for z in range(mask.shape[0]):
        mask[z] = ndimage.binary_fill_holes(mask[z])
    if method == "fill":
        return mask

    if method == "fill_close":
        se = _disk(closing_radius)
        for z in range(mask.shape[0]):
            if mask[z].any():
                mask[z] = ndimage.binary_fill_holes(
                    ndimage.binary_closing(mask[z], structure=se)
                )
        return mask

    from skimage.morphology import convex_hull_image

    min_area = min_component_frac * mask.shape[1] * mask.shape[2]
    for z in range(mask.shape[0]):
        if not mask[z].any():
            continue
        lab, n = ndimage.label(mask[z])
        if n == 0:
            continue
        sizes = ndimage.sum(mask[z], lab, range(1, n + 1))
        out = np.zeros_like(mask[z])
        # Hull each lung separately -- hulling both together would swallow the
        # mediastinum wholesale.
        for li in np.argsort(sizes)[::-1][:2] + 1:
            comp = lab == li
            if comp.sum() < min_area:
                continue
            out |= convex_hull_image(comp)
        mask[z] = out | mask[z]
    return mask


def select_lung_slices(
    volume_hu: np.ndarray, min_area_frac: float = 0.005, margin: int = 2
) -> np.ndarray:
    """Indices of axial slices containing meaningful lung.

    Returns every slice if the mask fails -- better to cache a bigger bag than to
    silently drop a whole series because the heuristic misfired.
    """
    mask = lung_mask_3d(volume_hu)
    area = mask.reshape(mask.shape[0], -1).mean(axis=1)
    idx = np.where(area > min_area_frac)[0]
    if idx.size == 0:
        return np.arange(volume_hu.shape[0])
    lo = max(0, idx.min() - margin)
    hi = min(volume_hu.shape[0], idx.max() + 1 + margin)
    return np.arange(lo, hi)


def slices_to_batch(
    volume01: np.ndarray, image_size: int = 224, device: str = "cpu"
) -> torch.Tensor:
    """``(S, H, W)`` in [0, 1] -> ``(S, 3, image_size, image_size)``.

    The channel is replicated three ways because the ImageNet/DINOv2 stem expects
    RGB; the three channels carry identical CT intensity.
    """
    x = torch.from_numpy(np.ascontiguousarray(volume01)).float().unsqueeze(1)
    x = x.to(device)
    if x.shape[-1] != image_size or x.shape[-2] != image_size:
        x = F.interpolate(
            x, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
    return x.repeat(1, 3, 1, 1)


def mask_to_patch_grid(
    mask: np.ndarray, grid: int, mode: str = "mean"
) -> np.ndarray:
    """Rasterise a per-slice binary lesion mask onto the ViT patch grid.

    ``(S, H, W)`` binary -> ``(S, grid, grid)``. ``mean`` gives the fraction of
    each patch covered by lesion (a soft target); ``any`` gives a hard 0/1.

    This is what puts the localisation ground truth on exactly the same grid as
    the cached features, so Dice against slot attention is an apples-to-apples
    comparison rather than an interpolation artefact.
    """
    x = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(1)
    pooled = F.adaptive_avg_pool2d(x, (grid, grid)).squeeze(1).numpy()
    if mode == "any":
        return (pooled > 0).astype(np.uint8)
    if mode == "mean":
        return pooled.astype(np.float32)
    raise ValueError(f"mode must be mean|any, got {mode!r}")
