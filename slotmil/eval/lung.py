"""Reading the stored lung masks, which H8 needs and nothing else consumed.

``scripts/lung_mask_lidc.py`` computed a per-patch lung fraction for all 999 LIDC
series and merged them into ``data/lidc/lung_masks.h5``. Until now that file had
no reader outside its own producer's resume check: ``grep -rn lung
slotmil/eval/`` returned nothing, so the estimand ``in_lung_stratified_auc``
declared at ``configs/prereg/isbi2027.yaml:446`` had no implementation and H8 had
no way to be scored. This module is that reader.

**The store is already at cached-slice resolution.** Verified across all 999
series: ``lung`` has exactly as many rows as the feature cache has cached slices,
and exactly as many as the cache's ``n_slices`` attribute. It is *not* at the
original DICOM slice resolution -- ``max(slice_index) + 1`` disagrees on 714 of
999, because the cache keeps a selected subset of slices and ``slice_index``
records which originals they were. So the alignment to a bag is **positional**
and needs no reindexing through ``slice_index``. That is worth stating rather
than rediscovering: had it gone the other way, a positional zip would have
mis-scored every patch of every bag without raising.

The raw DICOM this was derived from is gone -- ``lung_mask_lidc.py`` deletes each
series after masking it -- so the store cannot be regenerated locally. Treat it
as source data, not as a cache.
"""

from __future__ import annotations

import numpy as np

__all__ = ["in_lung_from_grid", "load_lung_grid", "restrict_to_lung"]

DEFAULT_LUNG_THRESH = 0.0   # protocol.lung_mask.lung_thresh


def load_lung_grid(f, uid: str) -> np.ndarray:
    """``[n_slices, grid, grid]`` per-patch lung fraction for one series.

    ``f`` is an open :class:`h5py.File`, not a path. The caller owns the handle
    because a DataLoader worker must open its own after forking -- sharing one
    across workers corrupts reads, which is why
    :meth:`slotmil.data.feature_cache.FeatureBagDataset._file` opens lazily.

    Values are continuous in ``[0, 1]``: the fraction of a 22mm patch that is
    lung, not a binary flag. Thresholding is :func:`in_lung_from_grid`'s job and
    is a pre-registered choice, so it does not happen here.
    """
    if uid not in f:
        raise KeyError(f"{uid} is not in the lung mask store")
    g = f[uid]
    if "lung" not in g:
        raise KeyError(f"{uid} has no 'lung' dataset; the store is incomplete")
    return np.asarray(g["lung"][:], dtype=np.float32)


def in_lung_from_grid(grid: np.ndarray, lung_thresh: float = DEFAULT_LUNG_THRESH,
                      n_expected: int | None = None) -> np.ndarray:
    """``[n_slices, g, g]`` lung fractions -> flat ``[N]`` in-lung boolean.

    The rule is ``protocol.lung_mask.patch_rule``: *a patch is in-lung when it
    contains any lung at all*. With ``lung_thresh`` pre-registered at 0.0 that is
    **strictly greater than**, not greater-or-equal -- ``>=`` at threshold zero
    would call every patch in-lung and silently turn H8 into the unrestricted
    number it is supposed to be contrasted with.

    Flattened in C order, so slice ``s`` patch ``p`` lands at ``s * g * g + p`` --
    the same layout the feature cache's ``mask`` and the attention dumps use.

    ``n_expected`` is the bag's instance count. Passing it turns a silent
    mis-score into an exception: every consumer here zips a per-patch score
    against a per-patch flag, and a short zip truncates the analysis rather than
    raising (which is why ruff's ``B905`` ignore is recorded as pending an audit
    rather than as permanent).
    """
    if grid.ndim != 3:
        raise ValueError(f"lung grid must be [n_slices, g, g], got {grid.shape}")
    flat = np.asarray(grid, dtype=np.float32).reshape(grid.shape[0], -1).ravel()
    if n_expected is not None and flat.size != n_expected:
        raise ValueError(
            f"lung grid gives {flat.size} patches but the bag has {n_expected}. "
            "The store is at cached-slice resolution and aligns positionally; a "
            "mismatch means the bag and the mask came from different caches."
        )
    return flat > lung_thresh


def restrict_to_lung(scores: np.ndarray, mask: np.ndarray, in_lung: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Drop out-of-lung patches from one bag's ``(scores, target)`` pair.

    Returns ``(scores[in_lung], target[in_lung])`` with the target already
    binarised. Restricting removes roughly three quarters of every bag -- the
    global in-lung fraction under ``method: fill`` is 0.243 -- and it removes
    them from *both* arrays together, which is the whole point: an AUC computed
    over a restricted score population and an unrestricted target would be a
    different quantity wearing H8's name.
    """
    if not (scores.shape[0] == mask.shape[0] == in_lung.shape[0]):
        raise ValueError(
            f"lengths disagree: scores {scores.shape[0]}, mask {mask.shape[0]}, "
            f"in_lung {in_lung.shape[0]}"
        )
    return scores[in_lung], (np.asarray(mask)[in_lung] > 0).astype(np.int8)
