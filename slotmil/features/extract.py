"""Frozen-backbone feature extraction into the HDF5 cache.

Idempotent by construction: a series already present in the cache is skipped.
That matters because these jobs run on preemptible/time-limited SLURM
allocations -- a job that dies at hour 11 of 12 resumes instead of restarting.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch

from .ct_preprocess import mask_to_patch_grid, select_lung_slices, slices_to_batch, window_hu


def cache_has(h5_path: str | Path, uid: str) -> bool:
    if not os.path.exists(h5_path):
        return False
    try:
        with h5py.File(h5_path, "r") as f:
            return uid in f and "features" in f[uid]
    except OSError:
        return False


def cached_uids(h5_path: str | Path) -> set[str]:
    if not os.path.exists(h5_path):
        return set()
    try:
        with h5py.File(h5_path, "r") as f:
            return {k for k in f.keys() if "features" in f[k]}
    except OSError:
        return set()


@torch.no_grad()
def extract_volume(
    backbone,
    volume_hu: np.ndarray,
    device: str = "cuda",
    batch_size: int = 32,
    image_size: int = 224,
    restrict_to_lung: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(features [S, P, D] float16, slice_index [S])``."""
    idx = (
        select_lung_slices(volume_hu)
        if restrict_to_lung
        else np.arange(volume_hu.shape[0])
    )
    vol01 = window_hu(volume_hu[idx])

    feats = []
    for start in range(0, len(idx), batch_size):
        chunk = vol01[start : start + batch_size]
        x = slices_to_batch(chunk, image_size=image_size, device=device)
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16):
            out = backbone(x)
        feats.append(out.to(torch.float16).cpu().numpy())

    return np.concatenate(feats, axis=0), idx


def write_series(
    h5_path: str | Path,
    uid: str,
    features: np.ndarray,
    label: int,
    slice_index: np.ndarray,
    grid: int,
    lesion_mask: np.ndarray | None = None,
    extra_attrs: dict | None = None,
) -> None:
    """Append one series to the cache.

    Opened in 'a' mode per series so a crash mid-run leaves a valid file holding
    everything written so far.
    """
    Path(h5_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "a") as f:
        if uid in f:
            del f[uid]
        g = f.create_group(uid)
        g.create_dataset(
            "features", data=features.astype(np.float16), compression="lzf"
        )
        g.create_dataset("slice_index", data=slice_index.astype(np.int32))
        if lesion_mask is not None:
            g.create_dataset(
                "mask",
                data=mask_to_patch_grid(lesion_mask, grid, mode="mean"),
                compression="lzf",
            )
        g.attrs["label"] = int(label)
        g.attrs["grid_h"] = int(grid)
        g.attrs["grid_w"] = int(grid)
        g.attrs["n_slices"] = int(features.shape[0])
        g.attrs["dim"] = int(features.shape[-1])
        for k, v in (extra_attrs or {}).items():
            g.attrs[k] = v


def cache_size_report(h5_path: str | Path) -> dict:
    """Size / coverage of the cache, for the storage guard."""
    p = Path(h5_path)
    if not p.exists():
        return {"exists": False, "n_series": 0, "bytes": 0}
    with h5py.File(p, "r") as f:
        n = len(f.keys())
        slices = sum(int(f[k].attrs.get("n_slices", 0)) for k in f.keys())
    return {
        "exists": True,
        "n_series": n,
        "n_slices": slices,
        "bytes": p.stat().st_size,
        "gb": round(p.stat().st_size / 1024**3, 2),
    }
