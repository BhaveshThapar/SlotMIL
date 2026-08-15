"""HDF5-backed bag dataset over cached frozen-backbone features.

This is the engineering asset the whole project rests on (plan.md line 158):
extract DINOv2 features once, then every training run and every ablation reads
from here. It is what turns the study into single-GPU work.

Layout, one group per CT series::

    /<series_uid>/features   float16  [n_slices, n_patches, D]
    /<series_uid>/mask       uint8    [n_slices, grid_h, grid_w]   (optional)
    /<series_uid>.attrs      label, grid_h, grid_w, slice_index, ...

Instances are *patch tokens*, flattened to N = n_slices * n_patches. Patch-level
granularity is what makes Dice/IoU against 3D lesion masks meaningful -- a
slice-level bag can only ever tell you *which slice*, not where in it. The
cheaper slice-level mode is kept as an ablation.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureBagDataset(Dataset):
    """Bags of cached patch-token features.

    Args:
        h5_path: cache file.
        keys: series UIDs in this split. ``None`` uses every group in the file.
        labels: mapping uid -> label. ``None`` reads the ``label`` attr.
        instance_level: ``"patch"`` (default) or ``"slice"`` (mean-pool patches
            within each slice).
        max_slices: stochastic slice subsampling during training for very long
            volumes (plan.md line 88). ``None`` or eval mode uses every slice.
        train: whether to subsample. Inference always sees the full bag.
    """

    def __init__(
        self,
        h5_path: str | Path,
        keys: list[str] | None = None,
        labels: dict[str, int] | None = None,
        instance_level: str = "patch",
        max_slices: int | None = None,
        train: bool = False,
        return_mask: bool = False,
        seed: int = 0,
    ):
        if instance_level not in ("patch", "slice"):
            raise ValueError(f"instance_level must be patch|slice, got {instance_level!r}")
        self.h5_path = str(h5_path)
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(
                f"feature cache not found: {self.h5_path}. Run "
                "scripts/extract_features.py (or the staged LIDC pipeline) first."
            )
        self.instance_level = instance_level
        self.max_slices = max_slices
        self.train = train
        self.return_mask = return_mask
        self.labels = labels
        self._rng = np.random.default_rng(seed)
        self._h5: h5py.File | None = None  # opened lazily, per worker

        with h5py.File(self.h5_path, "r") as f:
            self.keys = list(keys) if keys is not None else sorted(f.keys())
            missing = [k for k in self.keys if k not in f]
            if missing:
                raise KeyError(
                    f"{len(missing)} keys absent from cache (first: {missing[:3]})"
                )
            probe = f[self.keys[0]]
            self.dim = int(probe["features"].shape[-1])
            self.grid_h = int(probe.attrs["grid_h"])
            self.grid_w = int(probe.attrs["grid_w"])

    def masked_keys(self) -> list[str]:
        """Series in this split that carry a lesion mask.

        Alignment and localisation are only defined on annotated bags, so the
        callers filter to these rather than iterating the whole split. On MosMed
        that is 50 of 1110; on LIDC it is essentially all of them.
        """
        with h5py.File(self.h5_path, "r") as f:
            return [k for k in self.keys if "mask" in f[k]]

    def _file(self) -> h5py.File:
        # One handle per DataLoader worker; sharing across processes corrupts reads.
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int) -> dict:
        uid = self.keys[i]
        g = self._file()[uid]
        feats = g["features"]  # n_slices, n_patches, D
        n_slices = feats.shape[0]

        if self.train and self.max_slices and n_slices > self.max_slices:
            sel = np.sort(self._rng.choice(n_slices, self.max_slices, replace=False))
        else:
            sel = np.arange(n_slices)

        # Read slice-by-slice rather than with h5py fancy indexing.
        #
        # This looks like a pessimisation and is the opposite. The dataset is lzf
        # compressed with chunks spanning 11 slices, and h5py's fancy indexing
        # (`feats[sel]`) re-reads and re-decompresses overlapping chunks once per
        # selected index. Measured on this cache, selecting 48 of 171 slices:
        #
        #     feats[sel]                       6832 ms / 6 bags
        #     np.stack([feats[i] for i in sel]) 409 ms / 6 bags   (16.7x faster)
        #     feats[:][sel]                     476 ms / 6 bags
        #
        # All three return byte-identical arrays. Fancy indexing was also ~3x
        # slower than reading the *entire* volume, so slice subsampling -- meant
        # to make training cheaper -- was making the read strictly worse.
        if len(sel) == n_slices:
            arr = feats[:]  # contiguous read; nothing to skip
        else:
            arr = np.stack([feats[i] for i in sel])
        x = torch.from_numpy(arr.astype(np.float32))  # S, P, D

        mask = None
        if self.return_mask and "mask" in g:
            # Same fancy-indexing pathology as the features above; masks are
            # also lzf-compressed and chunked.
            md = g["mask"]
            marr = md[:] if len(sel) == md.shape[0] else np.stack([md[i] for i in sel])
            m = torch.from_numpy(marr.astype(np.float32))  # S, gh, gw
            mask = m.reshape(m.shape[0], -1)  # S, P

        if self.instance_level == "slice":
            x = x.mean(dim=1)  # S, D
            if mask is not None:
                mask = mask.mean(dim=1)  # S
        else:
            x = x.reshape(-1, x.shape[-1])  # S*P, D
            if mask is not None:
                mask = mask.reshape(-1)  # S*P

        label = (
            self.labels[uid] if self.labels is not None else int(g.attrs["label"])
        )

        item = {
            "features": x,
            "label": torch.tensor(label),
            "uid": uid,
            "n_slices": len(sel),
            "slice_index": torch.from_numpy(sel.astype(np.int64)),
        }
        if mask is not None:
            item["patch_target"] = mask
        return item

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


def collate_bags(batch: list[dict]) -> dict:
    """Pad variable-length bags and build the boolean pad mask.

    Padded feature rows are written as exact zeros. SlotAttention masks ``v``
    anyway, but leaving junk here would make the padding-invariance test depend
    on luck rather than on the masking being right.
    """
    b = len(batch)
    lengths = [item["features"].shape[0] for item in batch]
    n_max = max(lengths)
    dim = batch[0]["features"].shape[-1]

    feats = torch.zeros(b, n_max, dim)
    pad_mask = torch.zeros(b, n_max, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = lengths[i]
        feats[i, :n] = item["features"]
        pad_mask[i, :n] = True

    # True anatomical slice numbers, padded with -1. Carried because
    # ``--max-slices`` subsampling makes bag position k the true slice sel[k]:
    # without this, losses.normal_guidance_loss would compute its moments and
    # apply its 1-slice^2 variance floor in *subsampled* units, i.e. a random
    # per-bag multiple of the anatomical scale (~3.6x at S=171, ~14.6x at S=700),
    # so lam would silently mean something different for every bag.
    #
    # Only the loss reads it. MILModel.forward is deliberately untouched, which
    # is what keeps the nine `model(feats, pad_mask)` call sites across
    # eval_alignment / null_collect_attn / untrained_floor / reeval_all_alignment
    # / make_figures / audit_independent / faithfulness working unchanged.
    s_max = max(item["slice_index"].shape[0] for item in batch)
    slice_index = torch.full((b, s_max), -1, dtype=torch.long)
    for i, item in enumerate(batch):
        si = item["slice_index"]
        slice_index[i, : si.shape[0]] = si

    out = {
        "features": feats,
        "pad_mask": pad_mask,
        "label": torch.stack([item["label"] for item in batch]),
        "uid": [item["uid"] for item in batch],
        "n_slices": torch.tensor([item["n_slices"] for item in batch]),
        "lengths": torch.tensor(lengths),
        "slice_index": slice_index,
    }

    # Keyed off ANY item, not batch[0]. In a partially-annotated cache (MosMed:
    # 50 masked of 1110) a batch whose first item happens to be unmasked would
    # otherwise drop the field entirely, and downstream code that expects it
    # dies with a bare KeyError. Unmasked bags get zeros plus has_mask=False, so
    # callers can tell "no lesion here" from "not annotated".
    if any("patch_target" in item for item in batch):
        tgt = torch.zeros(b, n_max)
        has_mask = torch.zeros(b, dtype=torch.bool)
        for i, item in enumerate(batch):
            if "patch_target" in item:
                tgt[i, : lengths[i]] = item["patch_target"]
                has_mask[i] = True
        out["patch_target"] = tgt
        out["has_mask"] = has_mask

    return out


def attn_to_grid(
    attn: torch.Tensor,
    n_slices: int,
    grid_h: int,
    grid_w: int,
    length: int | None = None,
) -> torch.Tensor:
    """Reshape flat slot attention ``[K, N]`` back to ``[K, n_slices, gh, gw]``.

    The inverse of the flatten in ``__getitem__``; used by every localisation
    metric and by the overlay figures.
    """
    k = attn.shape[0]
    n = length if length is not None else attn.shape[-1]
    expected = n_slices * grid_h * grid_w
    if n != expected:
        raise ValueError(
            f"attention length {n} != n_slices*gh*gw = {expected}. Either the bag "
            "was slice-level, or slice subsampling was active and n_slices is stale."
        )
    return attn[:, :n].reshape(k, n_slices, grid_h, grid_w)
