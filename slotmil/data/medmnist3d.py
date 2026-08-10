"""MedMNIST3D bags -- the W1 smoke test (plan.md line 132).

Public, auto-downloading, no credentials. NoduleMNIST3D is derived from LIDC
(benign/malignant, official splits 1158/165/310) and the published ResNet-18(3D)
reference of 0.879 AUC / 84.5% ACC is the bar the go/no-go is measured against.

Bag construction mirrors the CT pipeline: each axial slice of the volume is an
instance, so the same MIL machinery runs here as on LIDC -- just small enough to
iterate on in minutes.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

MEDMNIST3D_FLAGS = {
    "nodulemnist3d": {"n_classes": 2, "task": "binary-class"},
    "organmnist3d": {"n_classes": 11, "task": "multi-class"},
    "fracturemnist3d": {"n_classes": 3, "task": "multi-class"},
    "adrenalmnist3d": {"n_classes": 2, "task": "binary-class"},
    "vesselmnist3d": {"n_classes": 2, "task": "binary-class"},
    "synapsemnist3d": {"n_classes": 2, "task": "binary-class"},
}


class MedMNIST3DBags(Dataset):
    """Axial slices of a MedMNIST3D volume as MIL instances.

    Args:
        flag: dataset name, e.g. ``"nodulemnist3d"``.
        split: ``train`` | ``val`` | ``test`` (official MedMNIST splits).
        size: 28 or 64. 64 gives the slice resolution DINOv2 needs to be
            meaningful after upsampling.
        root: download directory.
        as_slices: True yields ``[D, H*W]`` flattened slice instances for the
            end-to-end fast path; False yields the raw ``[1, D, H, W]`` volume.
    """

    def __init__(
        self,
        flag: str = "nodulemnist3d",
        split: str = "train",
        size: int = 64,
        root: str | None = None,
        as_slices: bool = True,
        download: bool = True,
    ):
        import medmnist
        from medmnist import INFO

        if flag not in MEDMNIST3D_FLAGS:
            raise ValueError(f"{flag!r} is not a MedMNIST3D dataset")

        info = INFO[flag]
        cls = getattr(medmnist, info["python_class"])
        self.ds = cls(split=split, root=root, size=size, download=download)
        self.flag = flag
        self.split = split
        self.size = size
        self.as_slices = as_slices
        self.n_classes = len(info["label"])
        self.task = info["task"]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> dict:
        vol, label = self.ds[i]  # vol: (1, D, H, W) or (D, H, W)
        vol = np.asarray(vol, dtype=np.float32)
        if vol.ndim == 4:
            vol = vol[0]
        vol = vol / 255.0 if vol.max() > 1.5 else vol

        x = torch.from_numpy(vol)  # D, H, W
        label = int(np.asarray(label).reshape(-1)[0])

        if self.as_slices:
            instances = x.reshape(x.shape[0], -1)  # D, H*W
        else:
            instances = x.unsqueeze(0)

        return {
            "features": instances,
            "label": torch.tensor(label),
            "uid": f"{self.flag}_{self.split}_{i}",
            "n_slices": x.shape[0],
            "volume_shape": tuple(x.shape),
        }


def medmnist_collate(batch: list[dict]) -> dict:
    """MedMNIST3D volumes are fixed-size, so no ragged padding is needed -- but
    the pad mask is still emitted so downstream code takes one path only."""
    feats = torch.stack([b["features"] for b in batch])
    n = feats.shape[1]
    return {
        "features": feats,
        "pad_mask": torch.ones(len(batch), n, dtype=torch.bool),
        "label": torch.stack([b["label"] for b in batch]),
        "uid": [b["uid"] for b in batch],
        "n_slices": torch.tensor([b["n_slices"] for b in batch]),
        "lengths": torch.tensor([n] * len(batch)),
    }
