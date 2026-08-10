"""MosMedData COVID-19 chest CT (plan.md line 64).

1,110 volumes across 5 severity classes, of which **50 carry expert masks of
ground-glass opacification and consolidation** -- two canonical, visually distinct
COVID findings, which makes them near-ideal targets for testing whether separate
slots bind separate findings.

Acquisition note: there is no unauthenticated route to this data. mosmed.ai
requires registration, and no HuggingFace or Zenodo mirror of the full release
exists (checked). Kaggle mirrors do:

    mathurinache/mosmeddata-chest-ct-scans-with-covid19   ~11.9 GB, all 1110
    andrewmvd/mosmed-covid19-ct-scans                     ~1.8 GB, includes masks

Either drop a Kaggle token at ~/.kaggle/kaggle.json and run
scripts/download_mosmed.py, or download manually and unpack to ``root``.

Two caveats worth carrying into the paper: the public release keeps only every
10th slice (~8 mm spacing), so bags are short and z-resolution is coarse; and the
licence is CC BY-NC-ND 3.0 (non-commercial, no-derivatives).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# CT-0 normal .. CT-4 severe. Counts from Morozov et al.: 254/684/125/45/2.
SEVERITY_CLASSES = ["CT-0", "CT-1", "CT-2", "CT-3", "CT-4"]

# The two annotated findings. Index order defines the finding axis used by
# eval.alignment, so it must stay fixed once an assignment has been frozen.
FINDINGS = ["ground_glass", "consolidation"]


class MosMedIndex:
    """Locates MosMed volumes and masks on disk, tolerating mirror layouts.

    Kaggle mirrors vary in directory naming, so this globs rather than assuming
    one structure, and reports clearly when nothing is found instead of failing
    deep inside a DataLoader worker.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"MosMed root {self.root} does not exist.\n"
                "Obtain the data first:\n"
                "  - put a Kaggle token at ~/.kaggle/kaggle.json and run "
                "scripts/download_mosmed.py, or\n"
                "  - register at mosmed.ai, download COVID19_1110, unpack here."
            )
        self.volumes = self._index_volumes()
        self.masks = self._index_masks()

    def _index_volumes(self) -> dict[str, dict]:
        out = {}
        for cls_idx, cls in enumerate(SEVERITY_CLASSES):
            for pattern in (f"**/{cls}/*.nii*", f"**/{cls.replace('-', '_')}/*.nii*"):
                for p in self.root.glob(pattern):
                    out[p.stem.replace(".nii", "")] = {
                        "path": p,
                        "label": cls_idx,
                        "severity": cls,
                    }
        return out

    def _index_masks(self) -> dict[str, Path]:
        out = {}
        for pattern in ("**/masks/*.nii*", "**/*mask*/*.nii*"):
            for p in self.root.glob(pattern):
                out[p.stem.replace(".nii", "").replace("_mask", "")] = p
        return out

    def summary(self) -> dict:
        by_class: dict[str, int] = {}
        for v in self.volumes.values():
            by_class[v["severity"]] = by_class.get(v["severity"], 0) + 1
        return {
            "n_volumes": len(self.volumes),
            "n_masked": len(self.masks),
            "by_class": by_class,
            "root": str(self.root),
        }


def load_volume(path: str | Path) -> np.ndarray:
    """Read a NIfTI volume as ``[S, H, W]`` in HU."""
    import nibabel as nib

    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj).astype(np.float32)
    return np.transpose(arr, (2, 0, 1))  # (H, W, S) -> (S, H, W)


def load_mask(path: str | Path) -> np.ndarray:
    """Read a binary lesion mask as ``[S, H, W]``.

    The public masks are a single binary channel covering both findings rather
    than one channel per finding. Splitting GGO from consolidation therefore has
    to be done by HU thresholding within the mask (consolidation is denser), which
    is an approximation and should be reported as one.
    """
    m = load_volume(path)
    return (m > 0.5).astype(np.uint8)


def split_findings_by_hu(
    volume_hu: np.ndarray, mask: np.ndarray, consolidation_hu: float = -200.0
) -> np.ndarray:
    """Approximate a 2-channel finding mask from the single binary mask.

    Ground glass is hazy (lower attenuation); consolidation is dense
    (higher). The threshold is a convention, not a ground truth -- any result
    resting on this split needs that caveat stated explicitly.

    Returns ``[2, S, H, W]`` ordered as :data:`FINDINGS`.
    """
    inside = mask.astype(bool)
    consolidation = inside & (volume_hu >= consolidation_hu)
    ground_glass = inside & ~consolidation
    return np.stack([ground_glass, consolidation]).astype(np.uint8)
