"""Mask-only adapters for the confound atlas.

The atlas asks one question of a public dataset: where does the positional
confound live? Answering it needs only lesion masks and patch geometry --
``fit_family(source="masks")`` and ``score_family`` never read an image -- so
these adapters never download images, never touch a GPU, and never train.

Every adapter yields ``(case_id, patient_id, masks_flat)`` where
``masks_flat`` is the case's lesion mask rasterised to the same
``16x16``-per-slice patch grid the LIDC cache uses
(:func:`slotmil.features.ct_preprocess.mask_to_patch_grid`), flattened in
slice-major C order -- the layout the feature cache, the attention dumps and
``slotmil.eval.lung`` all share, and the one a positional zip silently
mis-scores if it is ever violated. Bags are ``n_slices * 256`` by
construction, so ``per_bag_axes``'s whole-multiple drop rule never fires.

Orientation: every volume is reoriented to RAS with
``nibabel.as_closest_canonical`` and sliced along the third (z) axis, so
"axial" means the same anatomical axis in every dataset. The grid is pinned
at 16 on purpose: per-dataset profiles are comparable with the LIDC/MosMed
columns of the paper's dataset-contrast figure only if the geometry matches.

Lesion semantics are lesion-only throughout, mirroring the "finding"
semantics of LIDC (nodule) and MosMed (infection): organ labels are excluded
(KiTS kidney = 1, LiTS liver = 1 are dropped; tumour = 2 kept).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from slotmil.features.ct_preprocess import mask_to_patch_grid

GRID = 16  # pinned to the LIDC/MosMed geometry; see module docstring

Case = tuple[str, str, np.ndarray]


def _canonical_slices(path: Path) -> np.ndarray:
    """One NIfTI label volume -> ``(S, H, W)`` array, axial slices first."""
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(str(path)))
    vol = np.asanyarray(img.dataobj)
    if vol.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D label volume, got {vol.shape}")
    # RAS canonical: axes are (x, y, z); axial slices stack along z.
    return np.transpose(vol, (2, 1, 0))


def _flat_grid(lesion: np.ndarray) -> np.ndarray:
    """Binary ``(S, H, W)`` -> flat ``[S * GRID * GRID]`` float32 fractions."""
    return mask_to_patch_grid(lesion.astype(np.float32), GRID,
                              mode="mean").reshape(-1)


def _iter_niftis(files: list[Path], case_of: Callable[[Path], str],
                 lesion_of: Callable[[np.ndarray], np.ndarray]
                 ) -> Iterator[Case]:
    for f in sorted(files):
        case = case_of(f)
        vol = _canonical_slices(f)
        yield case, case, _flat_grid(lesion_of(vol))


def covid_ct_seg(root: Path) -> Iterator[Case]:
    """COVID-19-CT-Seg (Zenodo 3757476): 20 cases, infection masks.

    ``root`` holds ``Infection_Mask/*.nii.gz``; any positive voxel is
    infection.
    """
    files = [f for f in (root / "Infection_Mask").glob("*.nii*")
             if not f.name.startswith("._")]
    if not files:
        raise SystemExit(f"{root}/Infection_Mask holds no NIfTI masks")
    return _iter_niftis(files, lambda f: f.name.split(".nii")[0],
                        lambda v: v > 0)


def kits19(root: Path) -> Iterator[Case]:
    """KiTS19: ``case_*/segmentation.nii.gz``; kidney = 1, tumour = 2.

    Lesion-only: label 2. The masks ship in the kits19 git repository; the
    imaging never needs downloading.
    """
    files = list(root.glob("case_*/segmentation.nii.gz"))
    if not files:
        raise SystemExit(f"{root} holds no case_*/segmentation.nii.gz")
    return _iter_niftis(files, lambda f: f.parent.name, lambda v: v == 2)


def msd_task06(root: Path) -> Iterator[Case]:
    """MSD Task06 Lung: ``labelsTr/lung_*.nii.gz``; tumour = 1."""
    files = [f for f in (root / "labelsTr").glob("lung_*.nii*")
             if not f.name.startswith("._")]
    if not files:
        raise SystemExit(f"{root}/labelsTr holds no lung_*.nii*")
    return _iter_niftis(files, lambda f: f.name.split(".nii")[0],
                        lambda v: v > 0)


def lits(root: Path) -> Iterator[Case]:
    """LiTS: ``segmentation-*.nii``; liver = 1, tumour = 2. Lesion-only."""
    files = list(root.glob("segmentation-*.nii*"))
    if not files:
        raise SystemExit(f"{root} holds no segmentation-*.nii*")
    return _iter_niftis(
        files, lambda f: f.name.split(".nii")[0].replace("segmentation-", "case"),
        lambda v: v == 2)


ADAPTERS: dict[str, Callable[[Path], Iterator[Case]]] = {
    "covid_ct_seg": covid_ct_seg,
    "kits19": kits19,
    "msd_task06": msd_task06,
    "lits": lits,
}
