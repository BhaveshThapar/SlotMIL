"""Tests for the lung mask used to restrict the localisation evaluation region.

One failure mode justifies this whole file. A lung mask built by thresholding for
air excludes the nodules themselves -- a nodule is soft tissue, so it fails
``volume < -320``. Restricting a localisation metric to that mask deletes exactly
the targets the metric is scoring, which does not make the control conservative,
it inverts it: attention would be penalised for landing on the lesion.

Nothing crashes when that happens. The numbers just come out wrong, in the
direction that looks like a finding. So the phantom below plants two nodules
that an air threshold loses -- one enclosed by parenchyma, one fused with the
chest wall -- and pins which method recovers which.
"""

from __future__ import annotations

import numpy as np
import pytest

from slotmil.features.ct_preprocess import (
    lung_mask_3d,
    lung_mask_for_evaluation,
    mask_to_patch_grid,
)

BODY_HU, AIR_HU, NODULE_HU = 0.0, -800.0, 50.0


def _disk(shape, cy, cx, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def phantom(n_slices: int = 6, size: int = 64):
    """A body with two air-filled lungs and two nodules an air mask would lose.

    ``solitary`` sits fully inside the left lung -- a hole in the air region.
    ``juxtapleural`` sits against the left lung's outer wall and touches it, so
    it is a notch open to the chest wall rather than an enclosed hole, which is
    what makes it the harder case: hole filling cannot reach it.
    """
    vol = np.full((n_slices, size, size), BODY_HU, dtype=np.float32)
    left = _disk((size, size), 32, 20, 12)
    right = _disk((size, size), 32, 44, 12)
    solitary = _disk((size, size), 32, 20, 3)
    juxtapleural = _disk((size, size), 32, 11, 3)

    for z in range(n_slices):
        vol[z][left | right] = AIR_HU
        vol[z][solitary] = NODULE_HU
        vol[z][juxtapleural] = NODULE_HU

    lesions = np.zeros((n_slices, size, size), dtype=bool)
    lesions[:, solitary] = True
    lesions[:, juxtapleural] = True
    return vol, solitary, juxtapleural, lesions


def contained(mask3d, lesion2d) -> float:
    """Fraction of a lesion's voxels that fall inside the mask, over all slices."""
    return float(mask3d[:, lesion2d].mean())


class TestAirThresholdLosesNodules:
    """The motivating bug, pinned so it cannot come back through a refactor."""

    def test_air_mask_loses_a_solitary_nodule(self):
        """Not exactly zero -- the 3x3x3 closing nibbles the rim of the hole --
        but nowhere near enough to evaluate against."""
        vol, solitary, _, _ = phantom()
        assert contained(lung_mask_3d(vol), solitary) < 0.2

    def test_air_mask_loses_a_juxtapleural_nodule(self):
        vol, _, juxta, _ = phantom()
        assert contained(lung_mask_3d(vol), juxta) < 0.2

    def test_air_method_is_the_unchanged_baseline(self):
        vol, _, _, _ = phantom()
        np.testing.assert_array_equal(
            lung_mask_for_evaluation(vol, method="air"), lung_mask_3d(vol)
        )


class TestEndSliceErosion:
    """lung_mask_3d closes with a 3x3x3 element, and binary erosion reads outside
    the volume as background -- so the first and last slices come back empty.

    Harmless when the mask only picks a slice range, which is all it was built
    for. Fatal for an evaluation region: an empty slice holds no lung patches, so
    every lesion patch on it would be silently dropped from the denominator."""

    def test_the_artefact_is_real(self):
        vol, _, _, _ = phantom()
        m = lung_mask_3d(vol)
        assert not m[0].any() and not m[-1].any()
        assert m[1].any()

    def test_evaluation_mask_repairs_it(self):
        vol, _, _, _ = phantom()
        m = lung_mask_for_evaluation(vol, method="fill")
        assert m[0].any() and m[-1].any()

    def test_repair_reaches_lesions_on_the_end_slices(self):
        vol, solitary, _, _ = phantom()
        m = lung_mask_for_evaluation(vol, method="fill")
        assert m[0][solitary].all() and m[-1][solitary].all()


class TestGrowingTheMaskRecoversThem:
    def test_fill_recovers_the_solitary_nodule(self):
        vol, solitary, _, _ = phantom()
        m = lung_mask_for_evaluation(vol, method="fill")
        assert contained(m, solitary) == pytest.approx(1.0)

    def test_fill_alone_does_not_recover_the_juxtapleural_one(self):
        """It is a concavity fused with the chest wall, not an enclosed hole --
        which is why hole filling is not sufficient and the hull step exists."""
        vol, _, juxta, _ = phantom()
        assert contained(lung_mask_for_evaluation(vol, method="fill"), juxta) < 0.5

    def test_hull_recovers_both(self):
        vol, solitary, juxta, _ = phantom()
        m = lung_mask_for_evaluation(vol, method="fill_hull")
        assert contained(m, solitary) == pytest.approx(1.0)
        assert contained(m, juxta) > 0.95, (
            "the hull is what makes juxtapleural nodules evaluable; without it "
            "the control silently drops them"
        )

    def test_hull_still_excludes_most_of_the_body(self):
        """Over-inclusion is the safe direction, but a mask covering everything
        would make the restriction meaningless rather than merely weak."""
        vol, _, _, _ = phantom()
        assert lung_mask_for_evaluation(vol, method="fill_hull").mean() < 0.5

    def test_masks_are_nested_by_construction(self):
        vol, _, _, _ = phantom()
        air = lung_mask_for_evaluation(vol, method="air")
        fill = lung_mask_for_evaluation(vol, method="fill")
        hull = lung_mask_for_evaluation(vol, method="fill_hull")
        assert (fill | air == fill).all()
        assert (hull | fill == hull).all()

    def test_unknown_method_raises(self):
        vol, _, _, _ = phantom()
        with pytest.raises(ValueError, match="air|fill|fill_close|fill_hull"):
            lung_mask_for_evaluation(vol, method="convex")


class TestPatchGridAlignment:
    def test_lung_and_lesion_land_on_the_same_grid(self):
        """The lung mask goes through the same rasteriser as the lesion mask, so
        'lesion patch' and 'lung patch' are directly comparable rather than
        related by an interpolation artefact."""
        vol, _, _, lesions = phantom()
        lung = lung_mask_for_evaluation(vol, method="fill_hull").astype(np.float32)
        lung_g = mask_to_patch_grid(lung, 16, mode="mean")
        lesion_g = mask_to_patch_grid(lesions.astype(np.float32), 16, mode="mean")
        assert lung_g.shape == lesion_g.shape == (vol.shape[0], 16, 16)

    def test_every_lesion_patch_is_a_lung_patch_under_the_hull(self):
        """This is the acceptance gate scripts/lung_mask_lidc.py enforces on real
        data, in miniature."""
        vol, _, _, lesions = phantom()
        lung = lung_mask_for_evaluation(vol, method="fill_hull").astype(np.float32)
        lung_g = mask_to_patch_grid(lung, 16, mode="mean")
        lesion_g = mask_to_patch_grid(lesions.astype(np.float32), 16, mode="mean")
        les = lesion_g > 0
        assert les.any()
        assert ((lung_g > 0.5) & les).sum() / les.sum() == pytest.approx(1.0)
