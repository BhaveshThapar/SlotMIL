"""Content-free references, factorised by the axis they are allowed to use.

The question this module exists to answer: **which axis does a content-free
baseline need in order to score well on volumetric CT?**

The literature answers it one axis at a time and disagrees without noticing.
Harvey et al. (arXiv:2604.26807, arXiv:2605.27306) use a centred Gaussian over
the *slice index* -- axial only, no in-plane term, because their labels are
per-slice and they have no voxel masks. Arun et al. (*Radiology: AI* 2021) use a
mean training mask -- in-plane, applied per slice. Both report that the
content-free baseline is hard to beat. Neither can say which of the two axes is
carrying it, because neither measured both.

``nulls.global_template`` already gives the in-plane half, and it has a property
worth stating outright: it is 256 numbers tiled identically across every slice,
so its **slice-level AUC is exactly 0.5 by construction** -- every slice's mean
score is the same number, every pair ties, and ``roc_auc_score`` credits ties at
0.5. It carries provably zero axial information. Measured on the discovery
dumps (f32_seed0, 129 bags) it nonetheless reaches:

    scorer                       flat     slice    within
    slot attention (trained)   0.8423    0.4822    0.8409
    fitted in-plane template   0.7865    0.5000    0.7884
    model-free centre prior    0.7752    0.6026    0.7351

That is the paper's claim in three rows. A reference with no axial term at all
reaches 93% of the trained network's flat AUC, and is *higher* within-slice than
flat -- while on the axial axis the trained network (0.4822) sits below even the
tie floor, and far below model-free geometry (0.6026). The prior lives in-plane;
the published corrections act axially.

This module completes the 2x2 so the claim is a decomposition rather than a
juxtaposition. Each reference is fit on validation and applied unchanged to
every test bag, so none of them can read a test image:

============  =========================  ============  ==============
reference     fitted object              axial info    in-plane info
============  =========================  ============  ==============
axial         ``[n_bins]`` depth profile yes           **none**
in-plane      ``[n_patch]`` map          **none**      yes
separable     outer product of the two   yes           yes
joint         ``[n_bins, n_patch]``      yes           yes
============  =========================  ============  ==============

``axial`` is the *fitted* generalisation of Harvey's fixed sigma=1 Gaussian and
should beat it, which is the point: a stronger reference lowers everyone's
reported skill, including ours.

**``joint`` does NOT dominate the others on held-out data, and assuming it would
is the trap here.** It nests them in the *fit* -- ``axial``, ``inplane`` and
``separable`` are all restrictions of it -- but these are plug-in estimates
scored by held-out ranking, not predictors optimised for AUC, so more cells
means more variance per cell, not more skill. Measured (mask-fit, 149 validation
bags, discovery dumps): ``joint`` 0.6839 flat against ``inplane`` 0.8236 and
``separable`` 0.8501. At a lesion-patch prevalence of 0.00068, 32 x 256 = 8192
joint cells are estimated from very few positive events each and the estimate
overfits the fit split outright.

Two consequences. First, do not describe ``joint`` as an upper bound anywhere.
Second, **the denominator for ``estimands.prior_normalised_skill`` must be
declared in advance, not chosen as the family maximum on test** -- the null
battery already measured what best-of-k selection on test buys here (+0.11 AUC
from pure noise), and picking the strongest of four references after seeing
their test scores is that same error wearing a decomposition. The
pre-registered choice is ``inplane``.

**Depth is normalised, not raw.** Bags differ in ``n_slices`` (LIDC ranges
roughly 65-700), so a profile indexed by raw slice number is not comparable
across bags and is exactly the representability problem that makes Harvey's
printed formula uncomputable here (see ``AMENDMENTS.md``, 2026-08-15). Slice
``j`` of ``ns`` falls in bin ``min(floor(j * n_bins / ns), n_bins - 1)``.

**Two fit sources, and conflating them would be the easy mistake here.**

``source="attention"`` is the pre-registered convention: the object being tiled
is the arm's own average attention on validation, so ``inplane_template``
reproduces ``nulls.global_template`` (identically in float64; that function
casts its return to float32, so the two agree to ~5e-8 rather than bit-for-bit)
and the family is a strict generalisation of the declared reference. It answers
an arm-relative question --
*how much of this arm's score is reproducible by a content-free rearrangement of
its own average behaviour?*

``source="masks"`` fits validation lesion frequency instead. It is needed
because an attention-fitted **axial** template inherits the arm's axial
blindness: LIDC arms score 0.4822-0.5565 on the slice axis, so averaging their
attention over depth yields a near-flat profile, and the resulting reference
scores at chance. Reporting that as "the axial axis carries no information"
would be wrong -- model-free geometry reaches slice AUC 0.6026, so axial signal
demonstrably exists; what the attention-fitted version shows is that *this arm
found none*. The mask-fitted family removes the confound. It reads labels on the
fit split and never a test image, so it stays content-free at test time, and it
is strictly stronger than Arun et al.'s unfit average mask and than Harvey's
fixed sigma=1 Gaussian -- the best any content-free method could do here. If the
axial axis still fails to lift flat AUC against *that*, the finding is about the
metric rather than about any model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .nulls import GRID, N_PATCH

__all__ = [
    "DEFAULT_DEPTH_BINS",
    "TemplateFit",
    "attention_rows",
    "lesion_rows",
    "axial_template",
    "inplane_template",
    "joint_template",
    "depth_bins_for",
    "axial_scores",
    "inplane_scores",
    "separable_scores",
    "joint_scores",
    "fit_family",
    "score_family",
]

# 32 bins against a median LIDC volume of ~171 slices is ~5 slices per bin, and
# the smallest volumes (~65 slices) still put 2 slices in every bin. The fit sees
# 149 validation bags x n_slices x 256 patches -- on the order of 6e6 values --
# so 32 x 256 = 8192 cells is not a small-sample regime. Raising this makes the
# joint template strictly stronger and the reported skill strictly lower, so the
# conservative direction is *more* bins, not fewer.
DEFAULT_DEPTH_BINS = 32


class TemplateFit:
    """A fitted content-free reference plus what it could not observe.

    ``n_unobserved`` is carried rather than silently filled because an unobserved
    cell means the fit had nothing to say about that depth/position and is
    falling back to the global mean. Zero is expected on LIDC; anything else is
    a signal that ``n_bins`` is too high for the volumes in the fit split.
    """

    def __init__(self, kind: str, values: np.ndarray, n_bins: int,
                 n_patch: int, n_unobserved: int, n_bags: int):
        self.kind = kind
        self.values = values
        self.n_bins = n_bins
        self.n_patch = n_patch
        self.n_unobserved = n_unobserved
        self.n_bags = n_bags

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (f"TemplateFit({self.kind!r}, shape={self.values.shape}, "
                f"n_bags={self.n_bags}, n_unobserved={self.n_unobserved})")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "shape": list(self.values.shape),
                "n_bins": self.n_bins, "n_patch": self.n_patch,
                "n_unobserved": self.n_unobserved, "n_bags": self.n_bags}


def depth_bins_for(n_slices: int, n_bins: int = DEFAULT_DEPTH_BINS) -> np.ndarray:
    """``[n_slices]`` bin index over *normalised* depth.

    Clipped at the top end because ``j = ns - 1`` maps to exactly ``n_bins``
    before flooring only when ``ns`` divides evenly, and an off-by-one there
    would put the last slice of every volume in a bin that the fit never
    populated.
    """
    if n_slices <= 0:
        raise ValueError(f"n_slices must be positive, got {n_slices}")
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    j = np.arange(n_slices)
    return np.minimum((j * n_bins) // n_slices, n_bins - 1).astype(np.int64)


def _whole_slices(attn_row: np.ndarray, mask: np.ndarray, n_patch: int):
    """``[ns, n_patch]`` view of one bag, or ``None`` for a ragged tail.

    Truncates to whole slices rather than skipping the bag, matching
    ``nulls.global_template``: a bag with a trailing partial slice still
    contributes its complete ones. Real bags are exact multiples, so this only
    bites on synthetic inputs -- but a silent divergence from the function this
    family generalises would make the four references incomparable, which is the
    one thing they must not be.
    """
    ns = len(mask) // n_patch
    if ns == 0:
        return None
    return attn_row[: ns * n_patch].reshape(ns, n_patch), ns


def attention_rows(attns, slot: int):
    """``[N]`` per bag: the arm's own attention. The declared fit source."""
    return [np.asarray(a[slot], dtype=np.float64) for a in attns]


def lesion_rows(masks):
    """``[N]`` per bag: the binarised validation lesion mask.

    The **oracle** content-free source, and the reason it exists is a confound
    that only shows up once you measure both axes.

    Fit to *attention*, an axial template can be no better than the arm's own
    average axial behaviour -- and on LIDC the arms have essentially none (slice
    AUC 0.4822-0.5565). An attention-fitted axial reference therefore scores at
    chance, which looks like "the axial axis carries no information" when what it
    actually shows is "this arm found none". Those are different claims and only
    one of them is ours to make: model-free geometry reaches slice AUC 0.6026,
    so axial signal demonstrably exists.

    Fitting to validation *masks* removes the confound. It reads labels on the
    fit split and never a test image, so it is still content-free at test time,
    and it is strictly stronger than both Arun et al.'s unfit average mask and
    Harvey's fixed sigma=1 Gaussian -- it is the best any content-free method
    could do on this data. If the axial axis still fails to lift flat AUC
    against *this* reference, that is a statement about the metric rather than
    about any model.
    """
    return [(np.asarray(m) > 0).astype(np.float64) for m in masks]


def _accumulate(rows, masks, n_patch: int, n_bins: int):
    """Sum and count of a per-instance quantity per (depth bin, position)."""
    total = np.zeros((n_bins, n_patch), dtype=np.float64)
    count = np.zeros((n_bins, n_patch), dtype=np.int64)
    n_bags = 0
    for r, m in zip(rows, masks):
        got = _whole_slices(r, m, n_patch)
        if got is None:
            continue
        sc, ns = got
        bins = depth_bins_for(ns, n_bins)
        np.add.at(total, bins, sc)
        np.add.at(count, bins, 1)
        n_bags += 1
    if n_bags == 0:
        raise ValueError(
            "no bag had a whole number of slices; cannot fit a template"
        )
    return total, count, n_bags


def _fill(total: np.ndarray, count: np.ndarray):
    """Cell means, with unobserved cells falling back to the global mean."""
    observed = count > 0
    if not observed.any():
        raise ValueError("template fit observed nothing")
    grand = float(total[observed].sum() / count[observed].sum())
    out = np.full(total.shape, grand, dtype=np.float64)
    np.divide(total, count, out=out, where=observed)
    return out, int((~observed).sum())


def inplane_template(rows, masks, n_patch: int = N_PATCH) -> TemplateFit:
    """``[n_patch]``: one mean in-plane map, the declared ``fitted_template``.

    On the attention source this is exactly :func:`nulls.global_template`, and
    :func:`tests.test_templates` pins the two as bit-identical so the in-plane
    cell of the 2x2 cannot drift from the object the pre-registration declares.
    """
    total, count, n_bags = _accumulate(rows, masks, n_patch, n_bins=1)
    values, n_unobserved = _fill(total[0], count[0])
    return TemplateFit("inplane", values, n_bins=1, n_patch=n_patch,
                       n_unobserved=n_unobserved, n_bags=n_bags)


def axial_template(rows, masks, n_patch: int = N_PATCH,
                   n_bins: int = DEFAULT_DEPTH_BINS) -> TemplateFit:
    """``[n_bins]``: mean per normalised-depth bin, averaged in-plane.

    The fitted generalisation of Harvey's centred Gaussian. Unlike theirs it is
    neither unimodal nor symmetric by construction, so if an axial prior is
    there it has every opportunity to appear.
    """
    total, count, n_bags = _accumulate(rows, masks, n_patch, n_bins)
    values, n_unobserved = _fill(total.sum(axis=1), count.sum(axis=1))
    return TemplateFit("axial", values, n_bins=n_bins, n_patch=n_patch,
                       n_unobserved=n_unobserved, n_bags=n_bags)


def joint_template(rows, masks, n_patch: int = N_PATCH,
                   n_bins: int = DEFAULT_DEPTH_BINS) -> TemplateFit:
    """``[n_bins, n_patch]``: the least constrained member of the family.

    It nests the other three *in the fit* and beats none of them on held-out
    ranking -- 8192 cells at a lesion-patch prevalence of 0.00068 overfit the fit
    split. See the module docstring: it is not an upper bound and must not be
    used as one, and the skill denominator is declared in advance rather than
    selected from this family on test.
    """
    total, count, n_bags = _accumulate(rows, masks, n_patch, n_bins)
    values, n_unobserved = _fill(total, count)
    return TemplateFit("joint", values, n_bins=n_bins, n_patch=n_patch,
                       n_unobserved=n_unobserved, n_bags=n_bags)


def _shape_like_attention(masks, per_bag):
    """Wrap per-bag score vectors as ``[1, N]``, attention's shape.

    Everything downstream slices ``attn[i, :, :length]``, so scoring a reference
    through the identical path rather than a parallel one is what stops the two
    from quietly diverging. ``nulls.template_scores`` does the same.
    """
    return [np.asarray(v, dtype=np.float32)[None, :] for v in per_bag]


def _tile(fit: TemplateFit, masks, kind: str):
    out = []
    for m in masks:
        n = len(m)
        ns = int(np.ceil(n / fit.n_patch))
        bins = depth_bins_for(ns, fit.n_bins)
        if kind == "inplane":
            grid = np.tile(fit.values, (ns, 1))
        elif kind == "axial":
            grid = np.repeat(fit.values[bins][:, None], fit.n_patch, axis=1)
        elif kind == "joint":
            grid = fit.values[bins]
        else:  # pragma: no cover - guarded by the public wrappers
            raise ValueError(kind)
        out.append(grid.ravel()[:n])
    return _shape_like_attention(masks, out)


def inplane_scores(masks, fit: TemplateFit):
    """Tiled in-plane map. **Slice AUC is exactly 0.5 by construction.**"""
    if fit.kind != "inplane":
        raise ValueError(f"expected an inplane fit, got {fit.kind!r}")
    return _tile(fit, masks, "inplane")


def axial_scores(masks, fit: TemplateFit):
    """Depth profile, constant within each slice.

    **Within-slice AUC is exactly 0.5 by construction** -- the mirror of the
    in-plane template's slice AUC, and the reason the pair brackets the claim.
    """
    if fit.kind != "axial":
        raise ValueError(f"expected an axial fit, got {fit.kind!r}")
    return _tile(fit, masks, "axial")


def joint_scores(masks, fit: TemplateFit):
    if fit.kind != "joint":
        raise ValueError(f"expected a joint fit, got {fit.kind!r}")
    return _tile(fit, masks, "joint")


def separable_scores(masks, axial: TemplateFit, inplane: TemplateFit):
    """Outer product of the two one-axis fits: independence, as a reference.

    A product rather than a sum because both factors are mean *attention*, which
    is non-negative and behaves like an unnormalised density; the product is the
    density you get by assuming depth and in-plane position are independent. The
    gap between this and :func:`joint_scores` is therefore an interaction term --
    how much the in-plane prior itself changes with depth -- which is a quantity
    nobody has reported and which the separable/joint pair isolates.
    """
    if axial.kind != "axial" or inplane.kind != "inplane":
        raise ValueError(f"need (axial, inplane), got ({axial.kind!r}, "
                         f"{inplane.kind!r})")
    if axial.n_patch != inplane.n_patch:
        raise ValueError("axial and inplane fits disagree on n_patch")
    out = []
    for m in masks:
        n = len(m)
        ns = int(np.ceil(n / inplane.n_patch))
        bins = depth_bins_for(ns, axial.n_bins)
        grid = axial.values[bins][:, None] * inplane.values[None, :]
        out.append(grid.ravel()[:n])
    return _shape_like_attention(masks, out)


# Not dict[str, TemplateFit]: the returned mapping carries a "source" string alongside the three
# fits, so that a family always travels with the record of what it was fit to. Losing that would
# make an attention-fitted family and a mask-fitted one indistinguishable downstream, and they
# mean opposite things -- one is content-free, the other is an oracle.
def fit_family(masks, attns=None, slot: int | None = None,
               source: str = "attention", n_patch: int = N_PATCH,
               n_bins: int = DEFAULT_DEPTH_BINS) -> dict[str, Any]:
    """Fit the family on one split. **Call this on validation only.**

    ``source="attention"`` reproduces the pre-registered ``fitted_template`` and
    keeps the question arm-relative. ``source="masks"`` fits the oracle instead
    -- see :func:`lesion_rows` for why both are needed and what confusing them
    would cost.
    """
    if source == "attention":
        if attns is None or slot is None:
            raise ValueError("source='attention' needs attns and slot")
        rows = attention_rows(attns, slot)
    elif source == "masks":
        rows = lesion_rows(masks)
    else:
        raise ValueError(f"source must be 'attention' or 'masks', got {source!r}")
    return {
        "source": source,
        "inplane": inplane_template(rows, masks, n_patch=n_patch),
        "axial": axial_template(rows, masks, n_patch=n_patch, n_bins=n_bins),
        "joint": joint_template(rows, masks, n_patch=n_patch, n_bins=n_bins),
    }


def score_family(masks, family: dict[str, TemplateFit]) -> dict[str, list]:
    """Apply a fitted family to held-out bags. Never reads an image."""
    return {
        "inplane": inplane_scores(masks, family["inplane"]),
        "axial": axial_scores(masks, family["axial"]),
        "separable": separable_scores(masks, family["axial"], family["inplane"]),
        "joint": joint_scores(masks, family["joint"]),
    }


assert GRID * GRID == N_PATCH, "nulls.GRID and nulls.N_PATCH disagree"
