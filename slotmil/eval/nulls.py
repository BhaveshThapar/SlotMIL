"""Reference baselines: what a localisation score looks like with no information.

Every number in this module answers the same question in a different way -- *how
much of that AUC would you get for free?* They were written as one-off scripts
under `scripts/null_*.py` while the answer was still in doubt; they live here now
because the answer turned out to be "most of it", which makes them the paper's
instrument rather than its scratch work.

The baselines form a ladder, and the rungs matter:

``chance``
    0.5. Never observed, and asserting it without measuring is the mistake this
    module exists to prevent.
``roll_masks``
    Circular-shift each bag's target. Preserves per-bag prevalence *exactly*, so
    it isolates the metric's own floor from any prevalence artefact.
``random_attn`` (entropy-matched)
    Attention with the real thing's entropy but no content.
``centre_prior``
    A fixed geometric rule. No model, no training, no fitting.
``global_template``
    A 256-number in-plane map fit on validation that never reads a test image --
    the strongest content-free reference available, and therefore the right
    denominator for prior-normalised skill.
``shuffle_masks_across_bags``
    Score against a *different patient's* targets. Whatever survives was never
    about this patient.

Ported from ``scripts/null_battery.py`` and ``scripts/null_static_template.py``.
Verified against the originals on the real 148-bag test dump: ``roll_masks`` and
``shuffle_masks_across_bags`` are bit-identical given the same RNG stream, and
the two content-free references agree to within a float32 cast -- centre-prior
AUC 0.775160 vs 0.775164, template max |diff| 2.8e-8. Scores are returned as
float32 because a bag runs to 148k instances; the resulting AUC shift is below
1e-5, which is four orders of magnitude under anything the paper argues about.
"""

from __future__ import annotations

import numpy as np

N_PATCH = 256   # 16x16 DINOv2 ViT-B/14 patch grid per slice
GRID = 16

__all__ = [
    "binary_findings",
    "centre_prior_scores",
    "global_template",
    "template_scores",
    "roll_masks",
    "shuffle_masks_across_bags",
    "slot_entropy",
    "match_scale",
    "random_attn",
]


def binary_findings(masks):
    """``[N]`` lesion mask -> ``[2, N]`` complementary (lesion, background) channels.

    Two *complementary* columns is what makes the Hungarian assignment in
    ``alignment.fit_slot_assignment`` algebraically equal to argmax of the lesion
    column. That is worth knowing when writing about the "naming" step: with F=2
    there is no combinatorial matching happening, and claiming otherwise
    overstates the protocol.
    """
    return [np.stack([(m > 0).astype(np.float32), 1.0 - (m > 0).astype(np.float32)])
            for m in masks]


def centre_prior_scores(masks, n_patch: int = N_PATCH, grid: int = GRID):
    """Negative distance from the volume centre, shaped like ``[1, N]`` attention.

    No model, no training, no fitting -- a fixed rule that scores well purely
    because nodules cluster centrally. Returned in attention's shape so it can go
    through the identical scoring path rather than a parallel one that might
    quietly differ.
    """
    yy, xx = np.meshgrid(np.linspace(-1, 1, grid), np.linspace(-1, 1, grid),
                         indexing="ij")
    in_plane = np.sqrt(yy.ravel() ** 2 + xx.ravel() ** 2)
    out = []
    for m in masks:
        n = len(m)
        n_slices = int(np.ceil(n / n_patch))
        z = np.repeat(np.linspace(-1, 1, n_slices), n_patch)[:n]
        d = np.sqrt(z ** 2 + np.tile(in_plane, n_slices)[:n] ** 2)
        out.append((-d)[None, :].astype(np.float32))
    return out


def global_template(attns, masks, slot: int, n_patch: int = N_PATCH):
    """One mean in-plane map over every slice of every bag: 256 numbers, no more.

    Fit on validation and then frozen. At test time it is applied identically to
    every bag, so it *cannot* read the image -- which is exactly why it is the
    honest denominator. Arun et al. (Radiology: AI 2021) used the unfit mean of
    training masks; fitting to the attention instead gives a strictly stronger
    reference, and a stronger reference lowers everyone's reported skill.
    """
    total = np.zeros(n_patch, dtype=np.float64)
    count = 0
    for a, m in zip(attns, masks):
        # Truncate to whole slices rather than skipping the bag, matching the
        # original: a bag with a trailing partial slice still contributes its
        # complete ones. Real bags are exact multiples, so this only matters for
        # synthetic inputs -- but a silent divergence here would make numbers
        # computed before and after this module was extracted incomparable.
        n_slices = len(m) // n_patch
        total += a[slot][: n_slices * n_patch].reshape(n_slices, n_patch).sum(axis=0)
        count += n_slices
    if count == 0:
        raise ValueError("no bag had a whole number of slices; cannot fit a template")
    return (total / count).astype(np.float32)


def template_scores(masks, template, n_patch: int = N_PATCH):
    """Tile a fitted template across each bag's slices, in attention's shape."""
    if template.shape != (n_patch,):
        raise ValueError(f"template must be [{n_patch}], got {template.shape}")
    out = []
    for m in masks:
        n = len(m)
        n_slices = int(np.ceil(n / n_patch))
        out.append(np.tile(template, n_slices)[:n][None, :].astype(np.float32))
    return out


def roll_masks(masks, rng):
    """Circular-shift each bag's target by a random patch offset.

    The strictest null available. Per-bag prevalence is preserved *exactly*, so
    the score cannot be contaminated by a prevalence artefact, and the offset is
    in patches rather than whole slices **on purpose**: a whole-slice shift would
    leave every lesion's in-plane position untouched, so an in-plane positional
    prior would still score well against the rolled target and the "floor" would
    come out at the prior's value instead of chance. Rolling by patches destroys
    axial and in-plane registration together, which is what makes the resulting
    0.5004 a measurement of the metric itself.
    """
    return [np.roll(m, int(rng.integers(1, len(m)))) for m in masks]


def shuffle_masks_across_bags(masks, rng):
    """Give each bag another bag's target, tiled or cropped to length.

    A derangement, not a shuffle: a plain permutation leaves some bags matched
    with themselves, which pulls the cross-patient null toward the real score and
    understates how much of the signal is population anatomy. Fixed points are
    repaired by swapping with the next index, which cannot create a new one.

    Bags differ in length -- LIDC series run 58 to 700 slices -- so the donor
    mask is cropped or tiled to the recipient's length. Handing back a
    wrong-length array here would not raise; it would silently mis-score.
    """
    b = len(masks)
    if b < 2:
        return [m.copy() for m in masks]
    perm = rng.permutation(b)
    for i in range(b):
        if perm[i] == i:
            j = (i + 1) % b
            perm[i], perm[j] = perm[j], perm[i]
    out = []
    for i in range(b):
        src, n = masks[perm[i]], len(masks[i])
        out.append(src[:n].copy() if len(src) >= n else np.resize(src, n))
    return out


def slot_entropy(attns) -> float:
    """Mean entropy of the attention distribution *over slots*, per instance.

    Over slots, not over instances: slot attention normalises across the slot
    axis, and measuring the wrong axis is the error that produced this project's
    first reversal. This is the statistic a random null must match to be a fair
    comparison -- otherwise the null differs from the model in peakedness as well
    as in content, and the comparison confounds the two.
    """
    total, count = 0.0, 0
    for a in attns:
        p = np.clip(a, 1e-12, None)
        p = p / p.sum(axis=0, keepdims=True)
        total += float((-(p * np.log(p)).sum(axis=0)).sum())
        count += a.shape[1]
    return total / max(count, 1)


def match_scale(target_entropy: float, k: int, lo: float = 1e-3, hi: float = 30.0,
                iters: int = 40, seed: int = 0) -> float:
    """Logit scale whose random attention has the given slot-entropy.

    Bisection: entropy falls monotonically as the scale grows, from log(k) at
    scale 0 to 0 in the limit. Deterministic given ``seed`` so a null is
    reproducible.
    """
    rng = np.random.default_rng(seed)
    probe = rng.standard_normal((k, 4096))

    def entropy_at(scale):
        p = np.exp(probe * scale - (probe * scale).max(axis=0, keepdims=True))
        p /= p.sum(axis=0, keepdims=True)
        return float((-(p * np.log(np.clip(p, 1e-12, None))).sum(axis=0)).mean())

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if entropy_at(mid) > target_entropy:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def random_attn(lengths, k: int, scale: float, rng, slot_bias=None):
    """Random attention normalised over slots, optionally with a persistent bias.

    ``slot_bias`` gives every bag the same per-slot offset, which mimics a model
    that has learned a global slot preference but nothing image-specific. Without
    it a random null is *too* null: real attention has structure that is not
    lesion knowledge, and the baseline should have it too.
    """
    out = []
    for n in lengths:
        logits = rng.standard_normal((k, n)) * scale
        if slot_bias is not None:
            logits = logits + np.asarray(slot_bias)[:, None]
        logits -= logits.max(axis=0, keepdims=True)
        p = np.exp(logits)
        out.append((p / p.sum(axis=0, keepdims=True)).astype(np.float32))
    return out
