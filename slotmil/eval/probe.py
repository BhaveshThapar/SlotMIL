"""The supervised patch probe, as an estimand rather than as a printed number.

H7 is the power gate: the probe's prior-normalised skill must exceed 0.50 while
every content-free baseline's stays below 0.05. If it fails, the constructive
half of the paper is withdrawn -- so every term in it has to mean one thing, and
until the 2026-08-15 amendment two of them did not.

``scripts/probe_ceiling.py`` is not that instrument and was never meant to be. It
fits on 60 scans, tests on 40, subsamples negatives at 20 per positive and
returns a bare patch AUC of 0.9102. That number is not divisible into a skill:
there is no denominator attached to it, and carrying it through the eight stored
``attention:inplane`` denominators in ``runs/nulls/template_family.json`` spans
skill 0.4720 to 0.7224 -- straddling the very threshold H7 leads with. The gate
would have been passed or failed by a choice nobody made.

Two things change here, both pre-registered at
``configs/prereg/isbi2027.yaml:584-643``:

**The probe carries its own denominator.** ``fitted_template`` is fit to the
*scorer's* own validation attention, so every arm already has one; the probe's
per-patch scores take the place of attention and it carries one too. Borrowing
another arm's would make the verdict depend on which arm was borrowed from.

**Subsampling moves to fit time only.** Subsampling at fit time is a fitting
choice and touches no estimand. Subsampling at score time computes the numerator
over a different patch population than the denominator, which is a different
quantity wearing the same name -- so :func:`score_bags` scores every patch of
every bag, and ``tests/test_probe.py`` asserts it.

Scores are emitted in the ``[1, N]`` shape the attention dumps use, so they flow
through :func:`slotmil.eval.axes.per_bag_axes` and
:func:`slotmil.eval.nulls.global_template` unchanged rather than through a
parallel path that might quietly differ -- the same reason ``centre_prior``
returns ``K=1`` attention.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

__all__ = ["collect_fit_set", "fit_probe", "score_bags"]

# Rows read per HDF5 call in score_bags. The cache is lzf-chunked (11, 32, 96),
# so a whole-dataset fancy index decompresses far more than it returns; slicing a
# contiguous run of slices is the same idiom FeatureBagDataset.__getitem__ uses
# and for the same reason.
SLICE_CHUNK = 16


def collect_fit_set(f, uids, neg_per_pos: int, rng, max_scans: int | None = None):
    """Patch features and labels for fitting: every positive, sampled negatives.

    Lifted from ``scripts/probe_ceiling.py`` so the fit is the one that produced
    the published 0.9102 when handed the same arguments. ``max_scans`` is kept
    for exactly that reproduction and defaults to *no cap* -- ``probe_protocol``
    declares ``fit_split: train`` and no scan limit, so the pre-registered fit
    uses the whole split.

    Series with no lesion patch contribute nothing: a bag that is all-negative
    has no positive to sample against, and ``per_bag_axes`` drops it at score
    time too.
    """
    X: list[np.ndarray] = []
    y: list[np.ndarray] = []
    used = 0
    for uid in uids:
        if uid not in f:
            continue
        g = f[uid]
        if "mask" not in g:
            continue
        m = g["mask"][:]
        pos = np.argwhere(m > 0)
        if len(pos) == 0:
            continue
        gh = int(g.attrs["grid_h"])
        flat = m.reshape(m.shape[0], -1)
        neg = np.argwhere(flat == 0)
        pick = rng.choice(len(neg), min(len(pos) * neg_per_pos, len(neg)),
                          replace=False)

        # Gathered in one advanced index, positives first, then the sampled
        # negatives in the order rng.choice returned them -- the order the
        # per-patch loop this replaces produced, so the fit set is unchanged.
        #
        # Advanced indexing because it COPIES. `block[s, p]` on a basic index
        # returns a *view*, which keeps the whole series alive for as long as any
        # one of its patches is held: 608 lesion-bearing training series at up to
        # 275 MB each is ~55 GB, and this OOM-killed a 20 GB login node. The old
        # driver capped at 60 scans and never reached the cliff.
        rows_s = np.concatenate([pos[:, 0], neg[pick, 0]])
        rows_p = np.concatenate([pos[:, 1] * gh + pos[:, 2], neg[pick, 1]])
        block = g["features"][:]
        X.append(np.asarray(block[rows_s, rows_p], dtype=np.float32))
        y.append(np.concatenate([np.ones(len(pos), dtype=np.int64),
                                 np.zeros(len(pick), dtype=np.int64)]))
        del block

        used += 1
        if max_scans is not None and used >= max_scans:
            break
    if not X:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.concatenate(X), np.concatenate(y)


def fit_probe(f, uids, neg_per_pos: int = 20, seed: int = 0,
              max_scans: int | None = None) -> LogisticRegression:
    """Fit the patch probe on one split. ``f`` is an open :class:`h5py.File`.

    ``class_weight="balanced"`` and ``max_iter=2000`` are ``probe_ceiling.py``'s,
    unchanged: this is the same classifier, given a bigger fit set and asked for
    a different output.
    """
    rng = np.random.default_rng(seed)
    X, y = collect_fit_set(f, uids, neg_per_pos, rng, max_scans)
    if X.size == 0:
        raise ValueError("no lesion-bearing series in the fit split")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    return clf


def score_bags(clf: LogisticRegression, f, uids):
    """Score **every** patch of every bag -> ``(attns, masks)``, dump-shaped.

    Returns one ``[1, N]`` float32 array and one ``[N]`` int8 target per bag, in
    the order ``uids`` gives, along with the uids actually scored. That is
    exactly what ``null_battery.load`` hands to ``per_bag_axes``, so the probe is
    scored by the same code as every arm.

    ``decision_function`` rather than ``predict_proba``: AUC is rank-based and
    the logistic link is monotone, so the two give identical numbers, and the
    log-odds avoids saturating to 1.0 across a 148k-instance bag where float32
    ties would otherwise be manufactured out of nothing.

    Read in slice chunks. A 700-slice bag is 179k patches by 768 dimensions --
    550 MB as float32 if materialised whole, and the eval path has no AMP to hide
    behind. Only the ``[N]`` score vector is kept.
    """
    attns, masks, scored = [], [], []
    for uid in uids:
        if uid not in f:
            continue
        g = f[uid]
        if "mask" not in g:
            continue
        feats = g["features"]
        n_slices, n_patch = feats.shape[0], feats.shape[1]
        out = np.empty(n_slices * n_patch, dtype=np.float32)
        for lo in range(0, n_slices, SLICE_CHUNK):
            hi = min(lo + SLICE_CHUNK, n_slices)
            block = np.asarray(feats[lo:hi], dtype=np.float32)
            out[lo * n_patch:hi * n_patch] = clf.decision_function(
                block.reshape(-1, block.shape[-1]))
        m = np.asarray(g["mask"][:]).ravel()
        if m.shape[0] != out.shape[0]:
            raise ValueError(
                f"{uid}: mask has {m.shape[0]} patches, features give "
                f"{out.shape[0]}"
            )
        attns.append(out[None, :])
        masks.append((m > 0).astype(np.int8))
        scored.append(uid)
    return attns, masks, scored
