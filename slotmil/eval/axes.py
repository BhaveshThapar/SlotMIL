"""Which axis is the "3D" localisation number actually measuring?

A frozen-slot instance AUC of 0.8424 reads as "the model found the nodule in the
volume". Split the same 129 patients by axis and it is not that. The *slice*
axis -- which slice holds the nodule, the only part of the score that is about
depth -- comes out at 0.4822, while the within-slice axis comes out at 0.8411.
Flat and within-slice differ by 0.0013, and that gap is the paper's lead claim:
H1 asserts ``|flat instance AUC - within-slice instance AUC| < 0.02`` for every
arm, i.e. the reported 3D metric is an in-plane metric wearing 3D clothing.

The same three columns carry two more hypotheses, which is why the split belongs
in the library rather than in whichever script needed it first:

* H2 -- the model-free centre prior's slice AUC (0.6026) exceeds every trained
  arm's (0.4822-0.5565). Attention MIL is worse than geometry on the axial axis.
* H6 -- Normal Guidance is supposed to move the slice column without moving
  patient-specific skill.

The split is computed **per bag and returned per bag**, with the originating bag
index attached, rather than pooled into three means. Pooling would be smaller and
wrong: the interval on every one of these numbers is a patient-level cluster
bootstrap (:func:`slotmil.eval.estimands.cluster_bootstrap`), and resampling
patients is impossible once the rows have been averaged away. The index is also
what lets a caller map rows back to uids and thence to patients.

Ported from ``scripts/axis_gate.py``, where it was the canonical copy, and
``scripts/null_decompose.py``, which held a second one that had drifted into
returning pooled means. Verified against both originals on the real 129-bag
discovery dumps in ``runs/nulls``: every point estimate and every bootstrap
bound in ``runs/nulls/axis_gate.json`` reproduces **bit-identically** through
this module, at 2000 reps (the reps the stored file was written with) and at
10000. Bit-identical rather than approximately equal is the bar here on purpose
-- the confirmatory thresholds in ``configs/prereg/isbi2027.yaml`` were set from
those discovery numbers, so a decimal that moved during a refactor would
invalidate them silently.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from slotmil.eval.nulls import N_PATCH

__all__ = ["per_bag_axes"]


def per_bag_axes(attns, masks, slot, n_patch: int = N_PATCH):
    """Per-bag flat / slice / within-slice AUC, plus the bag index each came from.

    Returns one ``(index, flat, slice, within_slice, n_slices, n_lesion_slices)``
    tuple per scored bag. The three AUCs are:

    ``flat``
        Lesion patch vs the rest over the whole bag -- the reported statistic.
    ``slice``
        Mean attention per slice vs "this slice holds a nodule". Pure axial
        localisation, and the only column that is about depth.
    ``within_slice``
        Lesion patch vs the rest, restricted to lesion-bearing slices. Depth is
        held fixed by construction, so this is in-plane localisation alone.

    Degeneracy is handled two different ways, and the difference matters. A bag
    that is degenerate *as a bag* -- no lesion patch at all, or nothing but
    lesion patches, or a length that is not a whole number of ``n_patch``
    slices -- is **dropped**, so it contributes no row and cannot be resampled.
    A bag that is degenerate only *on one axis* -- every slice holds a nodule, so
    the slice axis has no contrast -- keeps its row and gets ``nan`` in that one
    column. Dropping such a bag outright would silently shrink the flat and
    within-slice samples too, and the three columns would then be bootstrapped
    over different sets of patients while being reported side by side;
    ``cluster_bootstrap`` drops the non-finite entries per column instead.

    ``slot`` indexes the first axis of each ``[K, N]`` attention array. A
    content-free reference with no slots (the centre prior) is passed as ``K=1``
    with ``slot=0`` so it goes through this identical path rather than a parallel
    one that might quietly differ.
    """
    rows = []
    for i, (a, m) in enumerate(zip(attns, masks)):
        t = (m > 0).astype(np.int8)
        s = int(t.sum())
        if s == 0 or s == len(t):
            continue
        n = len(m)
        ns = n // n_patch
        if ns * n_patch != n:
            continue

        flat = roc_auc_score(t, a[slot])

        sc = a[slot][: ns * n_patch].reshape(ns, n_patch)
        st = t[: ns * n_patch].reshape(ns, n_patch)
        has = (st.sum(axis=1) > 0).astype(np.int8)

        sl = np.nan
        if 0 < has.sum() < ns:
            sl = roc_auc_score(has, sc.mean(axis=1))

        wi = np.nan
        sel = has.astype(bool)
        sub_t, sub_s = st[sel].ravel(), sc[sel].ravel()
        if 0 < sub_t.sum() < len(sub_t):
            wi = roc_auc_score(sub_t, sub_s)

        rows.append((i, flat, sl, wi, ns, int(has.sum())))
    return rows
