"""MIL pooling baselines, all sharing one interface with SlotAttention.

Every pooling module here maps ``(feats [B, N, D_in], pad_mask [B, N])`` to
``(tokens [B, K, D], attn [B, K, N])``. Keeping the signature identical means the
same readout head, the same localisation code and the same slot-to-finding
alignment metrics run unchanged across SlotMIL and every baseline -- which is
what makes the comparisons honest rather than approximately-matched.

``MultiHeadABMIL`` is the load-bearing control (plan.md line 54, reviewer
objection #1: "slots are just multi-head attention"). It gets K heads and a
parameter count matched to SlotMIL, but no iterative refinement, no GRU, and
crucially no softmax-over-slots -- each head attends independently, so there is
no competition to force specialisation. If SlotMIL's slots specialise and these
heads come out redundant, the competitive mechanism is doing the work rather
than the extra capacity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _masked_softmax(logits: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
    """Softmax over the instance axis (dim=-1), respecting padding.

    Unlike slot attention, these baselines normalise *over instances*, so here
    masking the logits with -inf is the correct move (each head always has at
    least one real instance to attend to).
    """
    if pad_mask is not None:
        logits = logits.masked_fill(~pad_mask.unsqueeze(1), float("-inf"))
    return logits.softmax(dim=-1)


class MeanPool(nn.Module):
    def __init__(self, input_dim: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, dim)

    def forward(self, feats, pad_mask=None):
        x = self.proj(feats)
        if pad_mask is None:
            pooled = x.mean(dim=1, keepdim=True)
            n = x.shape[1]
            attn = torch.full((x.shape[0], 1, n), 1.0 / n, device=x.device)
        else:
            m = pad_mask.unsqueeze(-1).to(x.dtype)
            counts = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (x * m).sum(dim=1, keepdim=True) / counts
            attn = (m.squeeze(-1) / counts.squeeze(-1)).unsqueeze(1)
        return pooled, attn


class MaxPool(nn.Module):
    def __init__(self, input_dim: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, dim)

    def forward(self, feats, pad_mask=None):
        x = self.proj(feats)
        if pad_mask is not None:
            x = x.masked_fill(~pad_mask.unsqueeze(-1), float("-inf"))
        pooled, idx = x.max(dim=1, keepdim=True)
        attn = torch.zeros(x.shape[0], 1, x.shape[1], device=x.device)
        # Attribute to the instance that won the most feature dimensions.
        winner = idx.squeeze(1).mode(dim=-1).values
        attn.scatter_(2, winner.view(-1, 1, 1), 1.0)
        return pooled, attn


class ABMIL(nn.Module):
    """Ilse et al., ICML 2018: a = softmax(w^T tanh(V h))."""

    def __init__(self, input_dim: int, dim: int, hidden: int = 128, gated: bool = False):
        super().__init__()
        self.proj = nn.Linear(input_dim, dim)
        self.V = nn.Linear(dim, hidden)
        self.gated = gated
        if gated:
            self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, feats, pad_mask=None):
        x = self.proj(feats)
        a = torch.tanh(self.V(x))
        if self.gated:
            a = a * torch.sigmoid(self.U(x))
        logits = self.w(a).transpose(1, 2)  # B, 1, N
        attn = _masked_softmax(logits, pad_mask)
        pooled = torch.bmm(attn, x)  # B, 1, D
        return pooled, attn


class GatedABMIL(ABMIL):
    """The team's existing model family -- tanh(Vh) * sigmoid(Uh)."""

    def __init__(self, input_dim: int, dim: int, hidden: int = 128):
        super().__init__(input_dim, dim, hidden=hidden, gated=True)


class MultiHeadABMIL(nn.Module):
    """K independent gated-attention heads. The matched-capacity control.

    Deliberately *not* competitive: each head runs its own softmax over
    instances, so nothing stops two heads converging on the same region. That
    redundancy (or its absence) is the measurement.
    """

    def __init__(self, input_dim: int, dim: int, num_heads: int, hidden: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.proj = nn.Linear(input_dim, dim)
        self.V = nn.Linear(dim, hidden * num_heads)
        self.U = nn.Linear(dim, hidden * num_heads)
        self.w = nn.Parameter(torch.zeros(num_heads, hidden))
        nn.init.xavier_uniform_(self.w)
        self.hidden = hidden

    def forward(self, feats, pad_mask=None):
        b, n, _ = feats.shape
        x = self.proj(feats)
        a = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        a = a.view(b, n, self.num_heads, self.hidden)
        logits = torch.einsum("bnkh,kh->bkn", a, self.w)
        attn = _masked_softmax(logits, pad_mask)  # normalised over N, per head
        pooled = torch.einsum("bkn,bnd->bkd", attn, x)
        return pooled, attn


DEFAULT_PATCHES_PER_SLICE = 256  # prereg datasets.*.patches_per_slice
CENTRE_GAUSSIAN_SIGMA_Z = 0.25   # prereg arms[centre_gaussian].sigma_z


class CentreGaussianPool(nn.Module):
    """Harvey et al. (arXiv:2605.27306) content-free axial reference.

    ``a_ij`` proportional to ``NormPDF(z_j | 0, sigma_z)`` over the *normalised*
    slice index, where ``z = linspace(-1, 1, S_i)`` and instance ``j`` sits on
    slice ``j // patches_per_slice``. The attention never reads the image -- that
    is the entire point of the arm, and ``tests/test_position_arms.py`` pins it by
    feeding two different feature tensors and requiring bit-identical attention.
    ``proj`` exists only so the pooled token is D-dimensional and the readout can
    classify; nothing in the attention path has parameters.

    On sigma, and why this departs from the printed formula. Harvey write
    ``NormPDF(j | S_i/2, 1)`` over the *raw* slice index. That is not computable
    on volumetric CT: at S=171 the tail logit is -3612 nats and at S=700 it is
    -61075, against -103.3 for float32's smallest subnormal and -744.4 for
    float64's. The logits themselves survive -- fp32 keeps all 86 distinct values
    -- but this contract returns *normalised* attention, and the softmax collapses
    142 of 171 slices to exactly 0.0 (94 of 171 even in float64). That is a tie
    block over 83% of the bag, and roc_auc_score credits ties at 0.5, so a literal
    implementation would report a floating-point artefact as Harvey's baseline.

    Normalising the support fixes it at no cost to fidelity, because
    ``NormPDF(.|mu,sigma)`` is strictly monotone in ``-|j-mu|`` for every
    ``sigma > 0`` and every pre-registered estimand is rank-based -- measured,
    the slice AUC is identical to 10 decimal places across sigma_z in
    [0.1, 1.0]. sigma_z is therefore free, and is fixed at 0.25: the midpoint of
    the band [0.227, 0.3] where fp16 attention still resolves all 86
    mathematically-distinct values, i.e. the peakiest (most Harvey-like) sigma
    that survives AMP intact. See AMENDMENTS.md 2026-08-15.

    mu is ``(S-1)/2``, not Harvey's ``S/2``. A half-slice offset has no
    anatomical basis and, for even S, breaks what should be a symmetric tie
    between slices ``S/2 - 1`` and ``S/2``. ``(S-1)/2`` is also what makes this
    arm exactly rank-equal to the axial component of ``nulls.centre_prior_scores``,
    which is a testable claim rather than a resemblance.
    """

    def __init__(
        self,
        input_dim: int,
        dim: int,
        sigma: float = CENTRE_GAUSSIAN_SIGMA_Z,
        patches_per_slice: int = DEFAULT_PATCHES_PER_SLICE,
    ):
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if patches_per_slice < 1:
            raise ValueError(f"patches_per_slice must be >= 1, got {patches_per_slice}")
        self.proj = nn.Linear(input_dim, dim)
        self.sigma = float(sigma)
        self.patches_per_slice = int(patches_per_slice)

    def forward(self, feats, pad_mask=None):
        b, n, _ = feats.shape
        p = self.patches_per_slice

        # float32 throughout: protocol.dtype is pre-registered float32, and the
        # exponential must not run under autocast's fp16.
        with torch.autocast(feats.device.type, enabled=False):
            slice_id = torch.div(
                torch.arange(n, device=feats.device), p, rounding_mode="floor"
            ).float()
            if pad_mask is None:
                lengths = torch.full((b, 1), n, device=feats.device, dtype=torch.long)
            else:
                lengths = pad_mask.sum(dim=1, keepdim=True)
            # S_i = ceil(length_i / p), per bag: bags of different depth in one
            # batch each get their own centre, without a loop.
            s = torch.div(lengths + p - 1, p, rounding_mode="floor").float()  # B,1

            # linspace(-1, 1, S) exactly, including numpy's S == 1 -> [-1.0]
            # convention. S == 1 is not a corner case here: the conformance tests
            # use a 15-instance bag, so ceil(15/256) == 1 and a bare (S-1) divisor
            # would be a division by zero on the very first test.
            denom = (s - 1.0).clamp_min(1.0)
            z = torch.where(
                s > 1.0,
                2.0 * slice_id.unsqueeze(0) / denom - 1.0,
                torch.full((b, n), -1.0, device=feats.device),
            )  # B,N

            logits = (-0.5 * (z / self.sigma) ** 2).unsqueeze(1)  # B,1,N
            # _masked_softmax fills pads with -inf, so padded columns come out
            # exactly 0.0 -- one masking convention across every pooling here.
            attn = _masked_softmax(logits, pad_mask)

        x = self.proj(feats)
        attn = attn.to(x.dtype)
        return torch.bmm(attn, x), attn  # B,1,D  B,1,N


class InstanceScoringPool:
    """Marker for poolings that also score individual instances.

    CLAM's clustering branch and DSMIL's max stream are both supervised on
    per-instance logits, which the ``(feats, pad_mask) -> (tokens, attn)``
    contract has no room for. Rather than widen the contract for two arms,
    :class:`~slotmil.models.mil.MILModel` surfaces them as
    ``out["instance_logits"]`` for any pooling marked here -- the same route
    ``out["health"]`` already takes for slot attention.

    The cost is one extra ``proj`` pass per forward, because the hook recomputes
    from ``feats`` rather than caching state on the module. Caching would make
    the value depend on whether ``forward`` had been called first, which is the
    kind of ordering dependency that produces a wrong number rather than a crash.
    """

    def instance_logits(self, feats: torch.Tensor,
                        pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class CLAMSB(GatedABMIL, InstanceScoringPool):
    """CLAM-SB (Lu et al., Nature BME 2021): gated attention + instance clustering.

    Subclassed from :class:`GatedABMIL` for the same reason ``normal_guidance``
    is an alias of it -- the attention path is bit-identical, so the whole
    difference between the two arms is the clustering objective, and any gap
    between them is attributable to that and nothing else. ``hidden`` therefore
    stays at the project default rather than CLAM's 256.

    Single-branch means one attention distribution, so ``K = 1``; the branch that
    varies with class in CLAM-MB is the part this arm does not have.

    The instance classifiers are per class (``num_classes`` of them, packed into
    one ``Linear`` and viewed apart), which is what makes "in-the-class"
    clustering well defined: for a bag of class *c*, classifier *c* is trained
    to call the top-attention instances *c*-evidence and the bottom-attention
    ones not. A single shared classifier would invert that meaning on negative
    bags. ``subtyping`` is off, as in CLAM's default, so out-of-class
    classifiers are left unsupervised.
    """

    def __init__(self, input_dim: int, dim: int, hidden: int = 128,
                 num_classes: int = 2):
        super().__init__(input_dim, dim, hidden=hidden)
        self.num_classes = int(num_classes)
        self.inst = nn.Linear(dim, self.num_classes * 2)

    def instance_logits(self, feats, pad_mask=None):
        x = self.proj(feats)
        return self.inst(x).view(*x.shape[:2], self.num_classes, 2)  # B,N,C,2


DSMIL_Q_HIDDEN = 128     # prereg arms[dsmil].q_hidden -- DSMIL's printed width
DSMIL_DROPOUT_V = 0.0    # prereg arms[dsmil].dropout_v


class DSMIL(nn.Module, InstanceScoringPool):
    """DSMIL (Li, Li & Eliceiri, CVPR 2021): dual-stream, critical-instance MIL.

    Stream one scores every instance and takes the argmax as the bag's critical
    instance. Stream two attends every instance against that critical one --
    a non-local similarity, not a learned scalar per instance -- and pools.

    ``K = num_classes``, not 1. DSMIL selects a critical instance *per class* and
    builds one bag embedding per class, so collapsing to a single token would be
    a different method. The frozen-slot protocol already handles K > 1 arms
    (Hungarian on validation, argmax of the lesion column for F=2), so keeping
    the published shape costs nothing here.

    Attention normalises over the **instance** axis, like every other arm in this
    file and unlike slot attention -- so ``localization.instance_auc(slot=None)``
    is valid for it.

    Two declared departures, both in ``AMENDMENTS.md``: the bag score comes from
    the project's shared readout rather than DSMIL's per-class ``Conv1d``, and
    the instance stream enters through the loss (``dsmil_max_loss``, weight 0.5)
    rather than being averaged into the reported logits. Both exist so the
    classification head is identical across all arms and a difference is
    attributable to pooling; the fidelity cost is stated rather than hidden.
    """

    def __init__(self, input_dim: int, dim: int, num_classes: int = 2,
                 q_hidden: int = DSMIL_Q_HIDDEN, dropout_v: float = DSMIL_DROPOUT_V):
        super().__init__()
        self.proj = nn.Linear(input_dim, dim)
        self.inst = nn.Linear(dim, num_classes)
        self.q = nn.Sequential(nn.Linear(dim, q_hidden), nn.Tanh())
        self.v = nn.Sequential(nn.Dropout(dropout_v), nn.Linear(dim, dim))
        self.num_classes = int(num_classes)
        self.q_hidden = int(q_hidden)

    def instance_logits(self, feats, pad_mask=None):
        return self.inst(self.proj(feats))  # B,N,C

    def forward(self, feats, pad_mask=None):
        b, n, _ = feats.shape
        x = self.proj(feats)
        s = self.inst(x)  # B,N,C

        # Critical instance per class. Padded instances must never win the
        # argmax: a pad's score is whatever proj(0-ish features) happens to give,
        # and on a batch of mixed-depth bags that is most of the tensor.
        if pad_mask is not None:
            s = s.masked_fill(~pad_mask.unsqueeze(-1), float("-inf"))
        crit = s.argmax(dim=1)  # B,C

        qq = self.q(x)  # B,N,H
        q_m = torch.gather(
            qq, 1, crit.unsqueeze(-1).expand(b, self.num_classes, self.q_hidden)
        )  # B,C,H
        logits = torch.einsum("bnh,bch->bcn", qq, q_m) / (self.q_hidden ** 0.5)
        attn = _masked_softmax(logits, pad_mask)  # B,C,N over instances
        return torch.bmm(attn, self.v(x)), attn  # B,C,D  B,C,N


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def matched_multihead_abmil(
    input_dim: int,
    dim: int,
    num_heads: int,
    target_params: int,
    tol: float = 0.02,
) -> MultiHeadABMIL:
    """Build a MultiHeadABMIL whose parameter count matches ``target_params``.

    Solves for the per-head hidden width by bisection rather than algebra, so it
    stays correct if the module's internals change. Raises if it cannot land
    within ``tol``, because a silently-mismatched control would undermine the one
    experiment it exists to support.
    """
    lo, hi = 1, 8192
    best, best_err = None, float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        model = MultiHeadABMIL(input_dim, dim, num_heads, hidden=mid)
        p = count_params(model)
        err = abs(p - target_params) / max(target_params, 1)
        if err < best_err:
            best, best_err = model, err
        if p < target_params:
            lo = mid + 1
        elif p > target_params:
            hi = mid - 1
        else:
            return model
    if best_err > tol:
        raise RuntimeError(
            f"could not parameter-match MultiHeadABMIL to {target_params} "
            f"(best {count_params(best)}, {best_err:.1%} off, tol {tol:.1%}). "
            "Adjust dim or num_heads rather than shipping a mismatched control."
        )
    return best


def build_pooling(name: str, input_dim: int, dim: int, **kw) -> nn.Module:
    if name == "mean":
        return MeanPool(input_dim, dim)
    if name == "max":
        return MaxPool(input_dim, dim)
    if name == "abmil":
        return ABMIL(input_dim, dim, hidden=kw.get("hidden", 128), gated=False)
    if name == "gated_abmil":
        return GatedABMIL(input_dim, dim, hidden=kw.get("hidden", 128))
    if name == "mh_abmil":
        return MultiHeadABMIL(
            input_dim, dim, num_heads=kw["num_heads"], hidden=kw.get("hidden", 128)
        )
    if name == "centre_gaussian":
        return CentreGaussianPool(
            input_dim,
            dim,
            sigma=kw.get("sigma", CENTRE_GAUSSIAN_SIGMA_Z),
            patches_per_slice=kw.get("patches_per_slice", DEFAULT_PATCHES_PER_SLICE),
        )
    if name == "normal_guidance":
        # Deliberately an alias, not a subclass. Normal Guidance differs from its
        # base arm ONLY in the loss (losses.normal_guidance_loss, weighted by
        # w_kl), so sharing the class is what makes the H6 comparison exactly
        # matched: same parameters, same init, same seed. At lam=0 the two are
        # bit-identical, which tests/test_losses.py asserts.
        #
        # The prereg says NG is compared to "its base arm" without naming it.
        # gated_abmil, ruled by amendment 2026-08-15: it has exactly one
        # attention distribution over instances (so the slice marginal needs no
        # extra convention), carries no other regulariser (so H6's delta is
        # attributable to the KL alone), and is the study's strongest classifier
        # (so a localisation gain cannot be dismissed as rescuing a weak arm).
        return GatedABMIL(input_dim, dim, hidden=kw.get("hidden", 128))
    if name == "clam_sb":
        return CLAMSB(
            input_dim,
            dim,
            hidden=kw.get("hidden", 128),
            num_classes=kw.get("num_classes", 2),
        )
    if name == "dsmil":
        return DSMIL(
            input_dim,
            dim,
            num_classes=kw.get("num_classes", 2),
            q_hidden=kw.get("q_hidden", DSMIL_Q_HIDDEN),
            dropout_v=kw.get("dropout_v", DSMIL_DROPOUT_V),
        )
    raise ValueError(f"unknown pooling {name!r}")
