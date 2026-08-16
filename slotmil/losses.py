"""Losses and regularisers (plan.md line 102).

Bag supervision is the primary signal. Everything else is a regulariser aimed at
a specific documented failure mode, and each is independently switchable so the
ablation table can attribute effects rather than guess at them.

On the reconstruction loss specifically: plan.md line 168 flags it as genuinely
open whether DINOSAUR-style feature reconstruction helps once a supervised bag
label is present. It is wired up but defaults OFF, so the ablation measures a
real hypothesis instead of rationalising a default.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bag_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    multilabel: bool = False,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """CE for single-label (NoduleMNIST3D, COV19D), BCE for multi-label
    (RAD-ChestCT, CT-RATE)."""
    if multilabel:
        return F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=class_weights
        )
    return F.cross_entropy(
        logits, targets.long(), weight=class_weights, label_smoothing=label_smoothing
    )


def slot_diversity_loss(slots: torch.Tensor) -> torch.Tensor:
    """Penalise off-diagonal slot-slot cosine similarity.

    Targets collapse (all slots identical) and duplication (two slots binding the
    same finding) -- plan.md line 104. Uses the mean of squared off-diagonal
    cosines, which stays smooth at zero unlike a mean-absolute penalty.
    """
    b, k, _ = slots.shape
    if k < 2:
        return slots.new_zeros(())
    s = F.normalize(slots, dim=-1)
    sim = torch.einsum("bkd,bjd->bkj", s, s)
    eye = torch.eye(k, dtype=torch.bool, device=slots.device).unsqueeze(0)
    off = sim.masked_select(~eye).view(b, k * (k - 1))
    return (off**2).mean()


def attention_entropy_loss(
    attn: torch.Tensor, pad_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Encourage spatially compact slot masks by minimising the entropy of each
    slot's distribution over instances."""
    if pad_mask is not None:
        attn = attn.masked_fill(~pad_mask.unsqueeze(1), 0.0)
    p = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    ent = -(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum(dim=-1)
    return ent.mean()


NG_VAR_FLOOR_SLICES2 = 1.0   # Harvey's sigma=1, as a floor rather than a support
NG_PATCHES_PER_SLICE = 256   # prereg datasets.*.patches_per_slice


def normal_guidance_loss(
    attn: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
    slice_index: torch.Tensor | None = None,
    patches_per_slice: int = NG_PATCHES_PER_SLICE,
    var_floor: float = NG_VAR_FLOOR_SLICES2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL(attention's slice marginal || moment-matched Normal), per bag, per head.

    Harvey et al., arXiv:2605.27306, "Normal Guidance is what Attention Needs".
    Carries H6.

    The prereg says "KL to a moment-matched Normal over slice index, recomputed
    under stop-gradient each step" and nothing more, so three things are ruled by
    amendment 2026-08-15:

    *Matched to what.* To the marginal's own moments, detached. The alternative
    -- matching a fixed geometric centre -- leaves nothing to recompute, since
    only S varies and S carries no gradient, which would make "recomputed under
    stop-gradient each step" empty prose. Reading it as self-matching also makes
    the term a Gaussian *shape* prior: it forces the axial profile to be unimodal
    and Normal without saying where, so it is structurally incapable of injecting
    information about this patient's lesion. That is what makes H6 falsifiable
    rather than tautological.

    *Direction.* KL(a || q). The reverse diverges wherever a slice is starved and
    would make the loss scale with the model's peakedness rather than with prior
    mismatch. This direction also decomposes as -H(a) + CE(a, q), so it is
    continuous with :func:`attention_entropy_loss` above.

    *Variance floor, and it is not optional.* Unfloored, a single-slice attention
    is a global minimum at exactly 0 -- a Dirac is the sigma->0 Normal -- so the
    term would drive attention onto one slice and report the collapse as a
    localisation triumph. Floored at 1 slice^2 a delta pays 0.9189 nats while a
    true Gaussian pays 0. This is the one place a raw-slice sigma of 1 is both
    meaningful and representable, so Harvey's constant survives here.

    Measured on S=48, for the record and for tests/test_losses.py::
        delta (1 slice)   KL=0.0000 unfloored,  0.9189 floored
        gaussian sigma=1  KL=0.0000             0.0000
        uniform           KL=0.0895             0.0895
        bimodal           KL=1.0624             1.0624

    Bimodal pays most, so NG actively suppresses multi-focal attention. LIDC has
    multi-nodule cases: that is a real cost of the method, not a defect here.

    ``slice_index`` carries true anatomical slice numbers ([B, S], -1 padded) when
    ``--max-slices`` subsampled the bag. Without it the moments and the floor land
    in subsampled units. Absent, it falls back to ``arange(S)``, which is exactly
    what the true index equals at evaluation time.

    Moments are per-bag and per-head, never per-batch: LIDC bags run 58-700 slices
    and ``collate_bags`` pads to max(lengths), so a batch-shared mu in slice units
    would be dominated by the deepest bag. Reduction over the batch is the mean,
    matching :func:`attention_entropy_loss`.
    """
    b, k, n = attn.shape
    p = max(int(patches_per_slice), 1)

    if pad_mask is not None:
        attn = attn.masked_fill(~pad_mask.unsqueeze(1), 0.0)
        lengths = pad_mask.sum(dim=1)
    else:
        lengths = torch.full((b,), n, device=attn.device, dtype=torch.long)

    s_max = (n + p - 1) // p
    idx = torch.div(torch.arange(n, device=attn.device), p, rounding_mode="floor")
    # index_add_ rather than reshape: collate_bags pads to max(lengths), which is
    # not guaranteed to be a multiple of patches_per_slice (and is 15 in tests).
    m = attn.new_zeros(b, k, s_max).index_add_(2, idx, attn)
    m = m / m.sum(dim=-1, keepdim=True).clamp_min(eps)

    n_slices = torch.div(lengths + p - 1, p, rounding_mode="floor")
    valid = (
        torch.arange(s_max, device=attn.device).unsqueeze(0) < n_slices.unsqueeze(1)
    ).unsqueeze(1)  # B,1,S

    if slice_index is None:
        coord = torch.arange(s_max, device=attn.device, dtype=attn.dtype)
        coord = coord.unsqueeze(0).expand(b, s_max)
    else:
        coord = slice_index[:, :s_max].to(attn.dtype)
        if coord.shape[1] < s_max:  # shorter than the padded instance axis
            coord = F.pad(coord, (0, s_max - coord.shape[1]), value=-1.0)
        coord = coord.clamp_min(0.0)  # -1 pads are masked out below anyway
    coord = coord.unsqueeze(1)  # B,1,S

    # The stop-gradient, and its exact placement: on the target's moments only.
    # Gradient must still flow through m in both the outer factor and log m --
    # detaching m inside the KL instead would silently give exactly zero
    # gradient, which tests/test_losses.py pins.
    md = m.detach()
    mu = (md * coord).sum(dim=-1, keepdim=True)
    var = (md * (coord - mu) ** 2).sum(dim=-1, keepdim=True).clamp_min(var_floor)

    logq = -0.5 * (coord - mu) ** 2 / var
    logq = logq.masked_fill(~valid, float("-inf"))
    logq = logq - torch.logsumexp(logq, dim=-1, keepdim=True)
    logq = logq.masked_fill(~valid, 0.0)  # paired with m == 0 on those slices

    kl = (m * (m.clamp_min(eps).log() - logq)).masked_fill(~valid, 0.0).sum(dim=-1)
    return kl.mean()


CLAM_TOPK_B = 8            # prereg arms[clam_sb].topk_b -- CLAM's printed B
CLAM_BAG_WEIGHT = 0.7      # prereg arms[clam_sb].bag_weight
CLAM_INSTANCE_WEIGHT = 1.0 - CLAM_BAG_WEIGHT
DSMIL_BAG_WEIGHT = 0.5     # prereg arms[dsmil].bag_weight
DSMIL_MAX_WEIGHT = 1.0 - DSMIL_BAG_WEIGHT


def clam_instance_loss(
    instance_logits: torch.Tensor,
    attn: torch.Tensor,
    targets: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
    k: int = CLAM_TOPK_B,
) -> torch.Tensor:
    """CLAM's in-the-class instance clustering (Lu et al., Nature BME 2021).

    Per bag, the ``k`` highest-attention instances are pseudo-labelled evidence
    for the bag's class and the ``k`` lowest pseudo-labelled against it, and the
    bag-class instance classifier is trained on those 2k. ``instance_logits`` is
    ``[B, N, C, 2]`` -- one binary classifier per class, of which only the bag's
    own is supervised (CLAM's ``subtyping=False`` default).

    Two rulings, both in AMENDMENTS.md:

    *k stays at CLAM's printed 8*, not a fraction of the bag. LIDC bags run to
    43.8k instances against a WSI's few thousand, so a proportional k would make
    this term's effective weight scale with scan depth -- 12x across the cohort --
    and the branch exists to sharpen attention's extremes, not to label the bag.
    Where a bag is too short to yield 2k disjoint instances, k drops to
    ``n_valid // 2`` for that bag rather than letting the two sets overlap and
    train the classifier on contradictory labels for the same instance.

    *Cross-entropy, not the smooth top-1 SVM* of the reference implementation.
    The SVM surrogate arrives as a third-party package (``topk``) for a margin
    that no pre-registered estimand reads: the branch shapes attention, and
    attention is scored by rank. CE is the standard surrogate for the same
    pseudo-labels and keeps the dependency set auditable.
    """
    if targets.ndim > 1:
        raise ValueError(
            "clam_instance_loss expects single-label bag targets; CLAM's "
            "in-the-class clustering has no defined bag class under multilabel"
        )
    b, n = attn.shape[0], attn.shape[-1]
    a = attn[:, 0]  # B,N -- CLAM-SB has exactly one attention branch
    if pad_mask is not None:
        lengths = pad_mask.sum(dim=1)
        hi = a.masked_fill(~pad_mask, float("-inf"))
        lo = a.masked_fill(~pad_mask, float("inf"))
    else:
        lengths = torch.full((b,), n, device=a.device, dtype=torch.long)
        hi = lo = a

    # Looped over the batch because k is per-bag: batch_size is 4 here, and a
    # vectorised form would have to clamp k to the shortest bag in the batch,
    # making one bag's loss depend on which bags it was batched with.
    terms = []
    for i in range(b):
        kk = min(int(k), int(lengths[i]) // 2)
        if kk < 1:
            continue
        top = torch.topk(hi[i], kk).indices
        bot = torch.topk(-lo[i], kk).indices
        idx = torch.cat([top, bot])
        logits = instance_logits[i, idx, int(targets[i])]  # 2k, 2
        labels = torch.cat([
            torch.ones(kk, dtype=torch.long, device=a.device),
            torch.zeros(kk, dtype=torch.long, device=a.device),
        ])
        terms.append(F.cross_entropy(logits, labels))
    if not terms:
        return instance_logits.new_zeros(())
    return torch.stack(terms).mean()


def dsmil_max_loss(
    instance_logits: torch.Tensor,
    targets: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
    multilabel: bool = False,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """DSMIL's instance stream: bag loss on the max instance score per class.

    ``instance_logits`` is ``[B, N, C]``. This is the stream that picks the
    critical instance the bag stream then attends against, so leaving it
    unsupervised would leave the critical-instance selection driven only by
    whatever gradient reaches it through the attention -- which is what
    distinguishes DSMIL from a single-stream non-local pooling.

    Padded instances are excluded before the max. They are not merely unlikely to
    win it: ``collate_bags`` pads to the deepest bag in the batch, so on LIDC's
    58-700 slice range most of the tensor can be padding.
    """
    s = instance_logits
    if pad_mask is not None:
        s = s.masked_fill(~pad_mask.unsqueeze(-1), float("-inf"))
    return bag_loss(s.max(dim=1).values, targets, multilabel=multilabel,
                    class_weights=class_weights)


class SlotFeatureDecoder(nn.Module):
    """DINOSAUR-style broadcast MLP decoder reconstructing input features.

    Each slot decodes to a full-length feature map plus an alpha channel; the
    alphas softmax across slots and mix the per-slot reconstructions. Seitzer et
    al. (ICLR 2023) recommend the MLP decoder as first choice.
    """

    def __init__(self, dim: int, out_dim: int, hidden: int = 1024, max_instances: int = 4096):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, 1, max_instances, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim + 1),  # features + alpha
        )
        self.out_dim = out_dim
        self.max_instances = max_instances

    def forward(self, slots: torch.Tensor, num_instances: int) -> torch.Tensor:
        if num_instances > self.max_instances:
            raise ValueError(
                f"decoder built for <={self.max_instances} instances, got "
                f"{num_instances}. Raise max_instances or disable the recon loss "
                "for patch-level bags."
            )
        b, k, d = slots.shape
        x = slots.unsqueeze(2).expand(b, k, num_instances, d) + self.pos[:, :, :num_instances]
        out = self.net(x)
        recon, alpha = out[..., : self.out_dim], out[..., self.out_dim :]
        alpha = alpha.softmax(dim=1)
        return (recon * alpha).sum(dim=1)  # B, N, out_dim


def reconstruction_loss(
    recon: torch.Tensor, target: torch.Tensor, pad_mask: torch.Tensor | None = None
) -> torch.Tensor:
    err = (recon - target) ** 2
    if pad_mask is None:
        return err.mean()
    m = pad_mask.unsqueeze(-1).to(err.dtype)
    return (err * m).sum() / m.sum().clamp_min(1.0) / err.shape[-1]


class SlotMILLoss(nn.Module):
    """Weighted sum of the bag loss and whichever regularisers are enabled.

    Returns ``(total, components)`` so every term is logged separately -- when a
    run misbehaves, knowing which term moved is the difference between a fix and
    a guess.
    """

    def __init__(
        self,
        multilabel: bool = False,
        w_diversity: float = 0.0,
        w_entropy: float = 0.0,
        w_recon: float = 0.0,
        label_smoothing: float = 0.0,
        w_kl: float = 0.0,
        kl_patches_per_slice: int = NG_PATCHES_PER_SLICE,
        kl_var_floor: float = NG_VAR_FLOOR_SLICES2,
        w_bag: float = 1.0,
        w_clam_inst: float = 0.0,
        w_dsmil_max: float = 0.0,
        clam_topk_b: int = CLAM_TOPK_B,
    ):
        super().__init__()
        self.multilabel = multilabel
        self.w_diversity = w_diversity
        self.w_entropy = w_entropy
        self.w_recon = w_recon
        self.label_smoothing = label_smoothing
        # w_kl > 0 is what makes an arm Normal Guidance; the module it wraps is
        # plain gated_abmil. scripts/train_cached.py refuses to run the
        # normal_guidance arm with lam <= 0 for exactly that reason.
        self.w_kl = w_kl
        self.kl_patches_per_slice = kl_patches_per_slice
        self.kl_var_floor = kl_var_floor
        # CLAM and DSMIL both publish their objective as a *weighted split*
        # between the bag term and an auxiliary stream (0.7/0.3 and 0.5/0.5), so
        # w_bag exists to run the printed split literally rather than folding the
        # rescaling into the learning rate and calling it equivalent.
        self.w_bag = w_bag
        self.w_clam_inst = w_clam_inst
        self.w_dsmil_max = w_dsmil_max
        self.clam_topk_b = clam_topk_b

    def forward(
        self,
        out: dict,
        targets: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        recon: torch.Tensor | None = None,
        recon_target: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
        slice_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        comps = {}
        bag = bag_loss(
            out["logits"],
            targets,
            multilabel=self.multilabel,
            class_weights=class_weights,
            label_smoothing=self.label_smoothing,
        )
        comps["bag"] = bag.detach()
        loss = self.w_bag * bag

        if self.w_clam_inst > 0:
            c = clam_instance_loss(
                out["instance_logits"], out["attn"], targets, pad_mask,
                k=self.clam_topk_b,
            )
            loss = loss + self.w_clam_inst * c
            comps["clam_inst"] = c.detach()

        if self.w_dsmil_max > 0:
            m = dsmil_max_loss(
                out["instance_logits"], targets, pad_mask,
                multilabel=self.multilabel, class_weights=class_weights,
            )
            loss = loss + self.w_dsmil_max * m
            comps["dsmil_max"] = m.detach()

        if self.w_diversity > 0:
            d = slot_diversity_loss(out["tokens"])
            loss = loss + self.w_diversity * d
            comps["diversity"] = d.detach()

        if self.w_entropy > 0:
            e = attention_entropy_loss(out["attn"], pad_mask)
            loss = loss + self.w_entropy * e
            comps["entropy"] = e.detach()

        if self.w_kl > 0:
            g = normal_guidance_loss(
                out["attn"], pad_mask, slice_index=slice_index,
                patches_per_slice=self.kl_patches_per_slice,
                var_floor=self.kl_var_floor,
            )
            loss = loss + self.w_kl * g
            comps["kl_prior"] = g.detach()

        if self.w_recon > 0:
            if recon is None or recon_target is None:
                raise ValueError("w_recon > 0 but no reconstruction was supplied")
            r = reconstruction_loss(recon, recon_target, pad_mask)
            loss = loss + self.w_recon * r
            comps["recon"] = r.detach()

        comps["total"] = loss.detach()
        return loss, comps
