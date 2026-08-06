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
    ):
        super().__init__()
        self.multilabel = multilabel
        self.w_diversity = w_diversity
        self.w_entropy = w_entropy
        self.w_recon = w_recon
        self.label_smoothing = label_smoothing

    def forward(
        self,
        out: dict,
        targets: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        recon: torch.Tensor | None = None,
        recon_target: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        comps = {}
        loss = bag_loss(
            out["logits"],
            targets,
            multilabel=self.multilabel,
            class_weights=class_weights,
            label_smoothing=self.label_smoothing,
        )
        comps["bag"] = loss.detach()

        if self.w_diversity > 0:
            d = slot_diversity_loss(out["tokens"])
            loss = loss + self.w_diversity * d
            comps["diversity"] = d.detach()

        if self.w_entropy > 0:
            e = attention_entropy_loss(out["attn"], pad_mask)
            loss = loss + self.w_entropy * e
            comps["entropy"] = e.detach()

        if self.w_recon > 0:
            if recon is None or recon_target is None:
                raise ValueError("w_recon > 0 but no reconstruction was supplied")
            r = reconstruction_loss(recon, recon_target, pad_mask)
            loss = loss + self.w_recon * r
            comps["recon"] = r.detach()

        comps["total"] = loss.detach()
        return loss, comps
