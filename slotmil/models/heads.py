"""Readout heads mapping K slots to a bag prediction (plan.md line 100).

Every head here exposes a per-slot attribution so the question "which slot drove
the prediction?" has an answer. That is the whole point of the bottleneck -- a
head that mixes slots opaquely would throw away the interpretability the method
is being sold on. ``TransformerSlotReadout`` is the deliberate exception, included
to measure the interpretability/accuracy trade-off rather than to be used.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedSlotReadout(nn.Module):
    """Default. Per-slot logits combined by a learned softmax gate.

        bag_logit = sum_k g_k * classifier(slot_k)

    ``g_k`` *is* the attribution: it reports directly how much slot k drove the
    bag-level call, with no post-hoc attribution method in between.
    """

    def __init__(self, dim: int, num_classes: int, gate_hidden: int = 128):
        super().__init__()
        self.classifier = nn.Linear(dim, num_classes)
        self.gate = nn.Sequential(
            nn.Linear(dim, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, 1),
        )

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slot_logits = self.classifier(slots)  # B, K, C
        gate = self.gate(slots).softmax(dim=1)  # B, K, 1
        bag_logits = (gate * slot_logits).sum(dim=1)  # B, C
        return bag_logits, gate.squeeze(-1)


class MaxSlotReadout(nn.Module):
    """Max over per-slot logits -- the classic MIL assumption ("at least one
    slot is positive"). Most interpretable: the argmax slot is the explanation.
    """

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slot_logits = self.classifier(slots)  # B, K, C
        bag_logits, idx = slot_logits.max(dim=1)
        attribution = torch.zeros_like(slot_logits[..., 0])
        attribution.scatter_(1, idx[:, :1], 1.0)
        return bag_logits, attribution


class TransformerSlotReadout(nn.Module):
    """Self-attention over slots + CLS token. Most expressive, least
    interpretable -- present to quantify what the interpretable heads give up.
    """

    def __init__(self, dim: int, num_classes: int, nhead: int = 4, layers: int = 1):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=4 * dim,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = slots.shape[0]
        x = torch.cat([self.cls.expand(b, -1, -1), slots], dim=1)
        x = self.encoder(x)
        bag_logits = self.classifier(x[:, 0])
        # Uniform "attribution": this head genuinely does not expose one, and
        # faking a per-slot number here would misreport the trade-off.
        attribution = torch.full(
            slots.shape[:2], 1.0 / slots.shape[1], device=slots.device
        )
        return bag_logits, attribution


class ConcatMLPReadout(nn.Module):
    """Flatten K x D and MLP. Baseline; slot identity is not preserved (the head
    is not permutation-invariant over slots), which is exactly the property the
    other heads keep.
    """

    def __init__(self, dim: int, num_classes: int, num_slots: int, hidden: int = 256):
        super().__init__()
        self.num_slots = num_slots
        self.net = nn.Sequential(
            nn.Linear(dim * num_slots, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, k, _ = slots.shape
        if k != self.num_slots:
            raise ValueError(
                f"ConcatMLPReadout was built for K={self.num_slots}, got K={k}. "
                "This head cannot vary K at inference; use a gated/max head for "
                "the K-sensitivity sweep."
            )
        bag_logits = self.net(slots.reshape(b, -1))
        attribution = torch.full((b, k), 1.0 / k, device=slots.device)
        return bag_logits, attribution


def build_readout(name: str, dim: int, num_classes: int, num_slots: int) -> nn.Module:
    if name == "gated":
        return GatedSlotReadout(dim, num_classes)
    if name == "max":
        return MaxSlotReadout(dim, num_classes)
    if name == "transformer":
        return TransformerSlotReadout(dim, num_classes)
    if name == "concat_mlp":
        return ConcatMLPReadout(dim, num_classes, num_slots)
    raise ValueError(
        f"unknown readout {name!r}; expected one of gated/max/transformer/concat_mlp"
    )
