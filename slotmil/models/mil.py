"""The MIL wrapper: instance features -> pooling -> readout -> bag prediction.

One class covers SlotMIL and every baseline, because all poolings share the
``(feats, pad_mask) -> (tokens, attn)`` contract. That is deliberate: the
parameter-matched multi-head-ABMIL control only means something if it differs
from SlotMIL in the pooling mechanism and *nothing else*.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .baselines import (
    DEFAULT_PATCHES_PER_SLICE,
    InstanceScoringPool,
    build_pooling,
    count_params,
    matched_multihead_abmil,
)
from .heads import build_readout
from .slot_attention import SlotAttention, slot_health


class MILModel(nn.Module):
    def __init__(
        self,
        pooling: nn.Module,
        readout: nn.Module,
        pooling_name: str,
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.pooling = pooling
        self.readout = readout
        self.pooling_name = pooling_name
        self.encoder = encoder

    @property
    def is_slot(self) -> bool:
        return isinstance(self.pooling, SlotAttention)

    def forward(
        self,
        feats: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        num_slots: int | None = None,
    ) -> dict:
        """``feats``: (B, N, D_in) instance features, or raw instances if an
        ``encoder`` is attached (the end-to-end W1 fast path)."""
        if self.encoder is not None:
            feats = self.encoder(feats, pad_mask)

        if self.is_slot:
            tokens, attn = self.pooling(feats, pad_mask, num_slots=num_slots)
        else:
            tokens, attn = self.pooling(feats, pad_mask)

        logits, attribution = self.readout(tokens)

        out = {
            "logits": logits,
            "attn": attn,  # B, K, N -- reshaped to the patch grid for localisation
            "tokens": tokens,  # B, K, D
            "attribution": attribution,  # B, K -- which slot/head drove the call
        }
        if self.is_slot:
            out["health"] = slot_health(tokens, attn)
        if isinstance(self.pooling, InstanceScoringPool):
            # CLAM's clustering branch and DSMIL's max stream are supervised on
            # per-instance logits, which the pooling contract does not carry.
            # Surfaced here rather than widened into the contract, and recomputed
            # rather than cached -- see InstanceScoringPool.
            out["instance_logits"] = self.pooling.instance_logits(feats, pad_mask)
        return out


def build_model(
    pooling: str,
    input_dim: int,
    dim: int,
    num_classes: int,
    num_slots: int = 8,
    readout: str = "gated",
    iters: int = 3,
    implicit: bool = True,
    init: str = "learnable",
    bo_qsa_straight_through: bool = False,
    hidden: int = 128,
    encoder: nn.Module | None = None,
    match_params_to: int | None = None,
    patches_per_slice: int = DEFAULT_PATCHES_PER_SLICE,
) -> MILModel:
    """Factory.

    ``match_params_to``: for ``pooling="mh_abmil"``, the SlotMIL pooling parameter
    count to match. Pass the number from a SlotMIL built with the same dim/K --
    :func:`slot_pooling_param_count` computes it without instantiating the whole
    model.

    ``patches_per_slice``: only ``centre_gaussian`` reads it, to map instance ->
    slice. Must be 1 when ``instance_level="slice"``, or the arm computes its
    axial geometry against a grid that is not there.
    """
    if pooling == "slot":
        pool = SlotAttention(
            num_slots=num_slots,
            dim=dim,
            input_dim=input_dim,
            iters=iters,
            implicit=implicit,
            init=init,
            bo_qsa_straight_through=bo_qsa_straight_through,
        )
        k_eff = num_slots
    elif pooling == "mh_abmil":
        if match_params_to is None:
            pool = build_pooling(
                "mh_abmil", input_dim, dim, num_heads=num_slots, hidden=hidden
            )
        else:
            pool = matched_multihead_abmil(
                input_dim, dim, num_slots, target_params=match_params_to
            )
        k_eff = num_slots
    else:
        pool = build_pooling(
            pooling,
            input_dim,
            dim,
            hidden=hidden,
            num_heads=num_slots,
            patches_per_slice=patches_per_slice,
            num_classes=num_classes,
        )
        # DSMIL is the one non-slot arm that emits more than one token: it picks a
        # critical instance per class and builds a bag embedding per class, so
        # sizing the readout at K=1 would silently drop half of it.
        k_eff = num_classes if pooling == "dsmil" else 1

    head = build_readout(readout, dim, num_classes, k_eff)
    return MILModel(pool, head, pooling_name=pooling, encoder=encoder)


def slot_pooling_param_count(
    input_dim: int,
    dim: int,
    num_slots: int,
    iters: int = 3,
    init: str = "learnable",
) -> int:
    """Parameter count of the SlotAttention pooling module alone.

    Used to size the matched multi-head-ABMIL control. Built and discarded rather
    than derived by hand so it cannot drift out of sync with the module.
    """
    m = SlotAttention(
        num_slots=num_slots, dim=dim, input_dim=input_dim, iters=iters, init=init
    )
    return count_params(m)
