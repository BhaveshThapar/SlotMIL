"""Iterative slot attention (Locatello et al., NeurIPS 2020) as a MIL pooling bottleneck.

Follows the algorithm in plan.md Part 4 lines 90-98. Two departures from the
common reference implementations (e.g. lucidrains/slot-attention), both of which
matter for CT bags:

1. Implicit differentiation (Chang et al., NeurIPS 2022). T iterations run under
   no_grad, then a single update step carries the gradient. O(1) backward memory
   and a stabilised Jacobian, which removes the need for gradient clipping and LR
   warmup. On by default.

2. Variable-length bag masking. CT volumes give bags of 65-764 slices. Because the
   softmax is taken over slots, padded instances do not corrupt the softmax itself
   -- but they *do* contribute mass to the subsequent weighted mean over instances,
   which silently rescales every real weight. Padded positions are therefore zeroed
   after the softmax and before the sum over N.

   Note the asymmetry with ordinary attention: you cannot mask the *logits* with
   -inf here, because the softmax runs over the slot axis and every slot at a
   padded position would be -inf, producing NaN. The mask has to be applied to the
   post-softmax attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """Competitive iterative pooling of a set of instances into K slots.

    Args:
        num_slots: K, the number of slots. Overridable per-forward-call, since the
            mechanism is slot-count agnostic (used by the K-sensitivity sweep).
        dim: slot dimensionality D.
        input_dim: instance feature dimensionality. Defaults to ``dim``.
        iters: T, number of refinement iterations.
        hidden_dim: width of the residual MLP. Defaults to ``4 * dim``.
        implicit: use implicit differentiation (Chang et al. 2022).
        init: ``"learnable"`` for BO-QSA style learnable queries, ``"random"`` for
            the original Gaussian sampling. plan.md line 37 notes random init can
            beat learnable init in some regimes, so this is an ablation axis and
            not a settled default.
        bo_qsa_straight_through: route gradient back to the learnable queries via
            the straight-through estimator (BO-QSA, ICLR 2023). Only meaningful
            when ``init="learnable"``.
    """

    def __init__(
        self,
        num_slots: int,
        dim: int,
        input_dim: int | None = None,
        iters: int = 3,
        hidden_dim: int | None = None,
        implicit: bool = True,
        init: str = "learnable",
        bo_qsa_straight_through: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        if init not in ("learnable", "random"):
            raise ValueError(f"init must be 'learnable' or 'random', got {init!r}")

        input_dim = input_dim if input_dim is not None else dim
        hidden_dim = hidden_dim if hidden_dim is not None else 4 * dim

        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.implicit = implicit
        self.init = init
        self.bo_qsa_straight_through = bo_qsa_straight_through
        self.eps = eps
        self.scale = dim**-0.5

        if init == "learnable":
            self.slots_query = nn.Parameter(torch.zeros(1, num_slots, dim))
            nn.init.xavier_uniform_(self.slots_query)
        else:
            self.slots_mu = nn.Parameter(torch.zeros(1, 1, dim))
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.xavier_uniform_(self.slots_mu)
            nn.init.xavier_uniform_(self.slots_logsigma)

        if init == "learnable" and implicit and not bo_qsa_straight_through:
            # Implicit differentiation consumes the init inside no_grad and then
            # detaches, so slots_query receives no gradient and stays at its
            # random initialisation for the whole run. That is faithful to Chang
            # et al. -- the fixed point is meant to be init-independent -- but it
            # makes "learnable" a misnomer here, and it silently voids the
            # random-vs-learnable init ablation (plan.md line 114), because both
            # arms would then be frozen random inits. BO-QSA's straight-through
            # estimator is the intended fix.
            import warnings

            warnings.warn(
                "init='learnable' with implicit=True and "
                "bo_qsa_straight_through=False: the slot queries will NOT train "
                "(no gradient path). Set bo_qsa_straight_through=True to train "
                "them, or init='random' to be explicit about not doing so.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(input_dim, dim, bias=False)
        self.to_v = nn.Linear(input_dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def _init_slots(self, batch: int, num_slots: int, device, dtype) -> torch.Tensor:
        if self.init == "learnable":
            slots = self.slots_query
            if num_slots != slots.shape[1]:
                # Tile / truncate so K can be varied at inference without retraining.
                reps = (num_slots + slots.shape[1] - 1) // slots.shape[1]
                slots = slots.repeat(1, reps, 1)[:, :num_slots]
            return slots.expand(batch, -1, -1).to(device=device, dtype=dtype)

        mu = self.slots_mu.expand(batch, num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(batch, num_slots, -1)
        return mu + sigma * torch.randn(
            batch, num_slots, self.dim, device=device, dtype=dtype
        )

    def _step(
        self,
        slots: torch.Tensor,  # B, K, D
        k: torch.Tensor,  # B, N, D
        v: torch.Tensor,  # B, N, D
        pad_mask: torch.Tensor | None,  # B, N (True = real instance)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, num_slots, _ = slots.shape
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        # B, K, N
        logits = torch.einsum("bkd,bnd->bkn", q, k) * self.scale

        # THE competition: normalise across slots, so each instance distributes a
        # unit of "explanation" among the K slots rather than each slot
        # independently scoring instances.
        attn = logits.softmax(dim=1)

        if pad_mask is not None:
            attn = attn.masked_fill(~pad_mask.unsqueeze(1), 0.0)

        # Weighted mean over instances. eps sits only in the denominator so that a
        # padded bag stays bit-identical to its unpadded twin (padded positions
        # contribute exactly zero, not eps).
        weights = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
        updates = torch.einsum("bkn,bnd->bkd", weights, v)

        slots = self.gru(
            updates.reshape(batch * num_slots, self.dim),
            slots_prev.reshape(batch * num_slots, self.dim),
        ).view(batch, num_slots, self.dim)

        slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots, attn

    def forward(
        self,
        inputs: torch.Tensor,  # B, N, input_dim
        pad_mask: torch.Tensor | None = None,  # B, N, True = real
        num_slots: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(slots [B, K, D], attn [B, K, N])``.

        ``attn`` is the post-softmax, mask-zeroed attention -- this is the object
        that gets reshaped to the ViT patch grid and upsampled for localisation.
        """
        batch = inputs.shape[0]
        num_slots = num_slots or self.num_slots

        x = self.norm_input(inputs)
        k = self.to_k(x)
        v = self.to_v(x)

        if pad_mask is not None:
            # Guarantees 0 * v_pad == 0 even if the collate left junk in the pad
            # region; without this, a stray inf/NaN there would poison the update.
            v = v.masked_fill(~pad_mask.unsqueeze(-1), 0.0)

        slots = self._init_slots(batch, num_slots, inputs.device, inputs.dtype)

        if self.implicit:
            # iters-1 steps without grad, then one carrying the gradient, so the
            # total refinement count is exactly `iters` -- the same as vanilla.
            # Running a full `iters` under no_grad and then a further grad step
            # would give implicit mode one extra refinement, quietly confounding
            # the vanilla-vs-implicit ablation (plan.md line 114) with a
            # difference in effective depth rather than in gradient estimation.
            with torch.no_grad():
                for _ in range(self.iters - 1):
                    slots, _ = self._step(slots, k, v, pad_mask)
            slots = slots.detach()
            slots, attn = self._step(slots, k, v, pad_mask)
        else:
            for _ in range(self.iters):
                slots, attn = self._step(slots, k, v, pad_mask)

        if self.init == "learnable" and self.bo_qsa_straight_through:
            query = self._init_slots(batch, num_slots, inputs.device, inputs.dtype)
            slots = slots + query - query.detach()

        return slots, attn


def slot_health(slots: torch.Tensor, attn: torch.Tensor, mass_thresh: float = 0.01) -> dict:
    """Collapse / duplication early-warning signals (plan.md line 104).

    ``active_slots``   -- slots holding more than ``mass_thresh`` of the attention
                          mass. Dead slots show up here first.
    ``max_off_diag_cos`` -- peak pairwise slot cosine similarity. Approaching 1.0
                          means collapse (all slots identical) or duplication.
    """
    with torch.no_grad():
        mass = attn.sum(dim=-1)  # B, K
        mass = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        active = (mass > mass_thresh).float().sum(dim=-1)

        s = F.normalize(slots, dim=-1)
        sim = torch.einsum("bkd,bjd->bkj", s, s)
        k = sim.shape[-1]
        off = sim.masked_fill(
            torch.eye(k, dtype=torch.bool, device=sim.device).unsqueeze(0), float("-inf")
        )
        max_off = off.flatten(1).max(dim=-1).values if k > 1 else torch.zeros_like(active)

        return {
            "active_slots": active.mean().item(),
            "max_off_diag_cos": max_off.mean().item(),
            "slot_mass_entropy": (
                -(mass.clamp_min(1e-8) * mass.clamp_min(1e-8).log()).sum(-1).mean().item()
            ),
        }
