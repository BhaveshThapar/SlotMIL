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
    raise ValueError(f"unknown pooling {name!r}")
