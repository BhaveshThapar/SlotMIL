"""Tests for the slot attention core.

The padding test is the important one. Slot attention normalises over slots, so
padded instances survive the softmax and then quietly contribute mass to the
weighted mean over instances -- rescaling every real weight. Nothing crashes, no
metric obviously breaks, and every long-bag result is subtly wrong. CT bags run
65-764 slices, so this would have hit real data constantly.
"""

from __future__ import annotations

import pytest
import torch

from slotmil.models.baselines import MultiHeadABMIL, count_params, matched_multihead_abmil
from slotmil.models.mil import build_model, slot_pooling_param_count
from slotmil.models.slot_attention import SlotAttention, slot_health


@pytest.fixture
def sa():
    torch.manual_seed(0)
    return SlotAttention(num_slots=4, dim=64, input_dim=32, iters=3, implicit=False).eval()


def test_output_shapes(sa):
    x = torch.randn(2, 20, 32)
    slots, attn = sa(x)
    assert slots.shape == (2, 4, 64)
    assert attn.shape == (2, 4, 20)


def test_softmax_is_over_slots(sa):
    """Each instance's attention must sum to 1 across slots -- that is the
    competition. If this ever normalises over instances instead, slots stop
    competing and the method reduces to multi-head attention."""
    x = torch.randn(2, 20, 32)
    _, attn = sa(x)
    torch.testing.assert_close(
        attn.sum(dim=1), torch.ones(2, 20), rtol=1e-4, atol=1e-4
    )


def test_permutation_invariance(sa):
    """Bags are sets; reordering slices must not change the pooled slots."""
    x = torch.randn(1, 30, 32)
    perm = torch.randperm(30)

    slots_a, attn_a = sa(x)
    slots_b, attn_b = sa(x[:, perm])

    torch.testing.assert_close(slots_a, slots_b, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(attn_a[:, :, perm], attn_b, rtol=1e-4, atol=1e-4)


def test_padding_invariance(sa):
    """A padded bag must give the same slots as the unpadded original.

    This is the silent-corruption guard described in the module docstring.
    """
    x = torch.randn(1, 25, 32)
    full_mask = torch.ones(1, 25, dtype=torch.bool)

    padded = torch.cat([x, torch.randn(1, 40, 32)], dim=1)  # junk in the pad region
    pad_mask = torch.zeros(1, 65, dtype=torch.bool)
    pad_mask[0, :25] = True

    slots_a, attn_a = sa(x, full_mask)
    slots_b, attn_b = sa(padded, pad_mask)

    torch.testing.assert_close(slots_a, slots_b, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(attn_a, attn_b[:, :, :25], rtol=1e-5, atol=1e-6)
    assert attn_b[:, :, 25:].abs().max() == 0, "padded positions must carry zero mass"


def test_padding_with_non_finite_junk(sa):
    """Even inf/NaN in the pad region must not leak in (0 * inf = NaN)."""
    x = torch.randn(1, 10, 32)
    junk = torch.full((1, 5, 32), float("inf"))
    padded = torch.cat([x, junk], dim=1)
    pad_mask = torch.zeros(1, 15, dtype=torch.bool)
    pad_mask[0, :10] = True

    slots, attn = sa(padded, pad_mask)
    assert torch.isfinite(slots).all()
    assert torch.isfinite(attn).all()


def test_implicit_diff_gradients_are_finite():
    """Implicit differentiation must produce usable gradients."""
    torch.manual_seed(0)
    m = SlotAttention(num_slots=4, dim=64, input_dim=32, iters=3, implicit=True)
    x = torch.randn(2, 20, 32, requires_grad=True)
    slots, _ = m(x)
    slots.sum().backward()

    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads), "all gradients are exactly zero"
    assert torch.isfinite(x.grad).all()


def test_implicit_and_vanilla_agree_on_forward():
    """Implicit diff must change the backward pass only.

    Guards the effective-depth confound: if implicit mode ran `iters` no-grad
    steps and then an extra grad step, it would silently get T+1 refinements
    against vanilla's T, and the vanilla-vs-implicit ablation would be measuring
    depth rather than gradient estimation.
    """
    torch.manual_seed(0)
    kw = dict(num_slots=4, dim=64, input_dim=32, iters=3, init="learnable")
    a = SlotAttention(**kw, implicit=True).eval()
    b = SlotAttention(**kw, implicit=False).eval()
    b.load_state_dict(a.state_dict())

    x = torch.randn(2, 20, 32)
    with torch.no_grad():
        slots_a, _ = a(x)
        slots_b, _ = b(x)
    torch.testing.assert_close(slots_a, slots_b, rtol=1e-5, atol=1e-6)


class TestQueryGradientFlow:
    """Implicit diff detaches the init, so 'learnable' queries only actually
    learn when BO-QSA straight-through is on. Documented here because running the
    random-vs-learnable init ablation without knowing this would compare two
    frozen random inits and conclude the init does not matter."""

    @staticmethod
    def _query_grad(implicit: bool, st: bool):
        import warnings

        torch.manual_seed(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = SlotAttention(
                num_slots=4, dim=32, input_dim=16, iters=3,
                implicit=implicit, init="learnable", bo_qsa_straight_through=st,
            )
        m(torch.randn(2, 12, 16))[0].sum().backward()
        return m.slots_query.grad

    def test_implicit_without_straight_through_freezes_queries(self):
        assert self._query_grad(implicit=True, st=False) is None

    def test_straight_through_restores_the_gradient(self):
        g = self._query_grad(implicit=True, st=True)
        assert g is not None and g.abs().sum() > 0

    def test_vanilla_trains_queries_normally(self):
        g = self._query_grad(implicit=False, st=False)
        assert g is not None and g.abs().sum() > 0

    def test_the_silent_combination_warns(self):
        with pytest.warns(RuntimeWarning, match="will NOT train"):
            SlotAttention(num_slots=4, dim=32, input_dim=16, implicit=True,
                          init="learnable", bo_qsa_straight_through=False)


def test_variable_num_slots_at_inference(sa):
    """K can change without rebuilding -- required by the K-sensitivity sweep."""
    x = torch.randn(1, 20, 32)
    for k in (2, 4, 8):
        slots, attn = sa(x, num_slots=k)
        assert slots.shape[1] == k and attn.shape[1] == k


def test_random_init_is_stochastic():
    torch.manual_seed(0)
    m = SlotAttention(num_slots=4, dim=64, input_dim=32, init="random", implicit=False).eval()
    x = torch.randn(1, 20, 32)
    with torch.no_grad():
        assert not torch.allclose(m(x)[0], m(x)[0])


def test_slot_health_flags_collapse():
    """Identical slots must report near-1.0 off-diagonal cosine."""
    slots = torch.ones(2, 4, 16)
    attn = torch.full((2, 4, 10), 0.25)
    h = slot_health(slots, attn)
    assert h["max_off_diag_cos"] > 0.99
    assert h["active_slots"] == 4.0


class TestParameterMatching:
    """Reviewer objection #1 hinges on the control being genuinely matched."""

    def test_matched_within_tolerance(self):
        target = slot_pooling_param_count(input_dim=768, dim=256, num_slots=8)
        m = matched_multihead_abmil(768, 256, 8, target_params=target)
        err = abs(count_params(m) - target) / target
        assert err <= 0.02, f"control is {err:.1%} off target"

    def test_mh_abmil_normalises_over_instances_not_slots(self):
        """The defining difference from slot attention: heads do not compete."""
        torch.manual_seed(0)
        m = MultiHeadABMIL(32, 64, num_heads=4, hidden=32).eval()
        x = torch.randn(2, 20, 32)
        _, attn = m(x)
        torch.testing.assert_close(
            attn.sum(dim=-1), torch.ones(2, 4), rtol=1e-4, atol=1e-4
        )

    def test_mh_abmil_respects_padding(self):
        torch.manual_seed(0)
        m = MultiHeadABMIL(32, 64, num_heads=4, hidden=32).eval()
        x = torch.randn(1, 10, 32)
        padded = torch.cat([x, torch.randn(1, 6, 32)], dim=1)
        mask = torch.zeros(1, 16, dtype=torch.bool)
        mask[0, :10] = True

        pooled_a, _ = m(x, torch.ones(1, 10, dtype=torch.bool))
        pooled_b, attn_b = m(padded, mask)
        torch.testing.assert_close(pooled_a, pooled_b, rtol=1e-5, atol=1e-6)
        assert attn_b[:, :, 10:].abs().max() == 0


@pytest.mark.parametrize("pooling", ["mean", "max", "abmil", "gated_abmil", "mh_abmil", "slot"])
@pytest.mark.parametrize("readout", ["gated", "max"])
def test_all_poolings_share_the_interface(pooling, readout):
    """Every pooling must be swappable without touching the training loop."""
    torch.manual_seed(0)
    model = build_model(
        pooling=pooling, input_dim=64, dim=32, num_classes=2,
        num_slots=4, readout=readout,
    ).eval()

    x = torch.randn(2, 15, 64)
    mask = torch.ones(2, 15, dtype=torch.bool)
    mask[1, 10:] = False

    out = model(x, mask)
    assert out["logits"].shape == (2, 2)
    assert out["attn"].shape[0] == 2
    assert torch.isfinite(out["logits"]).all()
    assert ("health" in out) == (pooling == "slot")
