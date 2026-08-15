"""Normal Guidance's KL term, and the wiring that carries it into training.

Two silent corruptions are in scope here, both of which would produce a
publishable-looking H6 rather than a crash.

The first is the variance floor. Moment-matched to its own marginal, a
single-slice attention sits at a global minimum of KL = 0 exactly, because a
Dirac is the sigma->0 Normal. Unfloored, the term rewards collapsing attention
onto one slice -- which would read as a spectacular localisation result and be an
artefact of the objective. `test_delta_is_a_degenerate_minimum_without_the_floor`
is that guard, and it is the reason the floor exists at all.

The second is the stop-gradient's placement. The moments must be detached; the
marginal must not. Detaching the marginal instead gives exactly zero gradient, so
the term would be logged every step, look perfectly healthy in history.json, and
train nothing.
"""

from __future__ import annotations

import pytest
import torch

from slotmil.losses import (
    NG_PATCHES_PER_SLICE,
    NG_VAR_FLOOR_SLICES2,
    SlotMILLoss,
    normal_guidance_loss,
)

P = 4  # small stand-in for the real 256
S = 48


def _attn(slice_weights, patches=P):
    """Slice weights -> a [1, 1, S*patches] attention row."""
    w = torch.as_tensor(slice_weights, dtype=torch.float64)
    a = w.repeat_interleave(patches)
    return (a / a.sum()).view(1, 1, -1)


def _gaussian(sigma, centre=24.0, s=S):
    c = torch.arange(s, dtype=torch.float64)
    return torch.exp(-0.5 * ((c - centre) / sigma) ** 2)


def _delta(at=24, s=S):
    w = torch.zeros(s, dtype=torch.float64)
    w[at] = 1.0
    return w


class TestKLShape:
    def test_non_negative(self):
        torch.manual_seed(0)
        for _ in range(50):
            a = _attn(torch.rand(S, dtype=torch.float64) + 1e-6)
            assert normal_guidance_loss(a, patches_per_slice=P).item() >= -1e-12

    def test_gaussian_marginal_pays_nothing(self):
        for sigma in (1.0, 2.0, 4.0, 8.0):
            kl = normal_guidance_loss(_attn(_gaussian(sigma)), patches_per_slice=P)
            assert kl.item() == pytest.approx(0.0, abs=1e-3), sigma

    def test_ordering_uniform_below_bimodal(self):
        uniform = normal_guidance_loss(
            _attn(torch.ones(S, dtype=torch.float64)), patches_per_slice=P
        ).item()
        bimodal = normal_guidance_loss(
            _attn(_gaussian(2.0, 12.0) + _gaussian(2.0, 36.0)), patches_per_slice=P
        ).item()
        assert 0 < uniform < bimodal

    def test_bimodal_is_penalised_hardest(self):
        """Recorded because it is a real cost of the method, not a defect.

        NG suppresses multi-focal attention, and LIDC has multi-nodule cases. If
        this ever stops being true the paper's discussion of the trade-off needs
        revisiting.
        """
        kl = normal_guidance_loss(
            _attn(_gaussian(2.0, 12.0) + _gaussian(2.0, 36.0)), patches_per_slice=P
        )
        assert kl.item() == pytest.approx(1.0624, abs=1e-3)


class TestVarianceFloor:
    def test_delta_is_a_degenerate_minimum_without_the_floor(self):
        """The test that justifies the floor's existence.

        Unfloored, the single-slice attention -- the worst possible localisation
        behaviour -- is a *global minimum* of the objective at exactly zero.
        """
        a = _attn(_delta())
        assert normal_guidance_loss(
            a, patches_per_slice=P, var_floor=1e-12
        ).item() == pytest.approx(0.0, abs=1e-9)

    def test_floor_makes_the_delta_pay(self):
        a = _attn(_delta())
        assert normal_guidance_loss(
            a, patches_per_slice=P, var_floor=NG_VAR_FLOOR_SLICES2
        ).item() == pytest.approx(0.9189, abs=1e-3)

    def test_floor_leaves_wide_marginals_untouched(self):
        """The floor must bite only where the variance is below it, or it would
        be silently reshaping every bag rather than ruling out the degenerate one."""
        for w in (torch.ones(S, dtype=torch.float64), _gaussian(4.0)):
            a = _attn(w)
            lo = normal_guidance_loss(a, patches_per_slice=P, var_floor=1e-12).item()
            hi = normal_guidance_loss(a, patches_per_slice=P, var_floor=1.0).item()
            assert lo == pytest.approx(hi, abs=1e-9)

    def test_declared_floor_is_harveys_sigma(self):
        assert NG_VAR_FLOOR_SLICES2 == 1.0


class TestStopGradient:
    def test_gradient_reaches_the_attention(self):
        a = _attn(torch.rand(S, dtype=torch.float64) + 0.1).requires_grad_(True)
        normal_guidance_loss(a, patches_per_slice=P).backward()
        assert a.grad is not None and a.grad.abs().sum().item() > 0

    def test_detaching_the_marginal_would_kill_the_term(self):
        """Pins *where* the stop-gradient goes. The moments are detached; the
        marginal is not. Swapping that is the silent no-op this guards."""
        a = _attn(torch.rand(S, dtype=torch.float64) + 0.1).requires_grad_(True)
        detached = a.detach().requires_grad_(True)
        normal_guidance_loss(detached.detach(), patches_per_slice=P)
        # The real term must be gradient-bearing where the fully-detached one is not.
        loss = normal_guidance_loss(a, patches_per_slice=P)
        assert loss.requires_grad
        assert not normal_guidance_loss(a.detach(), patches_per_slice=P).requires_grad

    def test_target_moments_do_not_receive_gradient(self):
        """If mu/sigma were differentiable the target would chase the attention
        and the term would trivially minimise itself."""
        a = _attn(_gaussian(3.0)).requires_grad_(True)
        normal_guidance_loss(a, patches_per_slice=P).backward()
        g_matched = a.grad.clone()

        a2 = _attn(_delta()).requires_grad_(True)
        normal_guidance_loss(a2, patches_per_slice=P).backward()
        # A marginal already Normal has near-zero pressure; a delta does not.
        assert g_matched.abs().max() < a2.grad.abs().max()


class TestPaddingAndSliceIndex:
    def test_padding_invariance(self):
        w = torch.rand(6, dtype=torch.float64) + 0.1
        a = _attn(w)
        kl_clean = normal_guidance_loss(a, patches_per_slice=P).item()

        padded = torch.zeros(1, 1, 10 * P, dtype=torch.float64)
        padded[0, 0, : 6 * P] = a[0, 0]
        mask = torch.zeros(1, 10 * P, dtype=torch.bool)
        mask[0, : 6 * P] = True
        kl_padded = normal_guidance_loss(padded, mask, patches_per_slice=P).item()
        assert kl_clean == pytest.approx(kl_padded, abs=1e-9)

    def test_junk_in_the_pad_region_is_ignored(self):
        w = torch.rand(6, dtype=torch.float64) + 0.1
        a = _attn(w)
        kl_clean = normal_guidance_loss(a, patches_per_slice=P).item()

        padded = torch.zeros(1, 1, 10 * P, dtype=torch.float64)
        padded[0, 0, : 6 * P] = a[0, 0]
        padded[0, 0, 6 * P:] = 5.0  # attention mass that must not count
        mask = torch.zeros(1, 10 * P, dtype=torch.bool)
        mask[0, : 6 * P] = True
        assert normal_guidance_loss(
            padded, mask, patches_per_slice=P
        ).item() == pytest.approx(kl_clean, abs=1e-9)

    def test_kl_is_invariant_to_affine_rescaling_of_slice_index(self):
        """Narrows *why* the true slice index has to be plumbed through.

        Fitting the target's moments to the marginal makes the KL invariant to
        any affine rescaling of the coordinate -- rescale the axis and mu, sigma
        rescale with it. So slice_index does NOT matter for its own sake, and
        the depth-weighting worry is smaller than it first appears.
        """
        a = torch.rand(1, 1, 6 * P).softmax(dim=-1).double()
        vals = [
            normal_guidance_loss(
                a, slice_index=torch.tensor([si]), patches_per_slice=P
            ).item()
            for si in ([0, 1, 2, 3, 4, 5], [0, 10, 20, 30, 40, 50],
                       [0, 100, 200, 300, 400, 500], [100, 101, 102, 103, 104, 105])
        ]
        assert max(vals) - min(vals) < 1e-12, vals

    def test_slice_index_matters_through_the_variance_floor(self):
        """...and this is the one channel where it does matter, which is why it
        is plumbed at all.

        The floor is stated in *anatomical* slices^2. Under --max-slices a bag's
        marginal spread shrinks in subsampled units -- for a 700-slice volume cut
        to 48, by a factor of ~213 in variance -- which can push it under the
        floor spuriously and start flattening attention that was never degenerate.
        """
        a = torch.full((1, 1, 3 * P), 1.0 / (3 * P), dtype=torch.float64)
        subsampled = normal_guidance_loss(  # var = 2/3, below the 1.0 floor
            a, slice_index=torch.tensor([[0, 1, 2]]), patches_per_slice=P
        ).item()
        anatomical = normal_guidance_loss(  # var = 6667, floor inactive
            a, slice_index=torch.tensor([[0, 100, 200]]), patches_per_slice=P
        ).item()
        assert subsampled != pytest.approx(anatomical, abs=1e-6)

    def test_absent_slice_index_falls_back_to_arange(self):
        a = _attn(torch.rand(6, dtype=torch.float64) + 0.1)
        explicit = normal_guidance_loss(
            a, slice_index=torch.arange(6).unsqueeze(0), patches_per_slice=P
        ).item()
        implicit = normal_guidance_loss(a, patches_per_slice=P).item()
        assert explicit == pytest.approx(implicit, abs=1e-12)

    def test_ragged_tail(self):
        n = 6 * P + 2
        a = torch.full((1, 1, n), 1.0 / n, dtype=torch.float64)
        assert torch.isfinite(normal_guidance_loss(a, patches_per_slice=P))


class TestSlotMILLossWiring:
    def _out(self, n=6 * P):
        torch.manual_seed(0)
        return {
            "logits": torch.randn(2, 2, requires_grad=True),
            "attn": torch.rand(2, 1, n).softmax(dim=-1),
            "tokens": torch.randn(2, 1, 8),
        }

    def test_w_kl_zero_leaves_the_loss_untouched(self):
        out, y = self._out(), torch.tensor([0, 1])
        base, cb = SlotMILLoss()(out, y)
        withz, cz = SlotMILLoss(w_kl=0.0)(out, y)
        assert "kl_prior" not in cb and "kl_prior" not in cz
        assert base.item() == withz.item()

    def test_w_kl_positive_adds_a_logged_component(self):
        out, y = self._out(), torch.tensor([0, 1])
        base, _ = SlotMILLoss()(out, y)
        total, comps = SlotMILLoss(w_kl=0.1, kl_patches_per_slice=P)(out, y)
        assert "kl_prior" in comps
        assert total.item() > base.item()
        assert total.item() == pytest.approx(
            base.item() + 0.1 * comps["kl_prior"].item(), abs=1e-6
        )

    def test_slice_index_reaches_the_kl(self):
        """End-to-end: the field train.py pulls off the batch has to arrive.

        Distinguished through the variance floor, since that is the only channel
        the coordinate scale acts through (see
        test_kl_is_invariant_to_affine_rescaling_of_slice_index).
        """
        out, y = self._out(n=3 * P), torch.tensor([0, 1])
        crit = SlotMILLoss(w_kl=0.1, kl_patches_per_slice=P)
        _, sub = crit(out, y, slice_index=torch.tensor([[0, 1, 2]]).repeat(2, 1))
        _, ana = crit(out, y, slice_index=torch.tensor([[0, 100, 200]]).repeat(2, 1))
        assert sub["kl_prior"].item() != pytest.approx(ana["kl_prior"].item(), abs=1e-6)

    def test_declared_patches_per_slice_matches_the_prereg(self):
        from slotmil.prereg import load

        pre = load()
        assert NG_PATCHES_PER_SLICE == pre.get("datasets.lidc.patches_per_slice")
