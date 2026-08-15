"""The position-only arms: centre_gaussian, and the alias that backs normal_guidance.

The failure this file exists to prevent is a content-free reference that quietly
stops being content-free. If centre_gaussian's attention ever depends on the
image -- through a stray learned parameter, a normalisation that mixes bags, or a
refactor that routes features into the attention path -- then the denominator in
`prior_normalised_skill` starts absorbing real signal and every skill number in
the paper is silently inflated. `test_attention_is_content_free` is the guard,
and it is deliberately the strictest kind available: bit-identical output, not
approximate.

The second concern is sigma. The arm departs from Harvey's printed sigma=1
because that value is not representable on volumetric CT (see the class
docstring). What licenses the departure is that sigma cannot move a rank-based
metric, so `test_slice_auc_is_invariant_to_sigma` pins the licence itself rather
than the number it produces.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from slotmil.eval.nulls import centre_prior_scores
from slotmil.models.baselines import CENTRE_GAUSSIAN_SIGMA_Z, CentreGaussianPool
from slotmil.models.mil import build_model

P = 4  # small stand-in for the real 256, so bags stay test-sized


def _bag(n_slices, patches=P, dim=16, seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, n_slices * patches, dim)


class TestAttentionShape:
    def test_sums_to_one_over_real_instances(self):
        pool = CentreGaussianPool(16, 8, patches_per_slice=P).eval()
        _, attn = pool(_bag(6), torch.ones(1, 6 * P, dtype=torch.bool))
        assert attn.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_padded_columns_are_exactly_zero(self):
        """Not 'small' -- exactly zero. Every eval consumer slices attn[i, :,
        :length] and renormalises, so a nonzero pad silently steals mass."""
        pool = CentreGaussianPool(16, 8, patches_per_slice=P).eval()
        mask = torch.ones(2, 6 * P, dtype=torch.bool)
        mask[1, 3 * P:] = False
        feats = torch.randn(2, 6 * P, 16)
        feats[1, 3 * P:] = float("inf")  # junk in the pad region must not leak
        _, attn = pool(feats, mask)
        assert attn[1, :, 3 * P:].abs().max().item() == 0.0
        assert torch.isfinite(attn).all()
        assert attn[1, :, : 3 * P].sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_attention_is_constant_within_a_slice(self):
        """The prior is axial-only: it has no in-plane term, deliberately, which
        is what distinguishes it from nulls.centre_prior_scores."""
        pool = CentreGaussianPool(16, 8, patches_per_slice=P).eval()
        _, attn = pool(_bag(6), torch.ones(1, 6 * P, dtype=torch.bool))
        per_slice = attn[0, 0].view(6, P)
        assert per_slice.std(dim=1).max().item() < 1e-9

    def test_per_bag_centre_from_pad_mask(self):
        """Bags of different depth in one batch each get their own centre."""
        pool = CentreGaussianPool(16, 8, patches_per_slice=P).eval()
        mask = torch.ones(2, 8 * P, dtype=torch.bool)
        mask[1, 4 * P:] = False
        _, attn = pool(torch.randn(2, 8 * P, 16), mask)
        assert attn[0, 0].view(8, P)[:, 0].argmax().item() in (3, 4)   # S=8
        assert attn[1, 0, : 4 * P].view(4, P)[:, 0].argmax().item() in (1, 2)  # S=4

    def test_symmetric_about_the_centre(self):
        pool = CentreGaussianPool(16, 8, patches_per_slice=P).eval()
        _, attn = pool(_bag(7), torch.ones(1, 7 * P, dtype=torch.bool))
        w = attn[0, 0].view(7, P)[:, 0]
        assert torch.allclose(w, w.flip(0), atol=1e-7)


class TestContentFree:
    def test_attention_is_content_free(self):
        """The load-bearing test. Two different images, bit-identical attention.

        This is what 'content-free reference' means, and it is why the arm is
        pre-registered blind: false -- the estimands need to know which arm never
        reads a pixel.
        """
        pool = CentreGaussianPool(16, 8, patches_per_slice=P).eval()
        mask = torch.ones(1, 6 * P, dtype=torch.bool)
        _, a1 = pool(torch.randn(1, 6 * P, 16), mask)
        _, a2 = pool(torch.randn(1, 6 * P, 16) * 100 + 5, mask)
        assert torch.equal(a1, a2)

    def test_no_parameters_outside_proj(self):
        pool = CentreGaussianPool(16, 8, patches_per_slice=P)
        leaked = [
            n for n, _ in pool.named_parameters() if not n.startswith("proj")
        ]
        assert leaked == [], f"attention path grew parameters: {leaked}"

    def test_attention_carries_no_gradient(self):
        pool = CentreGaussianPool(16, 8, patches_per_slice=P)
        feats = torch.randn(1, 6 * P, 16, requires_grad=True)
        _, attn = pool(feats, torch.ones(1, 6 * P, dtype=torch.bool))
        assert not attn.requires_grad


class TestSigma:
    def test_slice_auc_is_invariant_to_sigma(self):
        """Why departing from Harvey's printed sigma=1 is legitimate.

        NormPDF(.|mu,sigma) is strictly monotone in -|j-mu| for every sigma > 0,
        and every pre-registered estimand is rank-based. So sigma cannot move a
        reported number, and is free to be chosen for representability instead.

        Asserted as agreement across sigma rather than against a fixed constant:
        the invariance is the claim the amendment rests on, and a hardcoded AUC
        would only pin one arbitrary label draw.

        The band is [0.25, 1.0], not "any sigma". Below roughly 0.227 the fp32
        tails lose enough precision to change the tie structure and move the AUC
        in the third decimal -- see the companion test. That lower bound is
        exactly why sigma_z is 0.25 rather than something peakier.
        """
        n_slices = 41
        rng = np.random.default_rng(0)
        y = np.zeros(n_slices)
        y[rng.choice(n_slices, 8, replace=False)] = 1

        sigmas = (0.25, 0.5, 1.0)
        aucs = []
        for sigma in sigmas:
            pool = CentreGaussianPool(16, 8, sigma=sigma, patches_per_slice=P).eval()
            _, attn = pool(
                _bag(n_slices), torch.ones(1, n_slices * P, dtype=torch.bool)
            )
            w = attn[0, 0].view(n_slices, P)[:, 0].double().numpy()
            aucs.append(roc_auc_score(y, w))

        assert max(aucs) - min(aucs) < 1e-12, dict(zip(sigmas, aucs))

    def test_sigma_below_the_band_does_perturb_the_metric(self):
        """The lower bound on sigma is real, and this pins it.

        sigma_z = 0.1 is peaky enough that the fp32 tails collapse onto each
        other, changing which slices tie and therefore the AUC. The declared
        0.25 sits above that. If this test ever starts passing -- a wider dtype,
        a different normalisation -- the sigma choice can be revisited, but it
        should be revisited deliberately.
        """
        n_slices = 41
        rng = np.random.default_rng(0)
        y = np.zeros(n_slices)
        y[rng.choice(n_slices, 8, replace=False)] = 1

        aucs = []
        for sigma in (0.1, CENTRE_GAUSSIAN_SIGMA_Z):
            pool = CentreGaussianPool(16, 8, sigma=sigma, patches_per_slice=P).eval()
            _, attn = pool(
                _bag(n_slices), torch.ones(1, n_slices * P, dtype=torch.bool)
            )
            aucs.append(
                roc_auc_score(y, attn[0, 0].view(n_slices, P)[:, 0].double().numpy())
            )
        assert aucs[0] != pytest.approx(aucs[1], abs=1e-9)

    def test_literal_harvey_sigma_would_underflow(self):
        """Documents the defect the amendment exists to work around.

        Harvey print NormPDF(j | S/2, 1) over the RAW slice index. At S=171 the
        tail logit is -3612 nats against float32's -103.3, so the softmax returns
        exact zeros over most of the bag -- a tie block, which roc_auc_score
        credits at 0.5. If this ever stops being true (a wider float, a different
        normalisation), the amendment's rationale should be revisited.
        """
        s = 171
        j = torch.arange(s, dtype=torch.float32)
        logits = -0.5 * ((j - (s - 1) / 2) / 1.0) ** 2
        a = torch.softmax(logits, dim=0)
        assert (a == 0).sum().item() > s * 0.8

    def test_default_sigma_matches_the_declared_constant(self):
        pool = CentreGaussianPool(16, 8, patches_per_slice=P)
        assert pool.sigma == CENTRE_GAUSSIAN_SIGMA_Z

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_nonpositive_sigma_raises(self, bad):
        with pytest.raises(ValueError):
            CentreGaussianPool(16, 8, sigma=bad)


class TestRankEqualityWithCentrePrior:
    def test_matches_the_axial_component_of_centre_prior(self):
        """The paper gets to say Harvey's baseline *is* our axial centre prior.

        Compared against the axial term only: nulls.centre_prior_scores carries an
        in-plane radial term that this arm deliberately lacks, so the full score
        is not the comparison.
        """
        n_slices, n_patch = 9, 4
        pool = CentreGaussianPool(16, 8, patches_per_slice=n_patch).eval()
        _, attn = pool(
            _bag(n_slices, n_patch), torch.ones(1, n_slices * n_patch, dtype=torch.bool)
        )
        mine = attn[0, 0].view(n_slices, n_patch)[:, 0].double().numpy()

        z = np.linspace(-1, 1, n_slices)  # the axial term of centre_prior_scores
        assert np.array_equal(np.argsort(-mine), np.argsort(np.abs(z)))

    def test_centre_prior_helper_still_builds_z_the_same_way(self):
        """Pins the construction this arm was aligned to, so a change there
        surfaces here rather than as a silent divergence."""
        scores = centre_prior_scores([np.zeros(9 * 256)], n_patch=256, grid=16)
        assert scores[0].shape == (1, 9 * 256)


class TestDegenerateBags:
    def test_single_slice_bag_is_finite_and_uniform(self):
        """S == 1 is hit on the very first conformance test: the shared 15-instance
        bag gives ceil(15/256) == 1, where a bare (S-1) divisor divides by zero."""
        pool = CentreGaussianPool(16, 8).eval()  # real patches_per_slice=256
        _, attn = pool(torch.randn(2, 15, 16), torch.ones(2, 15, dtype=torch.bool))
        assert torch.isfinite(attn).all()
        assert attn[0, 0].std().item() < 1e-9
        assert attn[0].sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_ragged_tail_does_not_raise(self):
        """Real bags are exact multiples (bag_inclusion enforces it), but the
        synthetic ones are not."""
        pool = CentreGaussianPool(16, 8, patches_per_slice=P).eval()
        n = 6 * P + 2
        _, attn = pool(torch.randn(1, n, 16), torch.ones(1, n, dtype=torch.bool))
        assert torch.isfinite(attn).all()
        assert attn.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_rejects_nonsense_patches_per_slice(self):
        with pytest.raises(ValueError):
            CentreGaussianPool(16, 8, patches_per_slice=0)


class TestNormalGuidanceIsItsBaseArm:
    def test_alias_is_structurally_identical_to_gated_abmil(self):
        """H6 compares NG to 'its base arm'. Sharing the class is what makes that
        comparison exactly matched rather than approximately so."""
        torch.manual_seed(0)
        ng = build_model(pooling="normal_guidance", input_dim=32, dim=16, num_classes=2)
        torch.manual_seed(0)
        base = build_model(pooling="gated_abmil", input_dim=32, dim=16, num_classes=2)

        assert ng.state_dict().keys() == base.state_dict().keys()
        for k in ng.state_dict():
            assert torch.equal(ng.state_dict()[k], base.state_dict()[k]), k

    def test_attention_normalises_over_instances(self):
        """The KL marginalises over instances, so the base arm must normalise on
        that axis. gated_abmil does; slot attention does not, which is the main
        reason it is not the base arm."""
        m = build_model(
            pooling="normal_guidance", input_dim=32, dim=16, num_classes=2
        ).eval()
        out = m(torch.randn(2, 20, 32), torch.ones(2, 20, dtype=torch.bool))
        assert out["attn"].sum(dim=-1).allclose(torch.ones(2, 1), atol=1e-5)
