"""TransMIL: the pooling contract, the squaring pad, and the reported attention.

This arm has three failure modes that a bag of 12 instances would not show and
that would each produce a plausible number rather than a crash.

*The squaring pad can leak.* PPEG convolves a 7x7 depthwise kernel over the
squared grid, and the pad occupies whatever is left of ``ceil(sqrt(N))**2`` after
the real instances. A nonzero pad does not merely occupy its own column -- the
convolution carries it into real columns, where the attention mask can no longer
remove it, because by then it is inside a real position's features.

*The reported attention can drift from the model.* The block output uses the
Nystrom approximation and the reported row is the exact softmax, deliberately:
the contract requires a distribution, and a Nystrom row is neither normalised nor
non-negative. That substitution is only honest if the two agree, so the gap is
measured here rather than asserted in a docstring.

*The class token can take mass that then goes missing.* The class query attends
to itself as well as to instances, and that mass is not part of any instance-level
estimand. It is dropped and the row renormalised, so the reported row must still
sum to 1 over real instances.
"""

from __future__ import annotations

import math

import pytest
import torch

from slotmil.models.baselines import TransMIL
from slotmil.models.mil import build_model


def _bag(b=2, n=15, dim=16, cut=7, junk=1e4, seed=0):
    """A batch whose second bag is padded from ``cut``, with junk in the pad."""
    torch.manual_seed(seed)
    feats = torch.randn(b, n, dim)
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[1, cut:] = False
    feats[1, cut:] = junk
    return feats, mask


class TestContract:
    def test_padded_columns_are_exactly_zero(self):
        feats, mask = _bag()
        m = build_model(pooling="transmil", input_dim=16, dim=32,
                        num_classes=2).eval()
        attn = m(feats, mask)["attn"]
        assert attn[1, :, 7:].abs().max().item() == 0.0
        assert torch.isfinite(attn).all()

    def test_attention_normalises_over_instances(self):
        feats, mask = _bag()
        m = build_model(pooling="transmil", input_dim=16, dim=32,
                        num_classes=2).eval()
        attn = m(feats, mask)["attn"]
        assert attn.sum(-1).allclose(torch.ones(2, 1), atol=1e-5)
        assert (attn >= 0).all()

    def test_one_token(self):
        """K = 1. There is exactly one class token, so unlike DSMIL the published
        shape is already a single token and mil.py's k_eff=1 fallthrough is right."""
        feats, mask = _bag()
        out = build_model(pooling="transmil", input_dim=16, dim=32,
                          num_classes=2).eval()(feats, mask)
        assert out["attn"].shape[1] == 1 and out["tokens"].shape[1] == 1

    def test_no_instance_logits_key(self):
        """The two-tuple contract widens only for InstanceScoringPool arms."""
        feats, mask = _bag()
        out = build_model(pooling="transmil", input_dim=16, dim=32,
                          num_classes=2).eval()(feats, mask)
        assert "instance_logits" not in out and "health" not in out

    @pytest.mark.parametrize("n", [1, 4, 9, 15, 16, 17, 100])
    def test_squaring_survives_every_length(self, n):
        """``ceil(sqrt(N))**2`` is exact for a square N and pads otherwise; a
        bare ``int(sqrt(N))`` is off by one on the exact squares."""
        torch.manual_seed(0)
        m = TransMIL(16, 32, heads=4, num_landmarks=8).eval()
        _, attn = m(torch.randn(1, n, 16), torch.ones(1, n, dtype=torch.bool))
        side = math.isqrt(n - 1) + 1 if n > 0 else 1
        assert side * side >= n and (side - 1) ** 2 < n
        assert attn.shape == (1, 1, n)
        assert attn.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_gradients_reach_the_class_token(self):
        feats, mask = _bag()
        m = build_model(pooling="transmil", input_dim=16, dim=32, num_classes=2)
        out = m(feats, mask)
        torch.nn.functional.cross_entropy(out["logits"], torch.tensor([1, 0])).backward()
        g = m.pooling.cls_token.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


class TestBatchInvariance:
    """A bag's output must not depend on which bags it was batched with.

    This is the arm's sharpest edge and it was wrong first. The squaring side is
    ``ceil(sqrt(n))``, and taking ``n`` from the batch's padded width rather than
    the bag's own instance count made a 10-instance bag square to 4x4 alone and
    10x10 beside a deep scan -- different PPEG neighbourhoods over identical
    instances, measured at 0.012 on attention and 0.64 on the pooled token.

    It matters beyond tidiness. Evaluation batches whole bags with no
    subsampling, so under the batched form a scan's attention -- the thing every
    localisation estimand ranks -- would have depended on the arbitrary order the
    DataLoader happened to group it in.
    """

    def _alone_vs_padded(self, n, width, seed=0):
        torch.manual_seed(seed)
        m = TransMIL(16, 32, heads=4, num_landmarks=8).eval()
        f = torch.randn(1, n, 16)
        pm = torch.ones(1, n, dtype=torch.bool)
        wide_f = torch.zeros(1, width, 16)
        wide_f[0, :n] = f[0]
        wide_pm = torch.zeros(1, width, dtype=torch.bool)
        wide_pm[0, :n] = True
        with torch.no_grad():
            t_a, a_a = m(f, pm)
            t_b, a_b = m(wide_f, wide_pm)
        return (t_a, a_a[0, 0, :n]), (t_b, a_b[0, 0, :n]), a_b

    @pytest.mark.parametrize("n,width", [(10, 100), (17, 64), (9, 81), (4, 5)])
    def test_a_bag_scores_the_same_alone_as_padded(self, n, width):
        (t_a, a_a), (t_b, a_b), _ = self._alone_vs_padded(n, width)
        assert torch.equal(a_a, a_b)
        assert torch.equal(t_a, t_b)

    def test_padding_still_carries_no_mass(self):
        _, _, full = self._alone_vs_padded(10, 100)
        assert full[0, 0, 10:].abs().max().item() == 0.0
        assert full.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_autocast_does_not_downcast_the_attention(self):
        """Training runs under AMP, and the per-bag path assembles its output row
        rather than getting one from `_masked_softmax`. Building that row from
        the FEATURES' dtype scattered a softmax output into a reduced-precision
        row: on CUDA it raised outright, and had it not, it would have quietly
        halved the precision of the one tensor every localisation estimand ranks.

        The invariant is relative, not absolute -- autocast promotes softmax to
        float32 on CUDA and does not on CPU -- so this pins TransMIL's attention
        to the same dtype another arm's takes under the identical context, which
        holds on both. CPU bfloat16 so it runs in CI, which has no GPU; the dtype
        disagreement reproduces the bug, not the device.
        """
        torch.manual_seed(0)
        m = TransMIL(16, 32, heads=4, num_landmarks=8).eval()
        reference = build_model(pooling="gated_abmil", input_dim=16, dim=32,
                                num_classes=2).eval()
        feats, mask = _bag()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            tokens, attn = m(feats, mask)
            ref_attn = reference(feats, mask)["attn"]
        assert attn.dtype == ref_attn.dtype, (attn.dtype, ref_attn.dtype)
        assert attn[1, :, 7:].abs().max().item() == 0.0
        assert attn.sum(-1).to(torch.float32).allclose(torch.ones(2, 1), atol=1e-2)
        assert torch.isfinite(tokens).all()

    def test_real_instances_need_not_be_contiguous(self):
        """Nothing in the pooling contract promises the pad is a suffix, so the
        per-bag path indexes by mask rather than slicing a prefix."""
        torch.manual_seed(0)
        m = TransMIL(16, 32, heads=4, num_landmarks=8).eval()
        feats = torch.randn(1, 12, 16)
        mask = torch.ones(1, 12, dtype=torch.bool)
        mask[0, [2, 5, 9]] = False
        feats[0, [2, 5, 9]] = 1e4
        with torch.no_grad():
            _, attn = m(feats, mask)
        assert attn[0, 0, [2, 5, 9]].abs().max().item() == 0.0
        assert attn.sum().item() == pytest.approx(1.0, abs=1e-6)


class TestPadIsolation:
    def test_junk_in_the_pad_changes_nothing(self):
        """The strongest form of the leak test: two batches identical on the real
        instances and wildly different in the pad must produce the same output.
        PPEG is why this is not free -- a 7x7 depthwise kernel reads across the
        grid, so the pad has to be zeroed before the convolution and not merely
        masked in the attention afterwards.
        """
        torch.manual_seed(0)
        m = TransMIL(16, 32, heads=4, num_landmarks=8).eval()
        a_feats, mask = _bag(dim=16, junk=1e4)
        b_feats, _ = _bag(dim=16, junk=-7.5)
        with torch.no_grad():
            ta, aa = m(a_feats, mask)
            tb, ab = m(b_feats, mask)
        assert torch.allclose(aa, ab, atol=1e-6)
        assert torch.allclose(ta, tb, atol=1e-5)

    def test_the_squaring_pad_is_masked_not_filled_with_repeats(self):
        """The reference implementation pads by repeating the bag's own first
        instances. Repeating real instances would enter them twice into an
        attention denominator every reported estimand ranks against, so here the
        pad is zeroed and masked. A repeat-pad would make a short bag's attention
        depend on its own prefix; masking makes the row over N sum to 1 alone.
        """
        torch.manual_seed(0)
        m = TransMIL(16, 32, heads=4, num_landmarks=8).eval()
        n = 10  # squares to 16, so 6 cells are pad
        with torch.no_grad():
            _, attn = m(torch.randn(1, n, 16), torch.ones(1, n, dtype=torch.bool))
        assert attn.shape[-1] == n
        assert attn.sum().item() == pytest.approx(1.0, abs=1e-6)


class TestReportedAttentionMatchesTheModel:
    def test_exact_row_agrees_with_nystrom_when_landmarks_cover_the_sequence(self):
        """With at least as many landmarks as positions, Nystrom is exact, so the
        reported row and the block's own attention must agree. This is the check
        that the row is the same quantity -- same layer, same normalisation, same
        head averaging -- rather than merely a plausible-looking distribution.
        """
        # num_landmarks == n: one position per landmark, so all three Nystrom
        # kernels ARE the true attention and the product collapses to
        # ``A @ pinv(A) @ A``. It also leaves the sequence a whole multiple of the
        # landmark count, which matters -- NystromAttention front-pads to a
        # multiple, so with a remainder the class token is no longer at index 0 of
        # its own attention matrix and this would compare against a padded row.
        #
        # The residual is not a disagreement about what is being computed: the
        # pseudo-inverse is six Newton-Schulz iterations, so ``A @ pinv(A) @ A``
        # returns A only in the limit. Raising the iteration count has to shrink
        # the gap, and that -- not a tolerance -- is what pins the two as the same
        # quantity.
        torch.manual_seed(0)
        dim, n, heads = 32, 12, 4
        x = torch.randn(1, n, dim)
        mask = torch.ones(1, n, dtype=torch.bool)

        gaps = {}
        for iters in (6, 24):
            torch.manual_seed(0)
            layer = TransMIL(16, dim, heads=heads, num_landmarks=n,
                             pinv_iterations=iters).eval().layer2
            with torch.no_grad():
                row = layer.class_token_attention(x, mask)
                _, full = layer.attn(layer.norm(x), mask=mask, return_attn=True)
            assert full.shape == (1, heads, n, n)
            gaps[iters] = (row - full[:, :, 0, :].mean(dim=1)).abs().max().item()

        assert gaps[6] < 5e-3, gaps
        assert gaps[24] < gaps[6] / 10, gaps

    def test_the_row_is_a_distribution_where_a_nystrom_row_need_not_be(self):
        """Under a landmark count far below the sequence length the approximation
        is loose enough that its own rows leave the simplex -- which is exactly
        why the contract cannot report them directly."""
        torch.manual_seed(0)
        dim, n, heads = 32, 256, 4
        layer = TransMIL(16, dim, heads=heads, num_landmarks=4).eval().layer2
        x = torch.randn(1, n, dim)
        mask = torch.ones(1, n, dtype=torch.bool)
        with torch.no_grad():
            row = layer.class_token_attention(x, mask)
        assert (row >= 0).all()
        assert row.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_masked_positions_get_no_mass_in_the_row(self):
        torch.manual_seed(0)
        layer = TransMIL(16, 32, heads=4, num_landmarks=8).eval().layer2
        x = torch.randn(1, 20, 32)
        mask = torch.ones(1, 20, dtype=torch.bool)
        mask[0, 12:] = False
        with torch.no_grad():
            row = layer.class_token_attention(x, mask)
        assert row[0, 12:].abs().max().item() == 0.0
