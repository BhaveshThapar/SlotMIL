"""The two auxiliary-stream arms: clam_sb and dsmil.

Both differ from a plain attention pooling by a second supervised stream, and
both route that stream to the loss through ``out["instance_logits"]``. Three
things can break silently here and each has a test below.

*The stream can vanish.* ``clam_sb`` without its clustering term is exactly
``gated_abmil`` and ``dsmil`` without its max term is a single-stream non-local
pooling. Either would train, write a valid ``result.json`` and be reported under
a published method's name -- the failure ``lam`` had before it was guarded.

*The pad convention can drift.* Both arms take a max or a top-k over the
instance axis, so a padded column that is merely small rather than exactly zero
does not just steal attention mass, it can win the argmax outright: on a batch
of mixed-depth LIDC bags most of the tensor is padding.

*The instance stream can disagree with the forward pass.* ``instance_logits`` is
recomputed from ``feats`` rather than cached during ``forward``, deliberately --
a cache would make the value depend on call order. That only holds if the two
paths compute the same thing, which ``test_dsmil_instance_logits_match`` pins.
"""

from __future__ import annotations

import pytest
import torch

from slotmil.losses import (
    CLAM_TOPK_B,
    SlotMILLoss,
    clam_instance_loss,
    dsmil_max_loss,
)
from slotmil.models.baselines import CLAMSB, DSMIL, GatedABMIL, InstanceScoringPool
from slotmil.models.mil import build_model

ARMS = ("clam_sb", "dsmil")


def _batch(b=2, n=12, dim=16, cut=7, seed=0):
    """A batch whose second bag is padded from ``cut``, with junk in the pad."""
    torch.manual_seed(seed)
    feats = torch.randn(b, n, dim)
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[1, cut:] = False
    feats[1, cut:] = 1e4  # a pad that would win any unmasked max or top-k
    return feats, mask, torch.tensor([1, 0])[:b]


@pytest.mark.parametrize("arm", ARMS)
class TestContract:
    def test_padded_columns_are_exactly_zero(self, arm):
        feats, mask, _ = _batch()
        m = build_model(pooling=arm, input_dim=16, dim=8, num_classes=2).eval()
        attn = m(feats, mask)["attn"]
        assert attn[1, :, 7:].abs().max().item() == 0.0
        assert torch.isfinite(attn).all()
        assert attn[1, :, :7].sum(-1).allclose(torch.ones(attn.shape[1]), atol=1e-6)

    def test_attention_normalises_over_instances(self, arm):
        """Stated in the pre-registration for both arms, and load-bearing:
        localization.instance_auc(slot=None) takes a max over the K axis and is
        invalid for slot-normalised attention. Rows summing to 1 is what makes
        it valid here."""
        feats, mask, _ = _batch()
        m = build_model(pooling=arm, input_dim=16, dim=8, num_classes=2).eval()
        attn = m(feats, mask)["attn"]
        assert attn.sum(-1).allclose(torch.ones_like(attn.sum(-1)), atol=1e-6)

    def test_instance_logits_are_surfaced(self, arm):
        feats, mask, _ = _batch()
        m = build_model(pooling=arm, input_dim=16, dim=8, num_classes=2).eval()
        assert isinstance(m.pooling, InstanceScoringPool)
        assert m(feats, mask)["instance_logits"].shape[:2] == feats.shape[:2]

    def test_the_short_bag_of_the_conformance_suite_survives(self, arm):
        """15 instances, the width tests/test_prereg.py builds every arm at."""
        m = build_model(pooling=arm, input_dim=64, dim=32, num_classes=2).eval()
        out = m(torch.randn(2, 15, 64), torch.ones(2, 15, dtype=torch.bool))
        assert out["logits"].shape == (2, 2)


class TestCLAM:
    def test_attention_path_is_its_base_arm(self):
        """clam_sb is gated_abmil plus a clustering branch, so at the same seed
        the attention must be bit-identical -- that identity is what makes any
        difference between the two arms attributable to the objective."""
        feats, mask, _ = _batch()
        torch.manual_seed(0)
        a = CLAMSB(16, 8).eval()
        torch.manual_seed(0)
        b = GatedABMIL(16, 8).eval()
        # Same init for the shared path; CLAMSB's extra head is drawn after it.
        b.load_state_dict({k: v for k, v in a.state_dict().items()
                           if not k.startswith("inst.")})
        assert torch.equal(a(feats, mask)[1], b(feats, mask)[1])

    def test_k_falls_to_half_the_bag_rather_than_overlapping(self):
        """With n_valid=7 and B=8 the top and bottom sets would otherwise share
        instances and train the classifier on contradictory labels for one
        instance. k drops to 3 for that bag instead."""
        feats, mask, y = _batch()
        m = build_model(pooling="clam_sb", input_dim=16, dim=8, num_classes=2).eval()
        out = m(feats, mask)
        loss = clam_instance_loss(out["instance_logits"], out["attn"], y, mask,
                                  k=CLAM_TOPK_B)
        assert torch.isfinite(loss)

    def test_a_bag_too_short_to_split_contributes_nothing(self):
        feats = torch.randn(1, 1, 16)
        mask = torch.ones(1, 1, dtype=torch.bool)
        m = build_model(pooling="clam_sb", input_dim=16, dim=8, num_classes=2).eval()
        out = m(feats, mask)
        loss = clam_instance_loss(out["instance_logits"], out["attn"],
                                  torch.tensor([1]), mask)
        assert loss.item() == 0.0

    def test_multilabel_targets_are_refused(self):
        """CLAM's in-the-class clustering selects one classifier by bag class;
        under multilabel there is no such class, and guessing one would train a
        classifier on labels nobody declared."""
        feats, mask, _ = _batch()
        m = build_model(pooling="clam_sb", input_dim=16, dim=8, num_classes=2).eval()
        out = m(feats, mask)
        with pytest.raises(ValueError, match="single-label"):
            clam_instance_loss(out["instance_logits"], out["attn"],
                               torch.zeros(2, 2), mask)


class TestDSMIL:
    def test_one_token_per_class(self):
        m = build_model(pooling="dsmil", input_dim=16, dim=8, num_classes=3).eval()
        out = m(torch.randn(2, 12, 16), torch.ones(2, 12, dtype=torch.bool))
        assert out["tokens"].shape[1] == 3 and out["attn"].shape[1] == 3

    def test_instance_logits_match_the_forward_pass(self):
        """The hook recomputes rather than caches. If the two paths ever diverge
        the max stream would be supervising scores the critical-instance
        selection never saw."""
        feats, mask, _ = _batch()
        pool = DSMIL(16, 8).eval()
        direct = pool.inst(pool.proj(feats))
        assert torch.equal(pool.instance_logits(feats, mask), direct)

    def test_padding_cannot_win_the_critical_instance(self):
        """The pad region is 1e4 here. Unmasked it takes the argmax on every
        bag, and the whole bag stream then attends against a padded column."""
        feats, mask, _ = _batch()
        pool = DSMIL(16, 8).eval()
        _, attn = pool(feats, mask)
        assert attn[1, :, 7:].abs().max().item() == 0.0
        assert attn[1].argmax(dim=-1).max().item() < 7

    def test_max_stream_ignores_padded_instances(self):
        feats, mask, y = _batch()
        pool = DSMIL(16, 8).eval()
        # The padded bag alone: bag 0 is valid to its full width, so trimming it
        # would drop real instances and change the max for a legitimate reason.
        s = pool.instance_logits(feats, mask)[1:]
        padded = dsmil_max_loss(s, y[1:], mask[1:])
        trimmed = dsmil_max_loss(s[:, :7], y[1:], mask[1:, :7])
        assert padded.item() == pytest.approx(trimmed.item(), abs=1e-6)


class TestLossWiring:
    @pytest.mark.parametrize("arm,key,kw", [
        ("clam_sb", "clam_inst", {"w_clam_inst": 0.3}),
        ("dsmil", "dsmil_max", {"w_dsmil_max": 0.5}),
    ])
    def test_the_stream_is_logged_and_moves_the_loss(self, arm, key, kw):
        """Every comps key is auto-logged as loss_<key> in history.json, so a
        term that is wired but silent is visible in the run rather than only in
        the source."""
        feats, mask, y = _batch()
        m = build_model(pooling=arm, input_dim=16, dim=8, num_classes=2)
        out = m(feats, mask)
        base, _ = SlotMILLoss()(out, y, pad_mask=mask)
        total, comps = SlotMILLoss(**kw)(out, y, pad_mask=mask)
        assert key in comps and comps[key].item() != 0.0
        assert total.item() != pytest.approx(base.item())

    @pytest.mark.parametrize("arm,kw", [
        ("clam_sb", {"w_bag": 0.7, "w_clam_inst": 0.3}),
        ("dsmil", {"w_bag": 0.5, "w_dsmil_max": 0.5}),
    ])
    def test_gradients_reach_the_auxiliary_head(self, arm, kw):
        feats, mask, y = _batch()
        m = build_model(pooling=arm, input_dim=16, dim=8, num_classes=2)
        loss, _ = SlotMILLoss(**kw)(m(feats, mask), y, pad_mask=mask)
        loss.backward()
        g = m.pooling.inst.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
