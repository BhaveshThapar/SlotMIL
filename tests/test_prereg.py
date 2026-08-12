"""Tests that keep the pre-registration from becoming decoration.

A pre-registration fails quietly. Nothing crashes when a hypothesis has no
falsifier, when an arm silently disappears from the config, when the hash in the
document drifts from the hash of the file it claims to describe, or when someone
adds a "default" to an undeclared parameter. It just stops meaning anything,
while continuing to look rigorous -- which is worse than not having one.

So each test here pins one way the mechanism could rot:

* the hash chain (config <-> document <-> split files) actually holds
* strict lookup really raises instead of defaulting
* every declared arm is either constructible or explicitly marked planned
* every hypothesis has a numeric way to fail
* the primary estimands carry their prior-art credit, so the paper cannot drift
  into claiming imports as inventions
* blinding is deterministic, injective, and does not leak the name it hides
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from slotmil.models.mil import build_model
from slotmil.prereg import (
    BlindKey,
    Prereg,
    PreregViolation,
    canonical_hash,
    file_sha256,
    git_state,
    load,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/prereg/isbi2027.yaml"
DOC = REPO / "PREREGISTRATION.md"


@pytest.fixture(scope="module")
def pre() -> Prereg:
    return load(CONFIG)


class TestHashChain:
    """The chain is the whole product: doc -> config -> split files."""

    def test_config_hash_is_stable_across_reload(self, pre):
        assert load(CONFIG).hash == pre.hash

    def test_hash_ignores_key_order_but_not_values(self):
        a = {"x": 1, "y": {"p": 2, "q": 3}}
        b = {"y": {"q": 3, "p": 2}, "x": 1}
        assert canonical_hash(a) == canonical_hash(b)
        assert canonical_hash({**a, "x": 2}) != canonical_hash(a)

    def test_hash_is_sixteen_hex_like_the_split_files(self, pre):
        assert len(pre.hash) == 16 and all(c in "0123456789abcdef" for c in pre.hash)

    def test_document_records_the_current_config_hash(self, pre):
        text = DOC.read_text()
        assert "PENDING_FREEZE" not in text, (
            "PREREGISTRATION.md still says PENDING_FREEZE -- run "
            "scripts/prereg_freeze.py before treating anything as confirmatory"
        )
        assert f"`{pre.hash}`" in text, (
            f"config hashes to {pre.hash} but the document records something else; "
            "the pre-registration changed after it was frozen. Amend it and log "
            "the change in AMENDMENTS.md."
        )

    def test_declared_splits_exist_and_still_hash_as_recorded(self, pre):
        report = pre.verify_splits(must_exist=False)
        bad = {k: v for k, v in report.items() if v != "ok"}
        assert not bad, f"splits not verified: {bad}"

    def test_split_byte_hash_is_recorded_independently(self, pre):
        """make_splits' own hash covers an absolute path, not the cache contents,
        so it cannot pin a file on its own. The byte hash can."""
        path = Path(pre.split("lidc", "confirmatory")["path"])
        assert file_sha256(REPO / path) != json.loads((REPO / path).read_text())["hash"]


class TestStrictLookup:
    """No silent defaults. That is the only thing making the config binding."""

    def test_undeclared_key_raises(self, pre):
        with pytest.raises(PreregViolation, match="not declared"):
            pre.get("statistics.bootstrap.reps_but_bigger")

    def test_undeclared_nested_root_raises(self, pre):
        with pytest.raises(PreregViolation, match="not declared"):
            pre.get("nonexistent.branch")

    def test_get_has_no_default_argument(self, pre):
        """A default parameter would reintroduce exactly the failure mode this
        module exists to prevent, so its absence is worth pinning."""
        with pytest.raises(TypeError):
            pre.get("nope", "fallback")

    def test_undeclared_arm_raises_and_names_the_alternatives(self, pre):
        with pytest.raises(PreregViolation, match="not pre-registered"):
            pre.arm("resnet50")

    def test_undeclared_estimand_raises(self, pre):
        with pytest.raises(PreregViolation, match="not pre-registered"):
            pre.estimand("f1")


class TestArms:
    def test_every_arm_is_implemented_or_planned(self, pre):
        for a in pre.arms():
            assert a["status"] in ("implemented", "planned"), a

    def test_implemented_arms_are_constructible(self, pre):
        """An arm may be unbuilt -- declaring it early is the point -- but an arm
        marked implemented must actually build, or the config is lying."""
        for a in pre.arms(status="implemented"):
            pooling = a["spec"].partition(":")[0]
            torch.manual_seed(0)
            model = build_model(
                pooling=pooling, input_dim=64, dim=32, num_classes=2,
                num_slots=4, readout="gated",
            ).eval()
            out = model(torch.randn(2, 15, 64), torch.ones(2, 15, dtype=torch.bool))
            assert out["logits"].shape == (2, 2), a["name"]

    def test_planned_arms_are_not_yet_constructible(self, pre):
        """If a planned arm starts building, the config is stale -- promote it,
        so the sweep does not quietly skip an arm it could have run."""
        for a in pre.arms(status="planned"):
            pooling = a["spec"].partition(":")[0]
            with pytest.raises((ValueError, KeyError, TypeError)):
                build_model(pooling=pooling, input_dim=64, dim=32, num_classes=2)

    def test_arm_names_are_unique(self, pre):
        names = [a["name"] for a in pre.arms()]
        assert len(names) == len(set(names))

    def test_overrides_are_numeric(self, pre):
        """parse_arm (train_cached.py:34) coerces every override to float, so a
        bare flag would blow up only once the sweep is already running."""
        for a in pre.arms():
            _, _, rest = a["spec"].partition(":")
            for kv in filter(None, rest.split(",")):
                _, _, v = kv.partition("=")
                float(v)  # raises for a bare flag

    def test_reference_arms_are_never_blinded(self, pre):
        """The estimands need to know which arm is the content-free reference."""
        assert pre.arm("centre_gaussian")["blind"] is False


class TestHypothesesAndEstimands:
    def test_every_hypothesis_has_a_falsifier(self, pre):
        """A hypothesis with no way to fail is ceremony, not pre-registration."""
        for h in pre.get("hypotheses"):
            assert h.get("falsifier"), h["id"]
            assert h.get("statement"), h["id"]

    def test_hypothesis_ids_are_unique_and_sequential(self, pre):
        ids = [h["id"] for h in pre.get("hypotheses")]
        assert ids == [f"H{i}" for i in range(1, len(ids) + 1)]

    def test_the_power_gate_exists(self, pre):
        """H7 blocks the constructive half of the paper. If it is ever dropped,
        the protocol would be recommended without evidence that it has power."""
        assert pre.hypothesis("H7")["role"] == "power_gate"

    def test_primary_estimands_credit_their_prior_art(self, pre):
        """Both primaries are imports -- Kummerer 2015 and shuffled AUC. Pinning
        the credit here stops the write-up drifting into claiming invention."""
        for e in pre.get("estimands.primary"):
            assert e.get("prior_art"), e["name"]

    def test_decomposition_is_a_nested_chain_not_a_sum(self, pre):
        """AUC is not additive; the additive form invites a correct objection."""
        d = pre.get("estimands.decomposition")
        assert d["kind"] == "nested_reference_chain"
        assert d["chain"][0] == "chance" and d["chain"][-1] == "full"

    def test_floor_is_sampled_enough_to_be_a_distribution(self, pre):
        n = next(r for r in pre.get("reference_baselines")
                 if r["name"] == "untrained_fleet")["n_inits"]
        assert n >= 30, "two inits spanning 0.64-0.79 cannot support a stated floor"

    def test_attention_dtype_is_float32(self, pre):
        """fp16 caching cost seed 2 0.034 AUC on peaked attention."""
        assert pre.get("protocol.dtype") == "float32"

    def test_bootstrap_clusters_by_patient(self, pre):
        """A LIDC patient can contribute more than one series, so a bag-level
        bootstrap would understate every interval in the paper."""
        assert pre.get("statistics.bootstrap.cluster_by") == "patient"


class TestBlinding:
    def test_codes_are_deterministic_under_a_fixed_salt(self):
        a = BlindKey.create(["mean", "slot"], salt="deadbeef")
        b = BlindKey.create(["mean", "slot"], salt="deadbeef")
        assert a.codes == b.codes

    def test_different_salts_give_different_codes(self):
        a = BlindKey.create(["mean"], salt="one")
        b = BlindKey.create(["mean"], salt="two")
        assert a.code("mean") != b.code("mean")

    def test_codes_are_injective(self, pre):
        names = [a["name"] for a in pre.arms(blind=True)]
        key = BlindKey.create(names, salt="fixed",
                              nibbles=pre.get("blinding.code_nibbles"))
        assert len(set(key.codes.values())) == len(names)

    def test_code_does_not_leak_the_name(self):
        key = BlindKey.create(["transmil", "clam_sb"], salt="fixed")
        for name, code in key.codes.items():
            assert name.lower() not in code.lower()

    def test_inverse_round_trips(self):
        key = BlindKey.create(["mean", "abmil"], salt="fixed")
        inv = key.inverse()
        assert inv[key.code("abmil")] == "abmil"

    def test_unknown_arm_raises(self):
        key = BlindKey.create(["mean"], salt="fixed")
        with pytest.raises(PreregViolation, match="not in the blinding key"):
            key.code("slot")

    def test_key_survives_a_save_load_round_trip(self, tmp_path):
        key = BlindKey.create(["mean", "slot"], salt="fixed")
        p = tmp_path / "key.json"
        key.save(p)
        assert BlindKey.load(p).codes == key.codes


class TestStamping:
    def test_stamp_injects_provenance_at_top_level(self, pre):
        out = pre.stamp({"result": 1.0})
        assert out["result"] == 1.0
        assert out["prereg"]["prereg_hash"] == pre.hash
        assert out["prereg"]["prereg_version"] == pre.version
        assert set(out["prereg"]) == {
            "prereg_version", "prereg_hash", "prereg_config",
            "git_commit", "git_dirty",
        }

    def test_stamp_carries_no_timestamp(self, pre):
        """Outputs are compared byte-for-byte across re-runs to show the stamp
        perturbs nothing. A clock reading would defeat that."""
        assert json.dumps(pre.stamp({})) == json.dumps(pre.stamp({}))

    def test_git_state_never_raises(self):
        s = git_state()
        assert set(s) == {"commit", "dirty"}

    def test_git_state_on_a_non_repo_is_none_not_an_explosion(self, tmp_path):
        assert git_state(tmp_path)["commit"] is None
