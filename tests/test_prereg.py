"""Tests that keep the pre-registration from becoming decoration.

A pre-registration fails quietly. Nothing crashes when a hypothesis has no
falsifier, when an arm silently disappears from the config, when the hash in the
document drifts from the hash of the file it claims to describe, or when someone
adds a "default" to an undeclared parameter. It just stops meaning anything,
while continuing to look rigorous -- which is worse than not having one.

So each test here pins one way the mechanism could rot:

* the hash chain (config <-> document <-> split files) actually holds
* the *lineage* of that chain is readable, so a result stamped with an ancestor
  of the frozen hash is reported as an ancestor rather than silently demoted
* strict lookup really raises instead of defaulting
* every declared arm is either constructible or explicitly marked planned
* every hypothesis has a numeric way to fail
* the primary estimands carry their prior-art credit, so the paper cannot drift
  into claiming imports as inventions
* blinding is deterministic, injective, and does not leak the name it hides
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

from slotmil.models.mil import build_model
from slotmil.prereg import (
    BlindKey,
    Prereg,
    PreregViolation,
    amendment_chain,
    canonical_hash,
    classify_hash,
    file_sha256,
    git_state,
    load,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/prereg/isbi2027.yaml"
DOC = REPO / "PREREGISTRATION.md"
AMENDMENTS = REPO / "AMENDMENTS.md"
RUNS = REPO / "runs"

# The two ancestral hashes actually stamped into result files on disk. Neither
# can drift: the log is append-only, so the origin is fixed forever, and the
# floor hash is baked into runs/untrained_floor*.json, which are results and are
# never rewritten.
ORIGIN_HASH = "2b580fa93894d86f"
FLOOR_HASH = "20bdd93b781d950d"


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


ENTRY = """
## {date} — {title}

- **Kind:** {kind}
- **Config hash before → after:** `{before}` → `{after}`
- **What changed:** irrelevant to the chain
- **Results already seen?** no
"""

UNBLINDING = """
## {date} — unblinding

- **Config hash at unblinding:** `{hash}`
- **Git commit:** `0123456789abcdef0123456789abcdef01234567`
- **Results in scope:** (unspecified)
- **Reason:** writing Table 1
- **Arms revealed:** 1 (ARM-ABCDEF)
"""


def write_log(tmp_path, *entries) -> Path:
    p = tmp_path / "AMENDMENTS.md"
    p.write_text("# Amendments\n\nAppend-only.\n" + "".join(entries))
    return p


def amend(date, before, after, title="t", kind="amendment") -> str:
    return ENTRY.format(date=date, title=title, kind=kind,
                        before=before, after=after)


class TestAmendmentLineage:
    """Ancestry is not a match, and it is not nothing either.

    PREREGISTRATION.md:189-191 states one rule: a result whose hash is not the
    frozen hash is exploratory. Applied literally that demotes every result the
    moment anything is amended -- today it demotes runs/untrained_floor.json,
    the project's only confirmatory results, over two amendments that touched
    arm declarations and a train-only max_slices and nothing the untrained floor
    reads.

    The fix is a chain reader, and a chain reader is the kind of instrument that
    is trivial to write uselessly: return "ancestral" for everything and no
    result is ever demoted again. So these come in pairs. Every test that the
    mechanism *recognises* an ancestor is matched by one that it *refuses* --
    a fabricated hash, a gap, a half-written entry, a chain that does not reach
    the config it claims to describe. A lineage that accepts a broken chain is
    worse than none, because it launders provenance instead of flagging it.
    """

    # ------------------------------------------------------- reading the chain
    def test_the_recorded_chain_parses_and_is_contiguous(self, pre):
        chain = amendment_chain(AMENDMENTS)
        assert chain, "AMENDMENTS.md records no hash transitions"
        for prev, nxt in zip(chain, chain[1:]):
            assert prev.after == nxt.before
        assert chain[-1].after == pre.hash, (
            "the chain does not end at the frozen config hash -- the config was "
            "changed without an entry recording it"
        )

    def test_a_gap_in_the_chain_raises(self, tmp_path):
        """The null case for the parser: a config state no entry accounts for.

        Papering over it would report the orphaned hash as ancestral on the
        strength of a record with a hole in it."""
        log = write_log(
            tmp_path,
            amend("2026-01-01", "a" * 16, "b" * 16),
            amend("2026-01-02", "c" * 16, "d" * 16),
        )
        with pytest.raises(PreregViolation, match="gap in the amendment chain"):
            amendment_chain(log)

    def test_the_ascii_arrow_spelling_also_parses(self, tmp_path):
        """The log is hand-edited, so `->` typed for the em arrow must not
        silently drop an entry out of the chain."""
        log = write_log(tmp_path, amend("2026-01-01", "a" * 16, "b" * 16)
                        .replace("→", "->"))
        assert amendment_chain(log)[0].after == "b" * 16

    def test_a_half_recorded_amendment_raises_rather_than_guessing(self, tmp_path):
        """ENGINEERING.md's procedure drafts the entry as `before -> PENDING` before
        the new hash exists. In that window provenance genuinely is undetermined
        and the reader must say so, not skip the entry and read the chain as
        contiguous."""
        log = write_log(tmp_path, amend("2026-01-01", "a" * 16, "PENDING"))
        with pytest.raises(PreregViolation, match="not 16 hex digits"):
            amendment_chain(log)

    def test_a_dated_entry_with_no_hash_line_raises(self, tmp_path):
        log = write_log(
            tmp_path, "\n## 2026-01-01 — prose only\n\n- **Kind:** amendment\n")
        with pytest.raises(PreregViolation, match="no config hash transition"):
            amendment_chain(log)

    def test_the_template_block_is_not_read_as_an_entry(self):
        """AMENDMENTS.md documents its own format in a fenced block, and to a
        regex that template is a perfectly good entry. If it were read as one it
        would sit at the head of the chain with hashes `xxxx` -> `yyyy`."""
        chain = amendment_chain(AMENDMENTS)
        assert all(a.date != "YYYY-MM-DD" for a in chain)
        assert chain[0].before == ORIGIN_HASH

    def test_an_unblinding_entry_is_a_marker_not_a_link(self, tmp_path):
        """prereg_unblind.py appends `Config hash **at** unblinding` -- a record
        of when someone looked, not a transition. It must neither become a link
        nor break contiguity across itself."""
        log = write_log(
            tmp_path,
            amend("2026-01-01", "a" * 16, "b" * 16),
            UNBLINDING.format(date="2026-01-02", hash="b" * 16),
            amend("2026-01-03", "b" * 16, "c" * 16),
        )
        chain = amendment_chain(log)
        assert [a.after for a in chain] == ["b" * 16, "c" * 16]

    # --------------------------------------------------------- classification
    def test_the_frozen_hash_classifies_as_current(self, pre):
        lin = classify_hash(pre.hash, pre.hash, amendment_chain(AMENDMENTS))
        assert lin.status == "current" and lin.label == "current"
        assert lin.superseded_by == ()

    def test_a_fabricated_hash_classifies_as_unknown(self, pre):
        """The null half of the pair. A hash on no recorded config state must
        not be guessed to be merely old -- that guess is how a result produced
        under a plan nobody wrote down would pass as an ancestor."""
        lin = classify_hash("deadbeefdeadbeef", pre.hash, amendment_chain(AMENDMENTS))
        assert lin.status == "unknown" and lin.label == "UNKNOWN"

    def test_the_predecessor_hash_is_superseded_and_names_the_amendment(self, pre):
        """The list grows by one for every amendment that actually changes the
        config -- record-only deviations (before == after) are in the chain but
        supersede nothing, which is why not every entry appears here. The label
        names the *first* amendment past the hash, so it is stable as the list
        lengthens.

        Asserted as properties rather than as a literal list: an earlier form
        pinned the exact dates, which made every future amendment fail a test
        about lineage bookkeeping rather than about anything that had broken.
        The invariants that matter are that the entry is superseded, that every
        superseding amendment really changed the config, that they are the ones
        at or after FLOOR_HASH's position, and that the label names the first.
        """
        chain = amendment_chain(AMENDMENTS)
        lin = classify_hash(FLOOR_HASH, pre.hash, chain)
        assert lin.status == "superseded"
        assert lin.superseded_by, "a superseded hash must name what superseded it"
        assert all(a.changed_the_config for a in lin.superseded_by)
        assert lin.label == f"superseded-by-{lin.superseded_by[0].date}"

        # The superseding set is exactly the config-changing amendments from
        # FLOOR_HASH's position onward -- not a prefix, not the whole chain.
        idx = next(i for i, a in enumerate(chain) if a.before == FLOOR_HASH)
        assert [a.date for a in lin.superseded_by] == [
            a.date for a in chain[idx:] if a.changed_the_config
        ]

    def test_the_origin_hash_is_ancestral_not_unknown(self, pre):
        """The trap this whole function exists to avoid. ORIGIN_HASH is the
        first entry's *before*, so it appears nowhere among the `after` values;
        a reader that scanned only those would call five stamped results on disk
        UNKNOWN -- a lie about files whose provenance the log states outright."""
        chain = amendment_chain(AMENDMENTS)
        lin = classify_hash(ORIGIN_HASH, pre.hash, chain)
        assert lin.status == "superseded"
        # Every config-changing amendment supersedes the origin, by definition --
        # it is the chain's first `before`. Asserted as that property rather than
        # as a literal date list, which would fail on each future amendment for
        # bookkeeping reasons rather than because anything broke.
        assert [a.date for a in lin.superseded_by] == [
            a.date for a in chain if a.changed_the_config
        ]
        assert lin.label == f"superseded-by-{chain[0].date}"

    def test_a_record_only_entry_supersedes_nothing(self, tmp_path):
        """The 2026-08-15 lam pre-flight entry is a deviation with
        before == after: it discloses what was seen and changes no parameter.
        It stays in the chain -- removing it would read as a gap -- but naming
        it as something that superseded an earlier hash overstates the record."""
        log = write_log(
            tmp_path,
            amend("2026-01-01", "a" * 16, "b" * 16),
            amend("2026-01-02", "b" * 16, "b" * 16, kind="deviation"),
        )
        chain = amendment_chain(log)
        assert len(chain) == 2 and not chain[1].changed_the_config
        lin = classify_hash("a" * 16, "b" * 16, chain)
        assert [a.date for a in lin.superseded_by] == ["2026-01-01"]

    def test_a_chain_that_does_not_reach_the_config_raises(self, tmp_path):
        """Running prereg_freeze.py --amend without logging the entry leaves the
        doc and config agreeing while the chain describes neither. Nothing can
        be called ancestral against a chain that stops somewhere else."""
        log = write_log(tmp_path, amend("2026-01-01", "a" * 16, "b" * 16))
        chain = amendment_chain(log)
        with pytest.raises(PreregViolation, match="chain ends at"):
            classify_hash("a" * 16, "c" * 16, chain)

    # ------------------------------------------------------- results on disk
    def test_every_stamped_result_on_disk_is_placeable(self, pre):
        """The payoff. Sixteen stamped files exist and none carries the frozen
        hash; every one of them must resolve to a recorded config state."""
        chain = amendment_chain(AMENDMENTS)
        placed = {}
        for p in sorted(RUNS.rglob("*.json")):
            try:
                obj = json.loads(p.read_text())
            except ValueError:
                continue  # unreadable, so it carries no readable stamp either
            stamp = obj.get("prereg") if isinstance(obj, dict) else None
            if isinstance(stamp, dict) and "prereg_hash" in stamp:
                placed[p] = classify_hash(stamp["prereg_hash"], pre.hash, chain)
        assert placed, f"no stamped results under {RUNS}; this test proves nothing"
        unknown = {str(p): lin.label for p, lin in placed.items()
                   if lin.status == "unknown"}
        assert not unknown, f"stamps on no recorded config state: {unknown}"

    def test_the_confirmatory_floor_is_ancestral_not_orphaned(self, pre):
        """runs/untrained_floor.json carries H3 and ran on the confirmatory
        split. The mechanical rule demotes it to exploratory over two amendments
        it does not read; the lineage says which ancestor it is instead."""
        stamp = json.loads((RUNS / "untrained_floor.json").read_text())["prereg"]
        assert stamp["prereg_hash"] != pre.hash
        lin = classify_hash(stamp["prereg_hash"], pre.hash, amendment_chain(AMENDMENTS))
        assert lin.status == "superseded", lin


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

    def test_normal_guidance_names_an_implemented_base_arm(self, pre):
        """H6 is stated against "its base arm" and the frozen text never named
        one. Leaving it nameless would make the single most consequential choice
        in the hypothesis a decision taken at analysis time."""
        base = pre.arm("normal_guidance")["base_arm"]
        implemented = {a["name"] for a in pre.arms(status="implemented")}
        assert base in implemented, f"{base!r} not among {sorted(implemented)}"

    def test_declared_constants_match_the_code(self, pre):
        """A declared value that has drifted from the value actually used is
        worse than an undeclared one: it reads as a commitment and is not."""
        from slotmil.losses import NG_PATCHES_PER_SLICE, NG_VAR_FLOOR_SLICES2
        from slotmil.models.baselines import CENTRE_GAUSSIAN_SIGMA_Z

        assert pre.arm("centre_gaussian")["sigma_z"] == CENTRE_GAUSSIAN_SIGMA_Z
        assert pre.arm("normal_guidance")["var_floor_slices2"] == NG_VAR_FLOOR_SLICES2
        assert NG_PATCHES_PER_SLICE == pre.get("datasets.lidc.patches_per_slice")

    def test_normal_guidance_spec_carries_a_positive_lam(self, pre):
        """Without lam the arm trains as its own base arm and is reported as NG.
        scripts/train_cached.py refuses to run it that way; this pins the spec
        that refusal is stated against."""
        from scripts.train_cached import parse_arm

        _, pooling, overrides = parse_arm(pre.arm("normal_guidance")["spec"])
        assert pooling == "normal_guidance"
        assert overrides.get("lam", 0.0) > 0

    def test_the_auxiliary_stream_weights_are_pinned_in_code(self, pre):
        """clam_sb without its clustering term is gated_abmil and dsmil without
        its max term is a single-stream pooling. Both would train and be reported
        under a published name, so the printed splits are asserted, not trusted."""
        from slotmil.losses import CLAM_BAG_WEIGHT, CLAM_TOPK_B, DSMIL_BAG_WEIGHT
        from slotmil.models.baselines import DSMIL_DROPOUT_V, DSMIL_Q_HIDDEN

        assert pre.arm("clam_sb")["bag_weight"] == CLAM_BAG_WEIGHT
        assert pre.arm("clam_sb")["topk_b"] == CLAM_TOPK_B
        assert pre.arm("dsmil")["bag_weight"] == DSMIL_BAG_WEIGHT
        assert pre.arm("dsmil")["q_hidden"] == DSMIL_Q_HIDDEN
        assert pre.arm("dsmil")["dropout_v"] == DSMIL_DROPOUT_V

    def test_h5_declares_its_unit(self, pre):
        """The freeze fixed H5's threshold and left its unit free, so an arm's
        number could be a per-seed value or an aggregate -- and on the discovery
        dumps the two disagree across the threshold (max 0.3121, mean 0.162).
        Ruled by amendment before f32_seed1 was collected; pinned here so it
        cannot be re-read once confirmatory numbers exist."""
        h5 = pre.hypothesis("H5")
        assert h5["unit"] == "mean_over_seeds"
        assert h5["report_per_seed"] is True

    def test_h10_has_three_outcomes_and_only_one_falsifies(self, pre):
        """Drafted with two outcomes, which scored an oracle win as a failure of
        a hypothesis about oracle wins. Only trained-wins withdraws the claim."""
        h10 = pre.hypothesis("H10")
        assert set(h10["outcomes"]) == {
            "oracle_wins", "indistinguishable", "trained_wins"}
        assert "trained-wins" in h10["falsifier"]
        assert h10["member"] == "separable", "the paired member is pinned, not selected"

    def test_h10s_three_outcomes_are_scored_by_code_not_by_eye(self, pre):
        """The declared outcomes must have a scorer, or the verdict lives only
        as an interval someone reads -- which is how H10 got drafted with two
        outcomes and how H5 kept an undeclared unit."""
        from slotmil.eval.estimands import h10_outcome

        assert h10_outcome({"lo": 0.04, "hi": 0.09}) == "oracle_wins"
        assert h10_outcome({"lo": -0.03, "hi": 0.002}) == "indistinguishable"
        assert h10_outcome({"lo": -0.09, "hi": -0.04}) == "trained_wins"
        assert h10_outcome({"lo": None, "hi": None}) is None
        assert set(pre.hypothesis("H10")["outcomes"]) == {
            h10_outcome({"lo": 0.04, "hi": 0.09}),
            h10_outcome({"lo": -0.03, "hi": 0.002}),
            h10_outcome({"lo": -0.09, "hi": -0.04}),
        }

    def test_h2_scope_excludes_the_content_free_arm(self, pre):
        """centre_gaussian ties the centre_prior reference by construction, and a
        tie is not "exceeds". Ruled in the config so it cannot become a post-hoc
        call once the numbers are in."""
        h2 = pre.hypothesis("H2")
        assert "centre_gaussian" in h2["excluded_arms"]


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


REPO = Path(__file__).resolve().parent.parent


def _argparse_choices(script: str, flag: str) -> set[str]:
    """Pull an argparse `choices=[...]` literal straight out of a script's source.

    Read as text rather than imported: these scripts pull in torch and h5py at module
    scope, and the thing under test is the literal list, not anything importing it
    would reveal.
    """
    src = (REPO / script).read_text()
    m = re.search(rf'add_argument\(\s*"{re.escape(flag)}".*?choices=\[(.*?)\]',
                  src, re.S)
    assert m, f"no choices= for {flag} in {script}"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


class TestArmSetScoping:
    """The 2026-08-15 scoping amendment, pinned so it cannot rot back into prose.

    `mean` and `centre_gaussian` have hypothesis outcomes fixed by construction --
    uniform attention ties every axis at 0.5, and slice-constant attention forces
    within-slice to 0.5 -- so counting them is counting arithmetic as evidence.
    """

    SCOPED = ("H1", "H2", "H4", "H5", "H6", "H10")

    def test_every_implemented_arm_declares_a_scoring_class(self, pre):
        for a in pre.arms(status="implemented"):
            assert a.get("scoring_class") in pre.SCORING_CLASSES, a["name"]

    @pytest.mark.parametrize("hid", SCOPED)
    def test_scoped_hypotheses_exclude_the_construction_fixed_arms(self, pre, hid):
        s = pre.arm_set(hid)
        assert "mean" not in s
        assert "centre_gaussian" not in s
        assert len(s) == 7

    def test_an_unscoped_hypothesis_still_sees_every_arm(self, pre):
        """The pre-freeze default must survive, or this method would silently
        rescope hypotheses nobody amended."""
        assert set(pre.arm_set("H3")) == {a["name"] for a in pre.arms(status="implemented")}

    def test_an_undeclared_arm_set_raises_rather_than_defaulting(self, pre):
        pre.raw["hypotheses"].append({"id": "HX", "arm_set": "not_a_class"})
        try:
            with pytest.raises(PreregViolation):
                pre.arm_set("HX")
        finally:
            pre.raw["hypotheses"].pop()

    def test_the_exclusion_does_not_move_H1s_falsification_bar(self, pre):
        """The property that makes the ruling admissible at all.

        H1 falsifies on a majority. Over 9 arms that is 5, and centre_gaussian is a
        guaranteed failure while mean is a guaranteed pass -- so 4 of the other 7
        must fail. Over the 7-arm set a majority is 4. The same four. If this ever
        stops holding, the exclusion has become a choice that changes the verdict.
        """
        n_all = len(pre.arms(status="implemented"))
        n_set = len(pre.arm_set("H1"))
        assert n_all - n_set == 2
        assert (n_all // 2 + 1) - 1 == n_set // 2 + 1

    def test_H2s_prose_ruling_and_the_mechanism_agree(self, pre):
        """Both forms are kept side by side; drifting apart is the failure."""
        h2 = pre.hypothesis("H2")
        for name in h2["excluded_arms"]:
            assert name not in pre.arm_set("H2")


class TestHypothesesAreComputable:
    """Each of these was a hole through which a number could have been chosen late."""

    def test_H10_names_the_arm_it_is_scored_on(self, pre):
        arm = pre.hypothesis("H10")["arm"]
        assert arm in {a["spec"] for a in pre.arms(status="implemented")}

    def test_H1_declares_positive_controls_that_must_exceed_the_threshold(self, pre):
        pc = pre.hypothesis("H1")["positive_controls"]
        assert set(pc["members"]) == {"masks:axial", "masks:separable", "centre_prior"}

    def test_H1_declares_the_tie_floor_check(self, pre):
        tf = pre.hypothesis("H1")["tie_floor_check"]
        assert tf["arm"] == "mean" and tf["expected_gap"] == 0.0

    def test_patient_specific_skill_is_operationalised(self, pre):
        cp = pre.hypothesis("H4")["cross_patient"]
        assert cp["n_derangements"] >= 1
        assert cp["rng_seed"] == 0
        assert cp["axis"] == "flat"
        assert pre.hypothesis("H6")["cross_patient"] == "same_as_H4"

    def test_the_holm_family_is_named_rather_than_left_to_analysis_time(self, pre):
        fam = pre.get("statistics.holm_family")
        assert fam["references"] and fam["alpha"] == pre.get("statistics.alpha")

    def test_verdicts_are_scored_on_point_estimates(self, pre):
        """Falsifiers are written as point-estimate comparisons. Rewriting them as
        interval rules after the discovery numbers are known would move every one."""
        assert pre.get("statistics.verdict_basis") == "point_estimate"

    def test_exactly_one_condition_carries_the_hypothesis_family(self, pre):
        carriers = [c["name"] for c in pre.get("conditions")
                    if c.get("carries_hypothesis_family")]
        assert carriers == [pre.get("hypothesis_family_condition")]

    def test_every_condition_names_a_declared_split(self, pre):
        for c in pre.get("conditions"):
            pre.split(c["dataset"], c["split"])


class TestH7IsComputable:
    """The power gate. If it fails, the constructive half of the paper is withdrawn,
    so every term in it has to mean one thing."""

    def test_the_probe_carries_its_own_denominator(self, pre):
        """fitted_template is fit to the SCORER's own validation attention, so every
        arm already has its own. Borrowing one would make the verdict depend on which
        arm was borrowed from -- across the eight discovery denominators the probe's
        skill spans 0.4720-0.7224, straddling its own 0.50 threshold."""
        assert pre.hypothesis("H7")["probe_denominator"] == "own_fitted_inplane_template"

    def test_the_content_free_set_is_enumerated_and_label_free(self, pre):
        """Every member must be a declared reference_baseline, so the set cannot grow
        a member that post-dates the freeze."""
        declared = {r["name"] for r in pre.get("reference_baselines")}
        for name in pre.hypothesis("H7")["content_free_set"]:
            assert name in declared, name

    def test_the_mask_fitted_family_is_not_in_the_gate(self, pre):
        """Content-free but not label-free: fit to validation lesion masks. Requiring
        it to score below 0.05 while requiring a supervised probe above 0.50 is not a
        strict test but an inconsistent one. Excluded -- and reported beside the gate."""
        h7 = pre.hypothesis("H7")
        assert not any(m.startswith("masks:") for m in h7["content_free_set"])
        assert any(m.startswith("masks:") for m in h7["reported_beside_the_gate"])

    def test_centre_prior_is_excluded_from_the_gate_but_still_reported(self, pre):
        h7 = pre.hypothesis("H7")
        assert "centre_prior" not in h7["content_free_set"]
        assert "centre_prior" in h7["reported_beside_the_gate"]

    def test_the_probe_scores_every_patch_even_though_it_fits_on_a_subsample(self, pre):
        """Subsampling at fit time touches no estimand. Subsampling at score time
        computes the numerator over a different patch population than the
        denominator, which is a different quantity wearing the same name."""
        proto = pre.hypothesis("H7")["probe_protocol"]
        assert proto["score_coverage"] == "every_patch_of_every_bag"
        assert proto["fit_split"] == "train" and proto["score_split"] == "test"


class TestArmAllowLists:
    """ENGINEERING.md: these fail only at submission time, hours into a job.

    `abmil` and `gated_abmil` were missing until 2026-08-15, which made H1, H2 and
    H6 uncomputable -- H6 most directly, since gated_abmil is normal_guidance's
    declared base arm.
    """

    @pytest.mark.parametrize("script", ["scripts/null_collect_attn.py",
                                        "scripts/untrained_floor.py"])
    def test_every_implemented_arm_is_collectible(self, pre, script):
        choices = _argparse_choices(script, "--pooling")
        for a in pre.arms(status="implemented"):
            pooling = a["spec"].partition(":")[0]
            assert pooling in choices, f"{a['name']} -> {pooling} missing from {script}"

    def test_the_reeval_arm_table_covers_every_implemented_arm(self, pre):
        src = (REPO / "scripts/reeval_all_alignment.py").read_text()
        block = re.search(r"ARMS = \[(.*?)\]", src, re.S).group(1)
        poolings = {m[1] for m in re.findall(r'\("([^"]+)",\s*"([^"]+)"\)', block)}
        for a in pre.arms(status="implemented"):
            assert a["spec"].partition(":")[0] in poolings, a["name"]
