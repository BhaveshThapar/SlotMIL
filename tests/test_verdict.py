"""Tests for the ten pre-registered falsifiers.

Every rule gets a case that must pass and a case that must fail, because a scorer
that can only return one answer is the same ceremony
``PREREGISTRATION.md`` warns about in hypotheses: "a hypothesis that cannot fail is
ceremony". Two further things are pinned here that a passing verdict table would
not reveal:

* **VOID is distinct from FAIL and from PASS.** H1's positive controls and tie
  floor, and an off-chain artefact, all void a verdict rather than deciding it. If
  VOID ever collapses into PASS, an unsupported hypothesis reads as supported; into
  FAIL, and a broken harness reads as evidence against a claim.
* **NOT_RUN is distinct from PASS.** A hypothesis with no artefact must never come
  out satisfied, which is the failure mode of every scoring script that iterates
  over whatever results it happens to find.

The arm-set resolution test is here rather than in ``tests/test_prereg.py`` because
what it guards is a verdict, not a config: ``mean`` returns uniform attention and
``centre_gaussian`` is constant within a slice, so both have outcomes fixed by
arithmetic, and counting them would be counting arithmetic as evidence.
"""

from __future__ import annotations

import json

import pytest

# scripts/ has no __init__.py and is not installed; pyproject's `pythonpath = ["."]`
# is what puts the repo root on sys.path for a bare `pytest`, and scripts/ then
# resolves as a namespace package. Same import route tests/test_lung_mask_io.py uses.
from scripts.prereg_verdict import (
    ARM_KEYED,
    Artefacts,
    _blind,
    _h9_power,
    blind_substitutions,
    score_h1,
    score_h2,
    score_h3,
    score_h4,
    score_h5,
    score_h6,
    score_h7,
    score_h8,
    score_h9,
    score_h10,
)
from slotmil import prereg
from slotmil.eval.verdict import (
    FAIL,
    NOT_RUN,
    PASS,
    REPORTED,
    VOID,
    arm_tag,
    h1_verdict,
    h2_verdict,
    h3_verdict,
    h4_verdict,
    h5_verdict,
    h6_verdict,
    h7_verdict,
    h8_verdict,
    h9_verdict,
    h10_verdict,
    majority,
)

# H1's controls, well clear of the 0.02 threshold. The discovery values are
# 0.0978 / 0.0259 / 0.0401, so these are in the right range rather than invented.
GOOD_CONTROLS = {"masks:axial": 0.0978, "masks:separable": 0.0259,
                 "centre_prior": 0.0401}


def ci(mean, lo=None, hi=None):
    return {"mean": mean, "lo": lo, "hi": hi, "n": 10, "n_clusters": 10}


class TestMajority:
    def test_odd_and_even(self):
        assert majority(7) == 4
        assert majority(9) == 5
        assert majority(4) == 3

    def test_seven_arm_and_nine_arm_bars_coincide(self):
        """The amendment's arithmetic: excluding two arms does not move H1's bar.

        Over 9 arms a majority is 5, and ``centre_gaussian`` contributes one
        guaranteed failure, so falsification needs 4 of the other 7 -- which is
        exactly a majority of the 7-arm set.
        """
        assert majority(9) - 1 == majority(7)


class TestArmTag:
    def test_built_from_the_spec_not_the_name(self):
        """``slot:div=0.5`` has name ``slot_div0.5`` and directory ``slot_div=0.5``."""
        assert arm_tag("slot:div=0.5", 3) == "slot_div=0.5_seed3"
        assert arm_tag("normal_guidance:lam=0.1", 0) == "normal_guidance_lam=0.1_seed0"
        assert arm_tag("abmil", 4) == "abmil_seed4"

    def test_every_declared_spec_round_trips_to_a_real_directory_name(self):
        """The tag must match what train_cached.py writes: spec with ':' -> '_'."""
        pre = prereg.load()
        for arm in pre.arms(status="implemented"):
            tag = arm_tag(arm["spec"], 0)
            assert tag == f"{arm['spec'].replace(':', '_')}_seed0"
            assert ":" not in tag


class TestH1:
    def test_all_arms_inside_the_threshold_passes(self):
        gaps = {f"arm{i}": 0.001 for i in range(7)}
        v = h1_verdict(gaps, 0.02, GOOD_CONTROLS, 0.0, 0.0)
        assert v["outcome"] == PASS

    def test_a_majority_over_the_threshold_fails(self):
        gaps = {f"arm{i}": (0.05 if i < 4 else 0.001) for i in range(7)}
        v = h1_verdict(gaps, 0.02, GOOD_CONTROLS, 0.0, 0.0)
        assert v["outcome"] == FAIL
        assert v["majority_needed"] == 4 and len(v["arms_over_threshold"]) == 4

    def test_a_minority_over_the_threshold_still_passes(self):
        gaps = {f"arm{i}": (0.05 if i < 3 else 0.001) for i in range(7)}
        assert h1_verdict(gaps, 0.02, GOOD_CONTROLS, 0.0, 0.0)["outcome"] == PASS

    def test_a_failed_positive_control_voids_rather_than_supports(self):
        """A control that carries axial content by construction coming in under the
        threshold means the harness cannot see axial content at all."""
        controls = dict(GOOD_CONTROLS, **{"masks:axial": 0.001})
        v = h1_verdict({f"arm{i}": 0.001 for i in range(7)}, 0.02, controls, 0.0, 0.0)
        assert v["outcome"] == VOID
        assert "masks:axial" in v["reason"]

    def test_a_moved_tie_floor_voids(self):
        v = h1_verdict({"a": 0.001}, 0.02, GOOD_CONTROLS, 0.004, 0.0)
        assert v["outcome"] == VOID and "tie floor" in v["reason"]

    def test_a_missing_tie_floor_voids(self):
        assert h1_verdict({"a": 0.001}, 0.02, GOOD_CONTROLS, None, 0.0)["outcome"] == VOID

    def test_no_arms_is_not_run_rather_than_pass(self):
        assert h1_verdict({}, 0.02, GOOD_CONTROLS, 0.0, 0.0)["outcome"] == NOT_RUN

    def test_exactly_at_the_threshold_counts_as_over(self):
        """The falsifier is ``< 0.02``, so 0.02 itself does not satisfy it."""
        gaps = {f"arm{i}": (0.02 if i < 4 else 0.0) for i in range(7)}
        assert h1_verdict(gaps, 0.02, GOOD_CONTROLS, 0.0, 0.0)["outcome"] == FAIL


class TestH2:
    def test_prior_above_every_arm_passes(self):
        assert h2_verdict(0.6026, {"a": 0.4822, "b": 0.5565})["outcome"] == PASS

    def test_any_arm_beating_the_prior_fails(self):
        v = h2_verdict(0.6026, {"a": 0.4822, "b": 0.7})
        assert v["outcome"] == FAIL and v["arms_at_or_above"] == ["b"]

    def test_a_tie_fails_because_exceeds_is_strict(self):
        v = h2_verdict(0.60, {"a": 0.60})
        assert v["outcome"] == FAIL

    def test_missing_prior_is_not_run(self):
        assert h2_verdict(None, {"a": 0.4})["outcome"] == NOT_RUN


class TestH3:
    def test_above_the_threshold_passes(self):
        assert h3_verdict(0.7666, 0.70, 30, 30)["outcome"] == PASS

    def test_at_or_below_the_threshold_fails(self):
        assert h3_verdict(0.70, 0.70, 30, 30)["outcome"] == FAIL

    def test_too_few_inits_voids_rather_than_passing(self):
        v = h3_verdict(0.90, 0.70, 12, 30)
        assert v["outcome"] == VOID and "12" in v["reason"]


class TestH4:
    def test_all_arms_under_both_bounds_passes(self):
        assert h4_verdict({f"a{i}": 0.05 for i in range(7)}, 0.15, 0.10)["outcome"] == PASS

    def test_one_arm_over_the_per_arm_bound_fails(self):
        skills = {f"a{i}": 0.05 for i in range(7)}
        skills["a0"] = 0.20
        v = h4_verdict(skills, 0.15, 0.10)
        assert v["outcome"] == FAIL and v["arms_over_threshold"] == ["a0"]

    def test_the_median_bound_fails_independently(self):
        """Every arm under 0.15 but the median at or above 0.10 still falsifies."""
        v = h4_verdict({f"a{i}": 0.14 for i in range(7)}, 0.15, 0.10)
        assert v["outcome"] == FAIL and "median" in v["reason"]

    def test_the_median_is_computed_over_arms(self):
        v = h4_verdict({"a": 0.01, "b": 0.02, "c": 0.03}, 0.15, 0.10)
        assert v["median"] == pytest.approx(0.02)

    def test_no_arm_names_leak_into_the_reason(self):
        """prereg_verdict.py blinds structured fields and cannot blind prose."""
        skills = {"slot_div0.5": 0.20, "abmil": 0.01, "dsmil": 0.01}
        v = h4_verdict(skills, 0.15, 0.10)
        for name in skills:
            assert name not in v["reason"]

    def test_no_arms_is_not_run(self):
        assert h4_verdict({}, 0.15, 0.10)["outcome"] == NOT_RUN


class TestH5:
    def test_all_means_under_the_threshold_passes(self):
        assert h5_verdict({"a": 0.1, "b": 0.2}, 0.30)["outcome"] == PASS

    def test_a_mean_over_the_threshold_fails(self):
        v = h5_verdict({"a": 0.1, "b": 0.31}, 0.30)
        assert v["outcome"] == FAIL and v["arms_over_threshold"] == ["b"]

    def test_a_seed_over_the_threshold_is_reported_without_falsifying(self):
        """``report_per_seed`` exists so the unit ruling cannot become a way of not
        publishing a seed above the bar."""
        v = h5_verdict({"a": 0.20}, 0.30, {"a": [0.05, 0.05, 0.05, 0.05, 0.80]})
        assert v["outcome"] == PASS
        assert v["seeds_over_threshold"] == {"a": [0.80]}


class TestH6:
    def test_a_small_skill_rise_passes(self):
        v = h6_verdict(0.50, 0.60, 0.09, 0.10, 0.02)
        assert v["outcome"] == PASS and v["slice_rose"] is True

    def test_a_skill_rise_at_the_threshold_fails(self):
        assert h6_verdict(0.50, 0.60, 0.09, 0.11, 0.02)["outcome"] == FAIL

    def test_a_flat_slice_column_does_not_falsify_on_its_own(self):
        """The declared falsifier names the skill only. Inventing a second one here
        would tighten a pre-registered rule after the fact."""
        v = h6_verdict(0.60, 0.50, 0.09, 0.09, 0.02)
        assert v["outcome"] == PASS and v["slice_rose"] is False

    def test_missing_skill_is_not_run(self):
        assert h6_verdict(0.5, 0.6, None, 0.1, 0.02)["outcome"] == NOT_RUN


class TestH7:
    CF = {"chance": 0.0, "roll_permutation": 0.01,
          "entropy_matched_random": 0.02, "fitted_template": 0.0}

    def test_probe_clears_and_content_free_stays_low(self):
        assert h7_verdict(0.7305, 0.50, self.CF, 0.05)["outcome"] == PASS

    def test_a_probe_below_its_threshold_fails_the_gate(self):
        assert h7_verdict(0.40, 0.50, self.CF, 0.05)["outcome"] == FAIL

    def test_a_content_free_member_over_its_threshold_fails_the_gate(self):
        cf = dict(self.CF, roll_permutation=0.09)
        v = h7_verdict(0.7305, 0.50, cf, 0.05)
        assert v["outcome"] == FAIL and v["members_over_threshold"] == ["roll_permutation"]

    def test_an_uncomputable_member_cannot_falsify(self):
        cf = dict(self.CF, entropy_matched_random=None)
        v = h7_verdict(0.7305, 0.50, cf, 0.05)
        assert v["outcome"] == PASS
        assert v["uncomputable"] == ["entropy_matched_random"]

    def test_every_member_uncomputable_voids_rather_than_clears(self):
        cf = dict.fromkeys(self.CF)
        assert h7_verdict(0.7305, 0.50, cf, 0.05)["outcome"] == VOID


class TestH8:
    def test_is_reported_not_passed_or_failed(self):
        """H8's falsifier is 'none -- reported either way', so scoring it as a pass
        or a failure would invent a verdict the pre-registration declines to make."""
        above = h8_verdict(0.786, 0.65)
        below = h8_verdict(0.52, 0.65)
        assert above["outcome"] == REPORTED and above["exceeds"] is True
        assert below["outcome"] == REPORTED and below["exceeds"] is False

    def test_missing_value_is_not_run(self):
        assert h8_verdict(None, 0.65)["outcome"] == NOT_RUN


class TestH9:
    def test_mosmed_interval_below_lidc_with_higher_skill_passes(self):
        v = h9_verdict(ci(0.60, 0.55, 0.64), ci(0.78, 0.74, 0.82), 0.30, 0.10)
        assert v["outcome"] == PASS and v["basis"] == "interval"

    def test_overlapping_intervals_fail_because_indistinguishable_falsifies(self):
        v = h9_verdict(ci(0.72, 0.66, 0.79), ci(0.78, 0.74, 0.82), 0.30, 0.10)
        assert v["outcome"] == FAIL and "overlap" in v["reason"]

    def test_a_reversed_ordering_fails(self):
        v = h9_verdict(ci(0.85, 0.82, 0.88), ci(0.70, 0.66, 0.74), 0.05, 0.30)
        assert v["outcome"] == FAIL and "reverses" in v["reason"]

    def test_the_auc_ordering_alone_is_not_enough(self):
        """The statement asserts a lower template AUC *with correspondingly higher*
        skill, so the skill half is part of the claim."""
        v = h9_verdict(ci(0.60, 0.55, 0.64), ci(0.78, 0.74, 0.82), 0.05, 0.30)
        assert v["outcome"] == FAIL and "skill" in v["reason"]

    def test_an_undefined_interval_voids(self):
        assert h9_verdict(ci(0.60), ci(0.78, 0.74, 0.82))["outcome"] == VOID

    def test_a_missing_side_is_not_run(self):
        assert h9_verdict(None, ci(0.78, 0.74, 0.82))["outcome"] == NOT_RUN


class TestBlindingDoesNotEatStatistics:
    """Blinding must rename arms, not the fields that report numbers.

    Substitution is by exact string match and one arm is literally named ``mean``
    -- which is also what ``cluster_bootstrap`` calls its point estimate. Blinding
    keys unconditionally renamed the point estimate of every confidence interval
    in every verdict artefact to that arm's code, so ``ci["mean"]`` raised on the
    published file. The fix is an allow-list of arm-keyed containers, and the
    risk it carries is the opposite one: a container missing from the list leaves
    an arm name in the clear.
    """

    SUBS = {"mean": "ARM-B91812", "abmil": "ARM-799118",
            "mh_abmil": "ARM-BE28CF", "slot:div=0.5": "ARM-0984A5"}

    def test_a_statistic_named_like_an_arm_survives(self):
        ci_like = {"mean": 0.83, "lo": 0.80, "hi": 0.86, "n": 9, "n_clusters": 9}
        out = _blind(self.SUBS, {"lidc_template_auc": ci_like})
        assert out["lidc_template_auc"]["mean"] == 0.83
        assert "ARM-B91812" not in out["lidc_template_auc"]

    def test_an_arm_keyed_container_is_still_blinded(self):
        out = _blind(self.SUBS, {"gaps": {"mean": 0.01, "abmil": 0.02}})
        assert set(out["gaps"]) == {"ARM-B91812", "ARM-799118"}

    def test_nested_statistics_inside_an_arm_keyed_container_survive(self):
        """`per_arm` keys are arms; the dicts underneath them are statistics."""
        out = _blind(self.SUBS, {"per_arm": {"mean": {"mean": 0.7, "lo": 0.6}}})
        inner = out["per_arm"]["ARM-B91812"]
        assert inner == {"mean": 0.7, "lo": 0.6}

    def test_arm_names_in_values_are_blinded_anywhere(self):
        out = _blind(self.SUBS, {"arms_over_threshold": ["mean", "abmil"],
                                 "arm": "mh_abmil"})
        assert out["arms_over_threshold"] == ["ARM-B91812", "ARM-799118"]
        assert out["arm"] == "ARM-BE28CF"

    def test_the_allow_list_covers_every_arm_keyed_field_the_scorers_emit(self):
        """If a scorer grows a new arm-keyed container and it is not added to
        ARM_KEYED, that arm name ships unblinded. Pinned against the falsifiers'
        actual output rather than against a remembered list."""
        ctl = {"masks:axial": 0.1, "masks:separable": 0.03, "centre_prior": 0.05}
        arms = {"mean": 0.01, "abmil": 0.02}
        emitted = [
            h1_verdict(arms, 0.02, ctl, 0.0, 0.0),
            h2_verdict(0.64, arms),
            h4_verdict(arms, 0.15, 0.10),
            h5_verdict(arms, 0.30, {"mean": [0.01], "abmil": [0.02]}),
        ]
        for v in emitted:
            for key, val in v.items():
                if isinstance(val, dict) and set(val) & set(self.SUBS):
                    assert key in ARM_KEYED, (
                        f"{v['id']}.{key} is keyed by arm names but is not in "
                        "ARM_KEYED, so those names would ship unblinded")


class TestH9Power:
    """A FAIL must distinguish 'no ordering' from 'intervals too wide to show one'.

    H9's falsifier counts *indistinguishable* as a failure, and MosMed brings 22
    test clusters against LIDC's 150. The config already flags 38 clusters as
    wide enough that most outcomes land indistinguishable by default, so a FAIL
    here is exactly the kind that could be arithmetic. The diagnostic is what
    lets the paper tell the two apart; it changes no verdict.
    """

    def test_a_wide_interval_fail_is_attributable(self):
        """Point estimates well separated, intervals far too wide to show it."""
        p = _h9_power("t", ci(0.60, 0.40, 0.80), ci(0.78, 0.58, 0.98))
        assert p["orders"] is False
        assert p["observed_separation"] == pytest.approx(0.18)
        # 0.20 of LIDC's lower half plus 0.20 of MosMed's upper half.
        assert p["required_separation"] == pytest.approx(0.40)
        assert p["shortfall"] > 0     # the ordering was arithmetically unreachable

    def test_a_genuinely_absent_ordering_reads_differently(self):
        """Tight intervals, but the means barely differ. Same FAIL, and the
        shortfall is small -- the measurement had the power and the effect was
        not there, which is a finding rather than a limitation."""
        p = _h9_power("t", ci(0.770, 0.760, 0.780), ci(0.780, 0.770, 0.790))
        assert p["orders"] is False
        assert p["observed_separation"] == pytest.approx(0.01, abs=1e-9)
        assert p["required_separation"] == pytest.approx(0.02, abs=1e-9)
        assert p["shortfall"] < 0.02

    def test_a_clearing_pair_has_a_negative_shortfall(self):
        p = _h9_power("t", ci(0.60, 0.55, 0.64), ci(0.78, 0.74, 0.82))
        assert p["orders"] is True and p["shortfall"] < 0

    def test_an_undefined_interval_is_not_computable_rather_than_a_crash(self):
        p = _h9_power("t", ci(0.60), ci(0.78, 0.74, 0.82))
        assert p["computable"] is False
        assert _h9_power("t", None, ci(0.78, 0.74, 0.82))["computable"] is False

    def test_the_diagnostic_agrees_with_the_verdict_on_the_auc_half(self):
        """`orders` must not become a second opinion on the ordering. It is the
        same comparison the falsifier makes, so where the skill half is
        satisfied the two have to agree."""
        for mos, lidc in [(ci(0.60, 0.55, 0.64), ci(0.78, 0.74, 0.82)),
                          (ci(0.72, 0.66, 0.79), ci(0.78, 0.74, 0.82)),
                          (ci(0.85, 0.82, 0.88), ci(0.70, 0.66, 0.74))]:
            v = h9_verdict(mos, lidc, 0.30, 0.10)
            assert _h9_power("t", mos, lidc)["orders"] == (v["outcome"] == PASS)


class TestH10:
    def test_indistinguishable_and_oracle_wins_both_support_the_claim(self):
        """H10 was drafted with two outcomes and mis-scored its own evidence: an
        oracle win is stronger than the claim, not a failure of it."""
        v = h10_verdict(["indistinguishable"] * 3 + ["oracle_wins"] * 2)
        assert v["outcome"] == PASS

    def test_a_majority_of_trained_wins_falsifies(self):
        v = h10_verdict(["trained_wins"] * 3 + ["indistinguishable"] * 2)
        assert v["outcome"] == FAIL and v["majority_needed"] == 3

    def test_a_minority_of_trained_wins_does_not(self):
        assert h10_verdict(["trained_wins"] * 2
                           + ["oracle_wins"] * 3)["outcome"] == PASS

    def test_undefined_seeds_are_excluded_from_the_denominator(self):
        v = h10_verdict(["trained_wins", "trained_wins", None, None, None])
        assert v["n_scored"] == 2 and v["undefined_seeds"] == 3
        assert v["outcome"] == FAIL   # 2 of 2 scored seeds is a majority of 2

    def test_no_scoreable_seed_is_not_run(self):
        assert h10_verdict([None, None])["outcome"] == NOT_RUN


class TestArmSetsExcludeTheArithmeticArms:
    """``mean`` and ``centre_gaussian`` must not enter the arm-scored hypotheses.

    ``mean`` returns exactly 1/n, so every axis AUC ties at 0.5; ``centre_gaussian``
    is constant within a slice, so its within-slice AUC is 0.5 by construction. Both
    are reported in every table with their construction-fixed values, and neither is
    evidence for or against a hypothesis about learned attention.
    """

    @pytest.mark.parametrize("hid", ["H1", "H2", "H4", "H5", "H6", "H10"])
    def test_learned_attention_only(self, hid):
        """The set is exactly the implemented arms declared ``learned_attention``.

        This was a literal count of 7 until transmil made it 8. A literal is the
        wrong guard: it goes red on every arm promotion, which trains the reader
        to bump it, and bumping it is precisely the move that would let a
        misclassified arm through. Comparing against ``scoring_class`` is not
        tautological -- it is the rule ``arm_set`` claims to implement -- and the
        two named exclusions below stay explicit because they are the specific
        failure this class exists to catch.
        """
        pre = prereg.load()
        arms = pre.arm_set(hid)
        assert "mean" not in arms
        assert "centre_gaussian" not in arms
        assert set(arms) == {a["name"] for a in pre.arms(status="implemented")
                             if a.get("scoring_class") == "learned_attention"}

    def test_h8_is_scored_over_every_implemented_arm(self):
        """H8 declares no ``arm_set``, and the pre-freeze default stays the default
        so that adding the mechanism could not silently rescope a hypothesis."""
        pre = prereg.load()
        assert set(pre.arm_set("H8")) == {a["name"] for a in
                                          pre.arms(status="implemented")}


class TestProvenanceGate:
    """An artefact off the amendment chain must void, not score.

    The threshold comes from the current config and the number comes from whatever
    config was frozen when the driver ran. If those differ, comparing them is not a
    pre-registered test of anything -- which is the situation
    ``prereg_freeze.py --check`` reporting ``0 current`` after an amendment
    describes, and the reason an amendment supersedes every stamp on disk.
    """

    @staticmethod
    def _write(tmp_path, name, prereg_hash, role="confirmatory", dirty=False):
        p = tmp_path / name
        p.write_text(json.dumps({
            "analysis_role": role, "results": [],
            "prereg": {"prereg_version": "isbi2027.v1", "prereg_hash": prereg_hash,
                       "prereg_config": "configs/prereg/isbi2027.yaml",
                       "git_commit": "deadbeef", "git_dirty": dirty},
        }))
        return p

    @pytest.fixture
    def art(self):
        pre = prereg.load()
        return Artefacts(pre, pre.hash, prereg.amendment_chain())

    def test_a_current_confirmatory_artefact_is_accepted(self, art, tmp_path):
        art.add("axis_gate", self._write(tmp_path, "a.json", prereg.load().hash))
        assert "axis_gate" in art.docs
        assert art.lineage["axis_gate"] == "current"

    def test_a_superseded_stamp_is_refused(self, art, tmp_path):
        # The hash the first recorded amendment amended away from: on the chain, and
        # therefore knowably an ancestor rather than unknown.
        old = prereg.amendment_chain()[0].before
        art.add("axis_gate", self._write(tmp_path, "a.json", old))
        assert "axis_gate" not in art.docs
        assert "superseded" in art.problems["axis_gate"]

    def test_an_unknown_stamp_is_refused(self, art, tmp_path):
        art.add("axis_gate", self._write(tmp_path, "a.json", "0" * 16))
        assert "unknown" in art.problems["axis_gate"]

    def test_an_exploratory_artefact_is_refused(self, art, tmp_path):
        """Pointed at runs/nulls, the scorer must decline rather than produce a table
        that looks confirmatory."""
        art.add("axis_gate", self._write(tmp_path, "a.json", prereg.load().hash,
                                         role="exploratory"))
        assert "exploratory" in art.problems["axis_gate"]

    def test_a_dirty_tree_stamp_is_refused(self, art, tmp_path):
        art.add("axis_gate", self._write(tmp_path, "a.json", prereg.load().hash,
                                         dirty=True))
        assert "dirty" in art.problems["axis_gate"]

    def test_an_unstamped_artefact_is_refused(self, tmp_path):
        pre = prereg.load()
        art = Artefacts(pre, pre.hash, prereg.amendment_chain())
        p = tmp_path / "a.json"
        p.write_text(json.dumps({"analysis_role": "confirmatory", "results": []}))
        art.add("axis_gate", p)
        assert "no prereg stamp" in art.problems["axis_gate"]

    def test_a_missing_file_is_reported_not_raised(self, art, tmp_path):
        art.add("axis_gate", tmp_path / "nope.json")
        assert "missing" in art.problems["axis_gate"]

    def test_need_reports_the_first_unusable_input(self, art, tmp_path):
        art.add("axis_gate", self._write(tmp_path, "a.json", prereg.load().hash))
        art.add("template_family", None)
        docs, why = art.need("axis_gate", "template_family")
        assert docs is None
        assert why["key"] == "template_family"
        assert why["kind"] == Artefacts.ABSENT


class TestEndToEndWiring:
    """Every hypothesis must actually reach its numbers, not just its rule.

    The rule tests above take extracted values as arguments, so they pass whatever
    the extraction does -- a tag built from an arm ``name`` instead of its ``spec``,
    a renamed key in a driver's output, an ``arm_set`` that resolves to nothing, and
    every one of them still scores. Each of those failures surfaces as NOT_RUN or a
    silently empty arm set, which is exactly the shape of a hypothesis that looks
    scored and is not. So this fabricates one stamped, current, confirmatory
    artefact per driver, over the real declared arms and seeds, and asserts all ten
    hypotheses come out with a real verdict.

    The numbers are chosen to pass, because what is under test is the wiring; the
    falsifier arithmetic is pinned per rule above.
    """

    @pytest.fixture
    def wired(self, tmp_path):
        pre = prereg.load()
        seeds = pre.get("protocol.seeds")
        arms = pre.arm_set("H1")
        specs = [pre.arm(a)["spec"] for a in arms]
        all_specs = [a["spec"] for a in pre.arms(status="implemented")]
        tags = [arm_tag(s, n) for s in specs for n in seeds]

        def stamped(body):
            return {**body, "analysis_role": "confirmatory",
                    "prereg": {"prereg_version": pre.version, "prereg_hash": pre.hash,
                               "prereg_config": pre.path, "git_commit": "cafe",
                               "git_dirty": False}}

        def write(name, body):
            p = tmp_path / name
            p.write_text(json.dumps(stamped(body)))
            return p

        # H1 wants a gap under 0.02, H2 wants every arm's slice AUC under the prior's.
        axis_rows = [{"tag": t, "flat_auc": ci(0.8423, 0.83, 0.86),
                      "slice_auc": ci(0.4822, 0.45, 0.51),
                      "within_slice_auc": ci(0.8411, 0.83, 0.86)} for t in tags]
        # The tie-floor arm: both axes identical, so the gap is exactly zero.
        axis_rows += [{"tag": arm_tag("mean", n), "flat_auc": ci(0.5, 0.5, 0.5),
                       "slice_auc": ci(0.5, 0.5, 0.5),
                       "within_slice_auc": ci(0.5, 0.5, 0.5)} for n in seeds]
        axis_rows.append({"tag": "centre_prior", "flat_auc": ci(0.7740, 0.75, 0.79),
                          "slice_auc": ci(0.6026, 0.58, 0.63),
                          "within_slice_auc": ci(0.7339, 0.71, 0.75)})
        write("axis_gate.json", {"results": axis_rows})

        def family_row(tag, template_auc=0.7865, skill=0.20):
            def scorer(name, flat, within, pinned=False):
                return {"scorer": name, "flat_auc": ci(flat, flat - .01, flat + .01),
                        "slice_auc": ci(0.5, 0.49, 0.51),
                        "within_slice_auc": ci(within, within - .01, within + .01),
                        "pre_registered_denominator": pinned}
            return {
                "tag": tag,
                "scorers": [
                    scorer("attention:inplane", template_auc, template_auc, True),
                    # The positive controls carry axial information, so their
                    # |flat - within| gaps must exceed 0.02. Discovery: 0.0978/0.0259.
                    scorer("masks:axial", 0.7200, 0.6222),
                    scorer("masks:separable", 0.8500, 0.8241),
                    scorer("centre_prior", 0.7740, 0.7339),
                ],
                "prior_normalised_skill": {"trained": skill},
                "h10": {"member": "separable", "outcome": "indistinguishable"},
            }

        write("template_family.json", {"results": [family_row(t) for t in tags]})
        write("h4_cross_patient.json", {"results": [
            {"tag": t, "patient_specific_skill": ci(0.0714, 0.05, 0.09)} for t in tags]})
        write("h7_content_free.json", {"results": [
            {"tag": t, "members": {"chance": {"skill": 0.0},
                                   "roll_permutation": {"skill": 0.0014},
                                   "entropy_matched_random": {"skill": 0.0354},
                                   "fitted_template": {"skill": 0.0}}} for t in tags]})
        write("h8_in_lung.json", {"results": [
            {"tag": arm_tag(s, n),
             "estimands": {"auc": {"in_lung": ci(0.70, 0.67, 0.73)},
                           "stratified_auc": {"in_lung": ci(0.53, 0.50, 0.56)}},
             "h8": {"threshold": pre.hypothesis("H8")["threshold"]}}
            for s in all_specs for n in seeds]})
        write("probe_gate.json", {
            "prior_normalised_skill": {"skill": 0.7305},
            "h7_probe_half": {"threshold": pre.hypothesis("H7")["threshold"]["probe"]}})
        write("untrained_floor.json", {
            "pooling": "slot",
            "floor": {"n_inits": 30, "p95": 0.7666, "median": 0.6739, "p5": 0.6040}})
        # MosMed: a lower template AUC with a non-overlapping interval and higher
        # skill, on the same (arm, seed) tags, which is what H9 asserts.
        write("mosmed_template_family.json", {"results": [
            family_row(t, template_auc=0.60, skill=0.35) for t in tags]})
        return pre, tmp_path, seeds

    @pytest.fixture
    def art(self, wired):
        pre, tmp_path, _ = wired
        a = Artefacts(pre, pre.hash, prereg.amendment_chain())
        for key in ("axis_gate", "template_family", "h4_cross_patient",
                    "h7_content_free", "h8_in_lung", "probe_gate",
                    "mosmed_template_family"):
            a.add(key, tmp_path / f"{key}.json")
        a.add("untrained_floor[0]", tmp_path / "untrained_floor.json")
        assert not a.problems, a.problems
        return a

    def test_all_ten_reach_a_real_verdict(self, wired, art):
        pre, _, seeds = wired
        verdicts = [
            score_h1(pre, art, seeds), score_h2(pre, art, seeds),
            score_h3(pre, art), score_h4(pre, art, seeds),
            score_h5(pre, art, seeds), score_h6(pre, art, seeds),
            score_h7(pre, art), score_h8(pre, art, seeds),
            score_h9(pre, art), score_h10(pre, art, seeds),
        ]
        assert [v["id"] for v in verdicts] == [f"H{i}" for i in
                                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        stuck = [(v["id"], v["reason"]) for v in verdicts
                 if v["outcome"] in (NOT_RUN, VOID)]
        assert not stuck, f"hypotheses that never reached their numbers: {stuck}"
        assert [v["outcome"] for v in verdicts] == [PASS] * 7 + [REPORTED] + [PASS] * 2

    def test_every_declared_arm_is_actually_found(self, wired, art):
        """An arm_set that resolves to tags nobody wrote yields an empty table, and
        an empty table passes every 'no arm exceeds' rule vacuously."""
        pre, _, seeds = wired
        n = len(pre.arm_set("H1"))
        assert len(score_h1(pre, art, seeds)["gaps"]) == n
        assert len(score_h2(pre, art, seeds)["arm_slice_auc"]) == n
        assert len(score_h4(pre, art, seeds)["skills"]) == n
        assert len(score_h5(pre, art, seeds)["mean_over_seeds"]) == n

    def test_h7_reports_the_aggregations_it_did_not_apply(self, wired, art):
        """The verdict attaches to the declared unit; mean and median are carried
        beside it. On the single-draw artefacts the choice between them decided
        H7's outcome in all three conditions, so a reader who cannot see the other
        two cannot tell a robust verdict from a knife-edge one.
        """
        pre, _, _ = wired
        v = score_h7(pre, art)
        assert v["aggregation"] == pre.hypothesis("H7")["content_free_unit"]
        sens = v["aggregation_sensitivity"]
        assert set(sens) == set(pre.hypothesis("H7")["content_free_set"])
        for name, row in sens.items():
            if row["n_tags"]:
                assert row["min"] <= row["median"] <= row["max"]
                assert row["min"] <= row["mean"] <= row["max"]
                # The verdict must read the max and nothing else.
                assert v["content_free"][name] == pytest.approx(row["max"])

    def test_h7_survives_an_artefact_with_no_draw_distribution(self, wired, art):
        """The replication is newer than the artefacts it reads. A pre-replication
        dump has no draw_distribution, and the scorer must report the spread as
        unknown rather than fail -- an exception here would turn a legible
        provenance problem into a crash."""
        pre, _, _ = wired
        v = score_h7(pre, art)
        assert set(v["draw_spread"]) == set(pre.hypothesis("H7")["content_free_set"])
        assert all(s is None for s in v["draw_spread"].values())

    def test_h9_carries_its_power_diagnostic_through_the_artefact_path(self, wired, art):
        """The unit tests pin the arithmetic; this pins that it survives the real
        artefact plumbing and is keyed by the same matched tags the verdict used.
        A diagnostic that silently drops out of the payload is worse than none,
        because the verdict still reads FAIL with nothing to attribute it to."""
        pre, _, _ = wired
        v = score_h9(pre, art)
        assert v["power"]["n_tags"] == len(v["matched_tags"])
        assert set(v["power_per_tag"]) == set(v["matched_tags"])
        if v["power"]["n_computable"]:
            assert "median_required_separation" in v["power"]
            assert "median_observed_separation" in v["power"]
            # `orders` must agree with the verdict's own AUC comparison.
            for tag, p in v["power_per_tag"].items():
                if p["computable"]:
                    assert p["orders"] == (v["per_tag"][tag]["outcome"] == PASS)

    def test_h10_reads_the_pinned_arm_at_every_seed(self, wired, art):
        pre, _, seeds = wired
        v = score_h10(pre, art, seeds)
        assert v["arm"] == pre.hypothesis("H10")["arm"]
        assert v["n_scored"] == len(seeds) and v["undefined_seeds"] == 0

    def test_h6_finds_both_the_arm_and_its_base(self, wired, art):
        pre, _, seeds = wired
        v = score_h6(pre, art, seeds)
        assert v["skill_arm"] is not None and v["skill_base"] is not None
        assert v["slice_arm"] is not None and v["slice_base"] is not None

    def test_a_stale_threshold_in_a_driver_voids(self, wired, art, tmp_path):
        """probe_gate.py and h8_in_lung.py each carry their own copy of a bound. If a
        copy disagrees with the config, one of the two is stale and neither can be
        trusted to be the pre-registered one."""
        pre, _, seeds = wired
        doc = art.docs["probe_gate"]
        doc["h7_probe_half"]["threshold"] = 0.42
        assert score_h7(pre, art)["outcome"] == VOID

    def test_the_verdict_table_carries_no_arm_identifier(self, wired, art):
        """No mapping key or string value may *be* an arm name, spec or tag.

        Checked structurally rather than by substring search over the serialised
        blob: the ``mean`` arm's name is an English word and occurs inside
        ``"mean_over_seeds"``, which is prose about a unit and not an arm reference.
        A substring test would fail on that and pass on a tag it never thought to
        look for, which is backwards.
        """
        pre, _, seeds = wired
        subs = blind_substitutions(pre, pre.blind_key(), seeds)
        secret = {k for k, v in subs.items() if k != v}
        assert secret, "no arm is blinded, so this test proves nothing"

        def walk(obj, path="$"):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    assert k not in secret, f"arm identifier leaked as a key at {path}"
                    walk(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    walk(v, f"{path}[{i}]")
            elif isinstance(obj, str):
                assert obj not in secret, f"arm identifier leaked as a value at {path}"

        for score in (score_h1, score_h4, score_h5, score_h10):
            walk(_blind(subs, score(pre, art, seeds)), f"$.{score.__name__}")


class TestBlinding:
    """Every spelling of an arm has to be substituted, or none of them is blinded."""

    @pytest.fixture
    def subs(self):
        pre = prereg.load()
        return pre, blind_substitutions(pre, pre.blind_key(), [0, 1, 2, 3, 4])

    def test_name_spec_and_tag_all_map_to_the_code(self, subs):
        pre, s = subs
        arm = pre.arm("slot_div0.5")
        code = pre.blind_key().code("slot_div0.5")
        assert s["slot_div0.5"] == code
        assert s[arm["spec"]] == code
        assert s[arm_tag(arm["spec"], 3)] == f"{code}_seed3"

    def test_a_tag_keyed_table_is_blinded(self, subs):
        _, s = subs
        table = {"per_seed": {arm_tag("slot:div=0.5", 0): 0.1,
                              arm_tag("abmil", 0): 0.2}}
        out = _blind(s, table)
        for k in out["per_seed"]:
            assert "slot" not in k and "abmil" not in k

    def test_unblinded_arms_and_references_pass_through(self, subs):
        _, s = subs
        # centre_gaussian is declared blind: false -- the estimands need to know
        # which arm is a fixed geometric rule -- and references are never blinded.
        assert s["centre_gaussian"] == "centre_gaussian"
        assert _blind(s, {"centre_prior": 0.6}) == {"centre_prior": 0.6}

    def test_no_declared_code_leaks_its_arm_name(self, subs):
        pre, _ = subs
        key = pre.blind_key()
        for name, code in key.codes.items():
            assert name.lower() not in code.lower()
