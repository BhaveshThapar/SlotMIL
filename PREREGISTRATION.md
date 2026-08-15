# Pre-registration — ISBI 2027

**Frozen config hash:** `20bdd93b781d950d`

**Config:** [`configs/prereg/isbi2027.yaml`](configs/prereg/isbi2027.yaml) ·
**Amendments:** [`AMENDMENTS.md`](AMENDMENTS.md) ·
**Target:** ISBI 2027, deadline 26 October 2026

---

## Why this document exists

This project has corrected itself three times: a metric bug (`attn.max(axis=0)`
on slot-normalised attention), an unmeasured null (`chance = 0.5` asserted and
never checked, real floor 0.64–0.83), and a headline that rested on a single seed
(`slice AUC = 0.4822`, actually 0.4822–0.5565 across seeds). Each correction was
found by us, which is the system working — but three in three weeks means the
next positive-looking result deserves no benefit of the doubt, including from us.

So the analysis plan is fixed in advance, in a machine-readable form the code
reads directly, and committed to git before the confirmatory runs. The claim this
document supports is not "we were careful." It is: *here is what we said we would
measure, here is the commit proving we said it first, and here is what would have
falsified it.*

## What is covered, and what is not

**Confirmatory** — everything below, computed on the LIDC confirmatory split
(seed 2027) and MosMed, after this file was committed: the untrained-init
reference floor, the architecture sweep, the condition sweep, the in-lung
control, and every estimand computed on them.

**Exploratory** — everything computed before this file was frozen: the null
battery, the static-template analysis, the crosstab decomposition, the axis gate,
the fp16 check, and all published SlotMIL results. These set the thresholds
below. Using discovery data to choose confirmatory thresholds is legitimate and
standard; hiding that it happened would not be. The paper labels these results as
exploratory wherever they appear.

**The discovery/confirmation boundary is a fresh split, not just a promise.** The
existing LIDC split (seed 0, hash `9f468f8bd8a59228`) is the split the protocol
was developed against, so it cannot carry confirmatory claims. A new
patient-grouped split (seed 2027) was drawn for confirmation. Re-splitting is
clean here because the DINOv2 feature cache is frozen and unsupervised — no label
information crosses the boundary — so this costs one retraining sweep and nothing
else.

MosMed is **not** re-split: only 50 of 1110 scans carry masks and all are CT-1, so
a fresh split would leave too few annotated scans in test to estimate anything.
MosMed therefore supports one hypothesis (H9, the stereotypy contrast) and is
reported with that caveat attached.

## Estimands

Two primary, and both are **imports from the saliency literature, not
inventions**. We claim the construction and the application, not the idea.

| estimand | formula | prior art |
|---|---|---|
| **prior-normalised skill** | `(AUC − AUC_template) / (1 − AUC_template)` | information gain over a baseline model — Kümmerer, Wallis & Bethge, *PNAS* 2015 |
| **patient-specific skill** | `AUC_real − AUC_cross-patient` | shuffled AUC — Borji et al. ICCV 2013; Bylinskii et al. *TPAMI* 2019 |

`AUC_template` is a 256-number in-plane map fit on validation that never reads a
test image. It is a tighter denominator than Arun et al.'s (*Radiology: AI* 2021)
unfit average-mask baseline or Harvey et al.'s fixed σ=1 slice Gaussian, both of
which it should beat — which is the point: the stronger the content-free
reference, the more honest the skill number above it.

Secondary: flat instance AUC, slice AUC, within-slice AUC, position-stratified
AUC (normalised slice × 6 radial bins), AP lift over prevalence, and in-lung
stratified AUC. Dice and pointing game are **not** reported as primary evidence —
at a lesion-patch prevalence of 0.00068 they are arithmetically dead.

The decomposition is reported as a **nested chain of reference models**
(chance → fitted template → cross-patient → full), as differences between
adjacent links. Not as an additive three-term sum: AUC is not naturally additive
and the additive form invites a correct objection.

## Protocol, fixed in advance

- **Attention dtype `float32`.** fp16 caching cost seed 2 0.034 AUC.
- **Seeds** 0–4 per arm; **≥30** untrained inits for the floor, reported as a
  distribution (median, 5th–95th percentile), never a point.
- **Instance label** is `patch_lesion_fraction > 0`. The cached mask is a coverage
  fraction topping out near 0.05 — a nodule never fills more than ~5% of a 22 mm
  patch — so this means "contains at least one lesion voxel". Generous, and part
  of why position dominates. Sensitivity to the threshold is a declared secondary
  analysis.
- **Slot selection** is Hungarian assignment on complementary findings, fit on
  validation and frozen before test. For F=2 complementary columns this is
  algebraically argmax of the lesion column; the write-up must not overstate it
  as a naming step.
- **Bag inclusion**: drop bags that are all-positive or all-negative; require the
  instance count to be a multiple of the per-slice patch count.
- **Statistics**: patient-level *cluster* bootstrap (10 000 reps, RNG seed 0,
  95% percentile CI) — patients, not bags, because a LIDC patient can contribute
  more than one series. DeLong for paired AUC. Holm correction across the
  hypothesis family. α = 0.05.

## Arms

Declared before implementation — that is what pre-registration is for. Each arm
is `implemented` or `planned`; `tests/test_prereg.py` asserts every implemented
arm is constructible and every planned arm is listed, so an arm can be unbuilt
but never silently missing.

Implemented today: `mean`, `abmil`, `gated_abmil`, `mh_abmil`, `slot:div=0.5`.
Planned: `centre_gaussian` and `normal_guidance` (Harvey et al., and the arms
that carry H6), `clam_sb`, `dsmil`, `transmil`.

**Pre-committed drop rule.** TransMIL faces LIDC bags of ~43.8k instances, an
order of magnitude beyond the whole-slide bags it was designed for. If it is not
training by **22 September 2026** it is cut, and the paper reports the arm count
achieved. The deadline does not move for an arm.

## Hypotheses and falsifiers

Every hypothesis has a numeric way to fail. A hypothesis that cannot fail is
ceremony.

| id | statement | falsifier |
|---|---|---|
| **H1** *(lead)* | for every arm, \|flat AUC − within-slice AUC\| < 0.02 — the reported 3D metric is an in-plane metric wearing 3D clothing | fails for a majority of arms → **the paper's lead claim is withdrawn** |
| **H2** | the model-free centre prior's slice AUC exceeds every trained arm's | any trained arm beats it on the slice axis |
| **H3** | over ≥30 untrained inits, 95th-percentile instance AUC > 0.70 | 95th percentile ≤ 0.70 |
| **H4** | every arm's patient-specific skill < 0.15; median < 0.10 | any arm ≥ 0.15, or median ≥ 0.10 |
| **H5** | no arm's prior-normalised skill exceeds 0.30 | any arm exceeds 0.30 |
| **H6** | Normal Guidance raises slice-level AUC over its base arm but raises patient-specific skill by < 0.02 | skill rises ≥ 0.02 — prior injection would be adding real information and the critique fails |
| **H7** *(power gate)* | the supervised patch probe's prior-normalised skill > 0.50 while every content-free baseline's is < 0.05 | the probe fails to separate → **the proposed estimands have no power and must not be recommended** |
| **H8** *(two-sided)* | in-lung fitted-template AUC > 0.65 | none — reported either way; if restricting to lung removes the prior, that becomes the paper's recommendation |
| **H9** | MosMed's fitted-template AUC is lower than LIDC's, with correspondingly higher prior-normalised skill | ordering reverses or is indistinguishable |

H7 is the one that matters most for the constructive half. A metric that scores
zero for everything is not a fix, it is nihilism; if the supervised probe does not
clear it, we do not recommend the protocol.

## Blinding — what it is and what it is not

Model arms are shown to the analysis layer as opaque codes (`ARM-XXXXXX`, an HMAC
of the arm name under a salt in `runs/prereg/unblind_key.json`). Reference
baselines are never blinded, because the estimands need to know which one is the
template. Unblinding is done through `scripts/prereg_unblind.py`, which appends a
dated entry to `AMENDMENTS.md`.

**This is procedural, not cryptographic.** The salt sits on the same disk as the
analysis; anyone running the analysis can read it and skip the script. It raises
the cost of *casual* mid-iteration peeking and creates an audit point for the
deliberate kind. It does not make cheating impossible, and nothing in the paper
will claim it does.

## Amending this

Anything not declared here raises `PreregViolation` rather than falling back to a
default. When the plan genuinely needs to change:

1. edit `configs/prereg/isbi2027.yaml`;
2. re-run `scripts/prereg_freeze.py --amend`;
3. add a dated entry to `AMENDMENTS.md` stating what changed, why, and
   **whether the affected results had already been seen**.

That last clause is the one that carries the weight. An amendment made before
looking is a plan change; one made after is a result-dependent choice, and the
affected numbers get labelled exploratory.

## Verifying the chain

A reader with the repository can check all of this without trusting us:

```bash
# 1. the doc's hash matches the committed config, and splits still verify
.venv/bin/python scripts/prereg_freeze.py --check

# 2. the enforcement tests pass
.venv/bin/python -m pytest tests/test_prereg.py -q

# 3. this commit predates every confirmatory-results commit
git log --oneline --follow PREREGISTRATION.md
```

Every result file carries a top-level `prereg` block with the config hash and the
git commit that produced it. If a result's hash does not match the frozen config,
that result is exploratory — regardless of what the surrounding prose says.
