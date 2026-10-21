# SlotMIL

**Position Priors Confound Weakly Supervised CT Localization.**

Volumetric CT forms a three-dimensional image from a stack of two-dimensional
cross-sections. This project asks whether weakly supervised attention actually
locates a patient's lesion in that volume or merely identifies anatomical
coordinates where lesions commonly occur. It decomposes localization AUC into
axial and in-plane components and evaluates two prior-normalized target
quantities (estimands); the reported numerical values are their estimates. A
supervised patch classifier serves as a positive control. Every claim is tied to
a hypothesis frozen before the confirmatory runs, and the analysis plan and its
amendment chain ship with the code.

The project began by evaluating slot attention as an interpretable MIL pooling
bottleneck. The initial hypothesis was not supported, and the study was
refocused on the evaluation of positional confounding. [`plan.md`](plan.md) is
the original research plan and is retained as a historical record.

## The claim

Instance-level localization metrics for weakly supervised MIL in volumetric CT
largely measure in-plane position. On LIDC-IDRI, across ten model/reference
configurations and five random seeds:

- flat (3D) instance AUC does not separate from within-slice AUC by more than
  0.02 for seven of eight scored arms;
- no trained arm reaches a model-free center prior on the axial axis;
- the 95th percentile over 30 **untrained** initializations is 0.7666, so 0.5 is
  not the floor;
- a content-free separable reference fitted on validation masks reaches 0.8543 —
  above every trained arm;
- scoring an arm's attention against a different scan's masks beats scoring it
  against the right ones for five of eight arms.

Harvey-style Normal Guidance is the main exception: it raises
patient-specific skill by +0.1509 against a pre-registered falsifier of 0.02, and
is independently the only arm the axis gate flags as carrying axial content.
Explicit prior guidance, not architecture alone, is what adds measurable localization.

The full manuscript distinguishes two adjacent approaches: **Slot-MIL**, the
slot aggregator introduced in the *Slot-Mixup with Subsampling* paper
(arXiv:2311.17466) for pathology WSI, and **INSIGHT** (arXiv:2412.02012) for
non-slot CT MIL. The prior-normalized estimands are adapted from Kümmerer et al.
(PNAS 2015) and Borji/Bylinskii, and Harvey/Loevlie (arXiv:2604.26807 and
arXiv:2605.27306) previously reported the content-free-baseline observation.
This study contributes the axial/in-plane decomposition against voxel masks,
the per-condition power gate, and the per-dataset measurement.

Full write-up: [`RESULTS.md`](RESULTS.md) from the "Confirmatory results"
heading. Manuscript sources: [`paper/`](paper/).

## Dataset access

The paper uses two public, de-identified CT datasets. The links below identify
both the original collection and the exact mirrors used by the acquisition code.

| Dataset | Source used in this project | Role |
|---|---|---|
| LIDC-IDRI | [Official TCIA collection](https://www.cancerimagingarchive.net/collection/lidc-idri/) · [NBIA collection search](https://nbia.cancerimagingarchive.net/nbia-search/?CollectionCriteria=LIDC-IDRI) · [`download_lidc_staged.py`](scripts/download_lidc_staged.py) | Primary dataset; nodule-presence, malignancy, and balanced-presence conditions |
| MosMedData | [Original dataset page](https://mosmed.ai/en/datasets/datasets/) · [1,110-volume Kaggle mirror](https://www.kaggle.com/datasets/mathurinache/mosmeddata-chest-ct-scans-with-covid19) · [50-mask Kaggle mirror](https://www.kaggle.com/datasets/andrewmvd/mosmed-covid19-ct-scans) · [`download_mosmed.py`](scripts/download_mosmed.py) | Secondary dataset used only for the pre-registered cohort contrast |

Review each provider's current license and access terms before downloading or
redistributing data. Dataset files are not included in this repository.

## Confirmatory configuration

The machine-readable source of truth is
[`configs/prereg/isbi2027.yaml`](configs/prereg/isbi2027.yaml), config hash
`4fd6e801157ecef5`. This table summarizes the settings needed to interpret the
paper; the YAML and [`AMENDMENTS.md`](AMENDMENTS.md) control if any summary and
the frozen protocol differ.

| Component | Frozen setting |
|---|---|
| Primary split | LIDC nodule presence: 700/149/150 series, patient grouped, split seed 2027 |
| Consistency checks | LIDC malignancy: 512/111/111 series; balanced presence: 184/40/38 series; MosMed uses its declared existing split |
| Features | Frozen pretrained DINOv2 ViT-B/14; $224\times224$ slices; $16\times16$ patch grid; float32 attention outputs |
| Model sweep | Mean, ABMIL, gated ABMIL, multi-head ABMIL, slot attention (diversity 0.5), fixed center Gaussian, Normal Guidance ($\lambda=0.1$), CLAM-SB, DSMIL, and TransMIL |
| Training | Seeds 0--4; AdamW, learning rate and weight decay $10^{-4}$; batch size 4; cosine schedule; at most 40 epochs; early-stopping patience 10; validation scan AUC checkpoint |
| Bag construction | At most 48 slices during training; every retained slice at evaluation; positive patch iff pooled lesion coverage is nonzero |
| Map selection | Best validation-mask map selected for multi-map models and frozen before testing |
| Uncertainty | 10,000-draw patient-clustered percentile bootstrap; RNG seed 0 |
| Spatial controls | 30 untrained initializations per selected pooling; 100 swapped-mask derangements; validation-fitted in-plane, axial, joint, and separable references |
| Lung restriction | Threshold below $-320$ HU, border-air removal, two largest 3D components, closing, empty-slice repair, and per-slice hole filling |

## Confirmatory results

The primary `nodule_present` condition carries the hypothesis family. “FAIL”
means the pre-registered prediction was falsified; H6 and H9 therefore remain
reportable findings rather than omitted experiments.

| Hypothesis | Verdict | Primary result |
|---|---|---|
| H1 | PASS | Seven of eight learned arms have $|\mathrm{flat}-\mathrm{within}|<0.02$; Normal Guidance is the exception at 0.0669 |
| H2 | PASS | Fixed center-prior slice AUC 0.6427; center prior minus Normal Guidance $+0.0109$, 95% CI $[-0.0078,+0.0300]$ |
| H3 | PASS | Untrained slot 95th percentile 0.7666; multi-head ABMIL 0.7679 |
| H4 | PASS | Median patient-specific skill $-0.0466$; five of eight learned-arm means are negative |
| H5 | PASS | Maximum mean prior-normalized skill 0.1990 versus the 0.30 threshold |
| H6 | FAIL | Normal Guidance improves patient-specific skill by $+0.1509$, exceeding the 0.02 falsifier |
| H7 | PASS after disclosed amendment | Original single-draw gate failed at 0.0784; the result-informed 30-draw amendment passes while retaining maximum-over-tags aggregation |
| H8 | REPORTED | Mean in-lung fitted-reference AUC 0.4884; slot attention falls from 0.8247 to 0.5560 |
| H9 | FAIL | Predicted ordering reverses: MosMed axial slice AUC 0.8455 versus LIDC 0.6180 |
| H10 | PASS | The separable mask-fitted reference ties or beats the trained slot arm on all five seeds |

Full estimates, consistency conditions, intervals, amendments, and provenance
are in [`RESULTS.md`](RESULTS.md#confirmatory-results-2026-08-18-config-hash-4fd6e801157ecef5).

## Pre-registration

| | |
|---|---|
| Frozen config | [`configs/prereg/isbi2027.yaml`](configs/prereg/isbi2027.yaml) |
| Config hash | `4fd6e801157ecef5` |
| Document | [`PREREGISTRATION.md`](PREREGISTRATION.md) |
| Amendment chain | [`AMENDMENTS.md`](AMENDMENTS.md) |
| Target | ISBI 2027, deadline 2026-10-26 |

No threshold is retyped into Python: every bound comes from the frozen YAML
through `Prereg.hypothesis`/`arm_set`, and the falsifiers themselves live in
[`slotmil/eval/verdict.py`](slotmil/eval/verdict.py) so they are testable without
data. Every artifact carries the config hash and git commit it was produced
under. Verify the chain — hash matches the committed config, every declared split
still hashes as recorded, no artifact carries an off-chain hash:

```bash
.venv/bin/python scripts/prereg_freeze.py --check
```

Two of the ten hypotheses failed under their pre-registered falsifiers, and both
are reported as such: H6 because prior guidance adds patient-specific
information, and H9 because the predicted ordering of spatial consistency is
reversed.

## Reproducing the paper

Every analysis driver is a standalone CLI that reads cached attention outputs,
needs **no GPU**, and writes a stamped artifact. The scorer recomputes nothing —
it reads only stamped JSON — so all ten verdicts re-derive in seconds.

```bash
# all ten verdicts for one condition
.venv/bin/python scripts/prereg_verdict.py --dir runs/nulls_nodule_present_confirmatory ...

# H5's floored-denominator sensitivity (outside the confirmatory family)
.venv/bin/python scripts/h5_floored_denominator.py \
    --dir runs/nulls_nodule_present_confirmatory \
    --out runs/nulls_nodule_present_confirmatory/h5_floored_denominator.json \
    --condition nodule_present --role confirmatory

# every figure and table in the manuscript, from JSON alone
.venv/bin/python scripts/make_paper_figures.py --out paper/figures --tables paper/tables
cd paper && make            # main.pdf + supplement.pdf
```

`make_paper_figures.py` also writes `paper/figures/figure_data.json`: every
plotted number, keyed by figure and series, with its source artifact and that
artifact's config hash and commit. This file can be used to verify the values in
each figure.

> **The `--out` trap.** `axis_gate.py`, `template_family.py` and `h8_in_lung.py`
> default to `--out runs/nulls/<name>.json` — the *exploratory* artifacts that set
> every threshold in `PREREGISTRATION.md`. A confirmatory run that omits `--out`
> overwrites them. Always pass `--dir` **and** `--out`. The two scripts added for
> the paper have no default `--out` at all, on purpose.

## Setup

```bash
bash env/setup_env.sh   # venv beside the project; ~6 GB
pytest -q               # bare pytest, as CI runs it
```

Environment is pinned around two real constraints: `numpy<2` and `sqlalchemy<2`
because `pylidc` 0.2.3 predates both breaks, and `setuptools<81` because `pylidc`
imports `pkg_resources` at module load.

## Layout

| Path | What |
|---|---|
| `configs/prereg/isbi2027.yaml` | The frozen pre-registration — conditions, arms, estimands, all ten falsifiers |
| `slotmil/prereg.py` | Config loading, hashing, blinding, stamping, the amendment chain |
| `slotmil/eval/verdict.py` | The ten falsifiers as pure functions over already-extracted numbers |
| `slotmil/eval/estimands.py` | Prior-normalized skill, patient-specific skill, cluster bootstrap |
| `slotmil/eval/templates.py` | The content-free template family (in-plane / axial / separable / joint) |
| `slotmil/models/slot_attention.py` | Locatello slot attention + implicit differentiation + bag masking |
| `slotmil/models/baselines.py` | mean/max/ABMIL/Gated/MH-ABMIL/CLAM-SB/DSMIL/TransMIL/center-Gaussian |
| `slotmil/data/feature_cache.py` | HDF5 bag dataset — the engineering asset the project rests on |
| `scripts/prereg_verdict.py` | Scores all ten hypotheses from stamped artifacts. Blinded |
| `scripts/make_paper_figures.py` | Figures + LaTeX tables, JSON in, no GPU |
| `paper/` | Manuscript and companion sources |

## Three implementation details that are load-bearing

**1. Padded instances must be zeroed after the softmax.** Slot attention
normalizes over *slots*, so padding survives the softmax and then contributes
mass to the weighted mean over instances, silently rescaling every real weight.
Nothing crashes; long-bag results are just quietly wrong. CT bags run 65–764
slices, so this would have been constant. Note you cannot mask the logits with
`-inf` as in ordinary attention — every slot at a padded position would be
`-inf`, giving NaN. Tested by `test_padding_invariance`.

**2. Implicit differentiation runs `iters-1` no-grad steps + 1 grad step**, not
`iters` + 1. The obvious reading of Chang et al. gives implicit mode T+1
refinements against vanilla's T, which would confound the vanilla-vs-implicit
ablation with a difference in effective depth. Tested by
`test_implicit_and_vanilla_agree_on_forward`.

**3. `pylidc` needs the DICOM on disk to build a mask.** Placing annotation
contours into a volume requires per-slice z positions from the DICOM headers,
which are not in pylidc's bundled annotation database. The staged pipeline must
therefore extract masks *before* deleting each batch. Getting this order wrong
costs a 124 GB re-download. That cost is real but measured: the lung-mask array
(job 7249296, 2026-08-14) re-downloaded all 999 series in ~55 minutes at 8-way
parallelism, and the staged single-stream pipeline does it inside one 24 h GPU
job. The second feature cache completed after five resumable TCIA retries.

## Storage

The canonical project root is
`/fs/nexus-projects/Cognitive_Decline_Dynamics_Project/SlotMIL` on the 1 TB
project volume. `/fs/nexus-scratch/bthapar/SlotMIL` is a compatibility symlink
for frozen artifacts and older job commands that record the original absolute
path.

LIDC is 1,018 CT series / 243,958 slices / **~124 GB of DICOM** (measured via the
TCIA API). `scripts/download_lidc_staged.py` downloads in batches and deletes
each batch's DICOM once features and masks are cached, holding peak transient
usage near 15 GB. It aborts if free space drops below `--min-free-gb`, and skips
series already cached so preemption costs one batch rather than the run.

Current tree is ~180 GiB: data ~158 GiB, runs ~17 GiB, and the virtualenv
~5.7 GiB. The project volume leaves roughly 820 GiB for retries and extensions.

## Data status

| Dataset | Status |
|---|---|
| LIDC-IDRI | **Cached.** 999 series, DINOv2 ViT-B/14, all 999 with lesion masks. Source DICOM deleted by the staged pipeline |
| MosMed | **Cached.** 1110 studies; only 50 carry masks, all CT-1 — which is why it supports one hypothesis |
| MedMNIST3D | Present. Public, auto-downloads. Splits verified 1158/165/310 |
| COV19D | **Downloaded and integrity-checked.** Five RAR5 training archives, one ZIP validation archive, and four center CSVs are present. Access was granted through Teli; this dataset is outside the pivoted confirmatory paper and carries a 12-paper citation obligation |
| RadImageNet DenseNet-121 | **Cached.** Jobs 7312418 and 7362809 produced a 28.38 GiB cache covering all 1018 TCIA series at 512 px, 16×16 grid, dim 1024. Weights are in `data/weights/RadImageNet_pytorch/` |
