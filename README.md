# SlotMIL

**Position Priors Confound Weakly Supervised CT Localization**

Code and companion analyses for our ISBI 2027 manuscript. We evaluate how spatial
priors affect weakly supervised CT localization using frozen DINOv2 features,
ten configurations, and five training seeds. Eight configurations learn attention;
mean pooling and the fixed center Gaussian are reference configurations.

The study evaluates localization metrics, not clinical utility. Training uses scan
labels, but multi-map selection uses validation lesion masks. Randomly initialized
MIL models also use pretrained features and validation-mask map selection.

## Datasets

| Dataset | Access and citation | Use in this paper |
|---|---|---|
| LIDC-IDRI | [TCIA collection and downloads](https://www.cancerimagingarchive.net/collection/lidc-idri/); [dataset DOI](https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX); [Armato et al., 2011](https://doi.org/10.1118/1.3528204) | 999 retained CT series from 991 patients; patient-disjoint train/validation/test split of 700/149/150. Primary task: nodule presence. |
| MosMedData | [Official dataset portal](https://mosmed.ai/datasets/covid19_1110); [Morozov et al., 2020](https://arxiv.org/abs/2005.06465) | 1,110 CT studies; 50 have lesion masks, including 22 in test. Used for the H9 dataset contrast, not a transfer-performance claim. |

Obtain data under the providers' current access terms. The MosMed portal may
require registration; its availability could not be confirmed during this update.
The existing [`scripts/download_mosmed.py`](scripts/download_mosmed.py) documents
Kaggle-based acquisition, requiring your own credentials. Raw images and feature
caches are not distributed with this repository. LIDC acquisition is implemented
in [`scripts/download_lidc_staged.py`](scripts/download_lidc_staged.py).

Split manifests are committed under [`data/lidc/`](data/lidc/) and
[`data/mosmed/`](data/mosmed/); use the exact paths in the frozen configuration.
Malignancy (734 series) and balanced nodule presence (262 series) are secondary
LIDC consistency checks. Historical experiments with other datasets are outside
the two-dataset ISBI analysis; see the historical data-status section below.

## Configuration

The analysis contract is [`configs/prereg/isbi2027.yaml`](configs/prereg/isbi2027.yaml)
(hash `4fd6e801157ecef5`). Training settings also come from
[`scripts/train_cached.py`](scripts/train_cached.py) and
[`slotmil/train.py`](slotmil/train.py); the frozen YAML is not a complete training
CLI configuration. Do not edit it merely to change local paths or document defaults.

| Setting | Paper configuration |
|---|---|
| Encoder | Frozen pretrained DINOv2 ViT-B/14; 768-dimensional tokens |
| Input | HU clipped to [-1000, 400]; 224 × 224 slices; 16 × 16 patch grid |
| Positive patch | Any lesion coverage (`patch_lesion_fraction > 0`) |
| Training bag | Up to 48 slices; evaluation uses every retained slice |
| Optimizer | AdamW; learning rate 1e-4; weight decay 1e-4 |
| Schedule / stopping | Cosine schedule; at most 40 epochs; patience 10 |
| Batch size / checkpoint | 4; selected by validation scan AUC |
| Training seeds | 0, 1, 2, 3, 4 |
| Slot arm | `slot:div=0.5`; 8 slots, dimension 256, 3 iterations; implicit differentiation |
| Normal Guidance | `normal_guidance:lam=0.1`; detached Gaussian moments; variance floor 1 slice² |
| Map selection | Fit on validation lesion masks, then frozen for test |
| Uncertainty | 10,000 patient-clustered bootstrap draws |
| Random MIL controls | 30 initializations each of slot attention and multi-head ABMIL |
| Swapped-mask control | 100 permutations; crop/repeat donor masks; no anatomical registration |

The slot-query initialization remains frozen under implicit differentiation;
see `arms[].init_contract` and [`AMENDMENTS.md`](AMENDMENTS.md). Architecture
adaptations and hypothesis-specific aggregation rules are in the frozen YAML.

Example: train the paper's slot arm after preparing the cache and split manifests:

```bash
python scripts/train_cached.py \
  --cache data/lidc/features_dinov2_vitb14.h5 \
  --splits data/lidc/splits_confirmatory.json \
  --out runs/reproduction_slot \
  --arms 'slot:div=0.5' --seeds 0 1 2 3 4 \
  --epochs 40 --batch-size 4 --lr 1e-4 \
  --dim 256 --num-slots 8 --iters 3 \
  --instance-level patch --max-slices 48 --role confirmatory
```

This is a single-arm training example, not the full analysis pipeline. It requires
the prepared data and a suitable GPU; the default three-seed CLI invocation does
not reproduce the paper's five-seed design.

## Results

Primary LIDC nodule-presence localization results (means over five seeds).
Localization AUC is averaged over eligible scans; the primary sample contains
131 series from 129 patients. These are localization, not classification AUCs.

| Trained attention arm | Whole-volume AUC | In-lung AUC | Whole-volume minus in-lung |
|---|---:|---:|---:|
| Slot attention (diversity 0.5) | 0.8247 | 0.5560 | +0.2688 |
| Normal Guidance | 0.6055 | 0.5087 | +0.0968 |
| TransMIL | 0.4263 | 0.5186 | -0.0924 |
| Multi-head ABMIL | 0.4126 | 0.5176 | -0.1050 |
| DSMIL | 0.3961 | 0.4734 | -0.0773 |
| ABMIL | 0.2963 | 0.4519 | -0.1556 |
| Gated ABMIL | 0.2629 | 0.4595 | -0.1966 |
| CLAM-SB | 0.2390 | 0.4774 | -0.2384 |

Source: [`paper/tables/arm_in_lung.tex`](paper/tables/arm_in_lung.tex).
Differences are computed before rounding. Lung restriction changes the negative
comparison population; the drop is not a numerical estimate of confounding.

| Control or comparison | Reported result |
|---|---|
| Validation-mask-fitted in-plane reference | Flat AUC 0.8302 |
| Validation-mask-fitted separable reference | Flat AUC 0.8543 |
| Random slot initializations | Median 0.6739; 5th–95th percentiles 0.6040–0.7666 |
| Random multi-head ABMIL initializations | 95th percentile 0.7679 |
| Fixed center prior / Normal Guidance | Slice AUC 0.6427 / 0.6318 |
| Center prior minus Normal Guidance | Paired difference +0.0109; 95% CI [-0.0078, +0.0300] |
| Normal Guidance minus gated ABMIL | Patient-specific skill improves by +0.1509 (-0.1019 → +0.0490) |
| Highest arm-mean prior-normalized skill | 0.1990 (one seed reaches 0.3432) |
| Mask-fitted axial reference, MosMed / LIDC | Slice AUC 0.8455 / 0.6180 |

The upper tail of random initializations is not a universal performance floor,
and mask-fitted references are benchmarks, not theoretical ceilings. The paired
interval does not establish equivalence between Normal Guidance and the center
prior. Its +0.1509 improvement exceeds a preregistered effect threshold, not a
paired statistical-significance threshold. Five of eight arm means have negative
correct-minus-swapped-mask AUC; six primary donor assignments pair different
series from the same patient, so the control is not strictly patient-disjoint.

**Protocol disclosure:** H6 and H9 predictions were not supported. The original
single-draw H7 gate failed (control maximum 0.0784). A results-informed amendment
on 2026-08-17 introduced 30 stochastic-control draws per output, retaining the
maximum-over-outputs rule. The amended H7 passes for nodule presence and
malignancy but fails for balanced presence. These amended passes do not erase
the original failure. The flat/within-slice gap is below 0.02 for seven of eight
trained arms; this is a descriptive threshold comparison, not an equivalence test.

See [`RESULTS.md`](RESULTS.md), [`AMENDMENTS.md`](AMENDMENTS.md), and
[`paper/`](paper/) for detailed analyses. Historical prose may predate the revised
ISBI manuscript; the qualifications above follow the supplied submission draft.

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
data. Every artefact carries the config hash and git commit it was produced
under. Verify the chain — hash matches the committed config, every declared split
still hashes as recorded, no artefact carries an off-chain hash:

```bash
.venv/bin/python scripts/prereg_freeze.py --check
```

H6 and H9 predictions failed; H7 must be reported separately under its original
and amended protocols, as detailed above.

## Reproducing the paper

Every analysis driver is a standalone CLI that reads cached attention dumps,
needs **no GPU**, and writes a stamped artefact. The scorer recomputes nothing —
it reads only stamped JSON — so all ten verdicts re-derive in seconds.

```bash
# all ten verdicts for one condition
.venv/bin/python scripts/prereg_verdict.py \
    --dir runs/nulls_nodule_present_confirmatory \
    --mosmed-template-family runs/nulls_mosmed_severity_confirmatory/template_family.json \
    --condition nodule_present \
    --out runs/nulls_nodule_present_confirmatory/prereg_verdict.json

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
plotted number, keyed by figure and series, with its source artefact and that
artefact's config hash and commit. A figure that disagrees with that file is a
bug in the figure.

> **The `--out` trap.** `axis_gate.py`, `template_family.py` and `h8_in_lung.py`
> default to `--out runs/nulls/<name>.json` — the *exploratory* artefacts that set
> every threshold in `PREREGISTRATION.md`. A confirmatory run that omits `--out`
> overwrites them. Always pass `--dir` **and** `--out`. The two scripts added for
> the paper have no default `--out` at all, on purpose.

## Setup

```bash
bash env/setup_env.sh   # venv in scratch; ~6 GB
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
| `slotmil/eval/estimands.py` | Prior-normalised skill, patient-specific skill, cluster bootstrap |
| `slotmil/eval/templates.py` | The content-free template family (in-plane / axial / separable / joint) |
| `slotmil/models/slot_attention.py` | Locatello slot attention + implicit differentiation + bag masking |
| `slotmil/models/baselines.py` | mean/max/ABMIL/Gated/MH-ABMIL/CLAM-SB/DSMIL/TransMIL/centre-Gaussian |
| `slotmil/data/feature_cache.py` | HDF5 bag dataset — the engineering asset the project rests on |
| `scripts/prereg_verdict.py` | Scores all ten hypotheses from stamped artefacts. Blinded |
| `scripts/make_paper_figures.py` | Figures + LaTeX tables, JSON in, no GPU |
| `paper/` | Manuscript and companion sources |

## Three implementation details that are load-bearing

**1. Padded instances must be zeroed after the softmax.** Slot attention
normalises over *slots*, so padding survives the softmax and then contributes
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
job. A second feature cache is a queued job, not a blocker.

## Storage

LIDC is 1,018 CT series / 243,958 slices / **~124 GB of DICOM** (measured via the
TCIA API), against a 200 GB scratch quota. `scripts/download_lidc_staged.py`
downloads in batches and deletes each batch's DICOM once features and masks are
cached, holding peak transient usage near 15 GB. It aborts if free space drops
below `--min-free-gb`, and skips series already cached so preemption costs one
batch rather than the run.

Steady state: LIDC features ~78 GB + MosMed ~17 GB + masks and run outputs
~10 GB.

## Historical data status (August 2026)

| Dataset | Status |
|---|---|
| LIDC-IDRI | **Cached.** 999 series, DINOv2 ViT-B/14, all 999 with lesion masks. Source DICOM deleted by the staged pipeline |
| MosMed | **Cached.** 1110 studies; only 50 carry masks, all CT-1 — which is why it supports one hypothesis |
| MedMNIST3D | Present. Public, auto-downloads. Splits verified 1158/165/310 |
| COV19D | Deferred; needs challenge access and a storage decision |
| RadImageNet DenseNet-121 | **Extraction in progress** (2026-08-22, job 7312418): `data/lidc/features_densenet121_radimagenet.h5`, DenseNet-121 @ 512 px → 16×16 grid, dim 1024, projected ~24 GiB. Weights at `data/weights/RadImageNet_pytorch/` |

## References and attribution

The bibliography is [`paper/refs.bib`](paper/refs.bib). Cite both the LIDC data DOI
and its database publication when using those data. Core method references are:

- [ABMIL / gated attention (Ilse et al., 2018)](https://proceedings.mlr.press/v80/ilse18a.html).
- [CLAM (Lu et al., 2021)](https://doi.org/10.1038/s41551-020-00682-w), [DSMIL (Li et al., 2021)](https://arxiv.org/abs/2011.08939), and [TransMIL (Shao et al., 2021)](https://arxiv.org/abs/2106.00908).
- [Slot Attention (Locatello et al., 2020)](https://arxiv.org/abs/2006.15055) and [implicit differentiation (Chang et al., 2022)](https://arxiv.org/abs/2207.00787).
- [DINOv2 (Oquab et al., 2024)](https://arxiv.org/abs/2304.07193) and [AdamW (Loshchilov and Hutter)](https://arxiv.org/abs/1711.05101).
- [Normal Guidance (Harvey et al., 2026)](https://arxiv.org/abs/2605.27306).
- [Saliency metric analysis (Kümmerer et al., 2015)](https://doi.org/10.1073/pnas.1510393112), [Borji et al., 2013](https://openaccess.thecvf.com/content_iccv_2013/html/Borji_Analysis_of_Scores_2013_ICCV_paper.html), and [Bylinskii et al., 2019](https://arxiv.org/abs/1604.03605).
- [Attention is not Explanation](https://aclanthology.org/N19-1357/), [Attention is not not Explanation](https://aclanthology.org/D19-1002/), and [medical saliency validation (Arun et al., 2021)](https://doi.org/10.1148/ryai.2021200267).

Related work includes [Slot-Mixup with Subsampling](https://arxiv.org/abs/2311.17466)
(which introduces the pathology model Slot-MIL) and
[INSIGHT](https://arxiv.org/abs/2412.02012). This repository's SlotMIL name should
not be read as claiming that earlier Slot-MIL method. Our focus is an evaluation
protocol for spatial confounding, rather than a new MIL pooling architecture.
