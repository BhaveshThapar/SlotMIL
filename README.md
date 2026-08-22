# SlotMIL

**Position priors dominate instance-level localisation metrics for
weakly-supervised multiple-instance learning in volumetric CT.**

A pre-registered evaluation of what instance-level localisation AUC actually
measures in volumetric CT. Reports the extent to which such metrics are dominated
by an anatomical position prior, decomposes that prior into its axial and
in-plane parts, and proposes prior-normalised estimands validated against a
supervised patch ceiling. Every claim is gated on a hypothesis frozen before the
confirmatory runs; the analysis plan and its amendment chain ship with the code.

The project began as a method paper — slot attention as an interpretable MIL
pooling bottleneck. That hypothesis was falsified three times over, and the
failures became the paper. [`plan.md`](plan.md) is the pre-pivot research
dossier and is kept as history, not as a description of the current work.

## The claim

Instance-level localisation metrics for weakly-supervised MIL in volumetric CT
largely measure in-plane position. On LIDC-IDRI, across ten MIL architectures and
five seeds:

- flat (3D) instance AUC does not separate from within-slice AUC by more than
  0.02 for seven of eight scored arms;
- no trained arm reaches a model-free centre prior on the axial axis;
- the 95th percentile over 30 **untrained** initialisations is 0.7666, so 0.5 is
  not the floor;
- a content-free in-plane template fit on validation reaches 0.8543 — above every
  trained arm;
- scoring an arm's attention against a *different* patient's masks beats scoring
  it against the right ones for five of eight arms.

The one exception is the second finding: Harvey-style Normal Guidance raises
patient-specific skill by +0.1509 against a pre-registered falsifier of 0.02, and
is independently the only arm the axis gate flags as carrying axial content.
Explicit prior injection, not architecture, is what adds measurable localisation.

Two papers sit adjacent and are differentiated in the manuscript: **Slot-MIL**
(arXiv:2311.17466) for pathology WSI and **INSIGHT** (arXiv:2412.02012) for
non-slot CT MIL. The prior-normalised estimands are imported, not invented —
Kümmerer et al. (PNAS 2015) and Borji/Bylinskii — and the content-free-baseline
observation is not ours first (Harvey/Loevlie, arXiv:2604.26807 and
arXiv:2605.27306). What is ours is the axial/in-plane decomposition against
voxel masks, the per-condition power gate, and the per-dataset measurement.

Full write-up: [`RESULTS.md`](RESULTS.md) from the "Confirmatory results"
heading. Manuscript sources: [`paper/`](paper/).

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

Two of the ten hypotheses failed, and both are reported as failures: H6 (prior
injection adds real information — the critique fails for that arm) and H9 (the
stereotypy prediction, reversed with power to spare).

## Reproducing the paper

Every analysis driver is a standalone CLI that reads cached attention dumps,
needs **no GPU**, and writes a stamped artefact. The scorer recomputes nothing —
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

## Data status

| Dataset | Status |
|---|---|
| LIDC-IDRI | **Cached.** 999 series, DINOv2 ViT-B/14, all 999 with lesion masks. Source DICOM deleted by the staged pipeline |
| MosMed | **Cached.** 1110 studies; only 50 carry masks, all CT-1 — which is why it supports one hypothesis |
| MedMNIST3D | Present. Public, auto-downloads. Splits verified 1158/165/310 |
| COV19D | Deferred; needs challenge access and a storage decision |
| RadImageNet DenseNet-121 | **Extraction in progress** (2026-08-22, job 7312418): `data/lidc/features_densenet121_radimagenet.h5`, DenseNet-121 @ 512 px → 16×16 grid, dim 1024, projected ~24 GiB. Weights at `data/weights/RadImageNet_pytorch/` |
