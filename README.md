# SlotMIL

Slot attention as an interpretable MIL pooling bottleneck for volumetric CT.

The research dossier — novelty analysis, related work, experimental design, risk
register — is [`plan.md`](plan.md). This README covers the implementation only.

## The claim, and what defends it

Iterative competitive slot attention (softmax-over-slots + GRU) as the pooling
layer in attention-based MIL for CT, with slot attention masks validated against
public lesion annotations.

Two papers sit close and must be differentiated: **Slot-MIL** (arXiv:2311.17466)
already claimed slot attention as MIL pooling for pathology WSI, and **INSIGHT**
(arXiv:2412.02012) already claimed interpretable CT MIL aggregation by a
non-slot mechanism. The defensible contribution is therefore the *mask-validated
slot-to-finding alignment* — the experiment neither of them ran — implemented in
[`slotmil/eval/alignment.py`](slotmil/eval/alignment.py) and driven by
[`scripts/eval_alignment.py`](scripts/eval_alignment.py).

## Setup

```bash
bash env/setup_env.sh          # venv in scratch; ~6 GB
.venv/bin/python -m pytest -q  # 40 tests
```

Environment is pinned around two real constraints: `numpy<2` and
`sqlalchemy<2` because `pylidc` 0.2.3 predates both breaks, and `setuptools<81`
because `pylidc` imports `pkg_resources` at module load.

## Layout

| Path | What |
|---|---|
| `slotmil/models/slot_attention.py` | The core. Locatello slot attention + implicit differentiation + bag masking |
| `slotmil/models/baselines.py` | mean/max/ABMIL/Gated-ABMIL/**MultiHeadABMIL** (the matched control) |
| `slotmil/models/heads.py` | Four readouts, all exposing per-slot attribution |
| `slotmil/losses.py` | Bag CE/BCE + slot diversity + attention entropy + optional DINOSAUR recon |
| `slotmil/features/` | Frozen DINOv2/DenseNet extraction, CT windowing, lung-slice selection |
| `slotmil/data/feature_cache.py` | HDF5 bag dataset — the engineering asset the project rests on |
| `slotmil/eval/alignment.py` | Slot→finding Hungarian assignment, purity/NMI, consistency, redundancy |
| `scripts/download_lidc_staged.py` | Staged LIDC download that fits in the disk budget |

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
refinements against vanilla's T, which would confound the planned
vanilla-vs-implicit ablation with a difference in effective depth. Tested by
`test_implicit_and_vanilla_agree_on_forward`.

**3. `pylidc` needs the DICOM on disk to build a mask.** Placing annotation
contours into a volume requires per-slice z positions from the DICOM headers,
which are not in pylidc's bundled annotation database. The staged pipeline must
therefore extract masks *before* deleting each batch. Getting this order wrong
costs a 124 GB re-download.

## Storage

LIDC is 1,018 CT series / 243,958 slices / **~124 GB of DICOM** (measured via the
TCIA API), against ~150 GB of scratch. `scripts/download_lidc_staged.py`
downloads in batches and deletes each batch's DICOM once features and masks are
cached, holding peak transient usage near 15 GB. It aborts if free space drops
below `--min-free-gb`, and skips series already cached so preemption costs one
batch rather than the run.

Steady state: LIDC features ~55 GB + MosMed ~17 GB + MedMNIST3D ~5 GB + masks and
run outputs ~10 GB ≈ 87 GB.

## Running

```bash
# W1 go/no-go on NoduleMNIST3D (reference bar: 0.879 AUC, ResNet-18(3D))
sbatch scripts/slurm/train.sbatch scripts/w1_gonogo.py \
    --poolings mean gated_abmil mh_abmil slot slot:div=0.1 --seeds 0 1 2

# Staged LIDC download + feature cache
sbatch scripts/slurm/extract_lidc.sbatch --batch-series 25

# The money experiment
.venv/bin/python scripts/eval_alignment.py --cache <h5> --checkpoint <pt> --splits <json>
```

Arms accept inline overrides: `slot:div=0.1,K=8,iters=5`.

## Data status

| Dataset | Status |
|---|---|
| MedMNIST3D | **Present.** Public, auto-downloads. Splits verified 1158/165/310 |
| LIDC-IDRI | Reachable without auth via the TCIA NBIA API; pipeline ready |
| MosMed | **Needs credentials.** No unauthenticated route exists — HuggingFace, Zenodo and mosmed.ai direct links were all checked and are absent/404. Put a Kaggle token at `~/.kaggle/kaggle.json` and run `scripts/download_mosmed.py`, or download manually from mosmed.ai. Licence CC BY-NC-ND 3.0 |
| COV19D | Deferred to W10; needs challenge access and a storage decision |
| RadImageNet DenseNet-121 | Backbone ablation only; gated behind a Google Drive request form |
