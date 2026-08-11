# Results

## W1 go/no-go — NoduleMNIST3D (2026-08-06)

Job 7214608, tron/RTX A5000. 5 arms × 3 seeds × 30 epochs, official MedMNIST
splits 1158/165/310. End-to-end fast path (small 2D CNN encoder + MIL pooling),
K=4, T=3, implicit differentiation, gated readout.

| arm | test AUC | test ACC | params | active slots | max slot cos |
|---|---|---|---|---|---|
| mean pool | 0.8165 ± 0.0019 | 0.8333 | 143k | — | — |
| gated ABMIL | 0.8586 ± 0.0087 | 0.8355 | 176k | — | — |
| multi-head ABMIL (matched) | 0.8471 ± 0.0057 | 0.8419 | 408k | — | — |
| SlotMIL | 0.8273 ± 0.0225 | 0.8387 | 408k | 3.26 / 4 | **0.997** |
| **SlotMIL + diversity 0.1** | **0.8613 ± 0.0163** | **0.8495** | 408k | 2.00 / 4 | **0.139** |

Head-to-head, Welch t-test over seeds:

| comparison | Δ AUC | p | verdict |
|---|---|---|---|
| vs mean pool | +0.0448 | 0.040 | **PASS** |
| vs gated ABMIL | +0.0027 | 0.816 | FAIL — noise |
| vs multi-head ABMIL | +0.0142 | 0.268 | FAIL — not significant |

### Read

**The diversity regulariser is not optional.** Without it slots collapse
outright: max pairwise slot cosine 0.997, i.e. all four slots converge to
essentially the same vector. With it, 0.139. This is the failure mode plan.md
line 104 predicted, it appeared immediately, and the pre-committed mitigation
fixed it. The collapse also cost accuracy (0.8273 → 0.8613), so diversity is
buying both interpretability and performance here.

**Accuracy is neutral, not better.** SlotMIL+diversity edges gated ABMIL by
0.0027 AUC at p=0.82. That is not a win, and it should not be reported as one.
It lands exactly on plan.md line 57's pre-committed fallback: *"if neutral, pivot
to the interpretability claim"*. Nothing about W1 undermines the paper — the
interpretability claim was always the moat, given Slot-MIL and INSIGHT — but the
accuracy story is parity, and the ISBI framing should say so.

**The 0.879 reference is not cleared, and is not the right comparison.** The
MedMNIST bar comes from ResNet-18(**3D**) trained on volumes. This arm is a small
2D CNN encoder feeding MIL pooling, chosen for a minutes-long W1 iteration cycle,
not to compete with a tuned 3D CNN. The plan's actual W1 gate — "SlotMIL trains
and beats mean-pooling" — passes significantly. Clearing 0.879 is a job for the
DINOv2-cached path.

**Only 2 of 4 slots stay active** under diversity 0.1. Either K=4 is more than a
binary benign/malignant task needs, or the diversity weight is over-tuned toward
suppression. This is a W4–5 question and feeds directly into the K ablation.

### Caveats

- 3 seeds, not the ≥5 that plan.md line 116 requires for the paper. W1 is a
  go/no-go, not a results table.
- The fast path uses a trainable CNN encoder, so this does not yet validate the
  frozen-feature cache end to end.

## MosMed data audit (2026-08-06) — changes how it should be used

Downloaded and verified: 1,110 volumes with class counts **exactly** matching
Morozov et al. (CT-0 254 / CT-1 684 / CT-2 125 / CT-3 45 / CT-4 2), plus all 50
expert masks, every mask key matching a volume key. Volumes are 38 slices at
512×512, confirming the every-10th-slice public release.

Two properties of the masked subset weaken the experiment plan.md line 64
envisages ("ground-glass and consolidation — the two canonical COVID findings,
perfect slot targets"):

**1. Consolidation is a sliver of the annotated lesion.** Splitting the binary
masks by HU across all 50 scans:

| | ground glass | consolidation |
|---|---|---|
| total voxels | 978,988 | 46,945 |
| median per scan | 16,990 | 264 |

Consolidation is a median **2.6%** of lesion volume; **18/50** scans have under
1% and **35/50** under 5%. Asking a slot to specialise on a finding that is 2.6%
of the signal is a far weaker test than a balanced two-finding split would be.
Compounding this, the public masks are a *single binary channel* — the
GGO/consolidation split is our own HU threshold, a convention rather than
ground truth, so both the imbalance and the boundary are approximations.

**2. All 50 masked scans are CT-1.** The annotated subset has no severity
variation, so slot behaviour across severity cannot be studied with masks; the
severity-classification target and the localisation target sit on disjoint parts
of the dataset.

**Consequence.** LIDC-IDRI should carry the headline alignment experiment — it
has 4-radiologist consensus masks, a genuine per-nodule finding structure, and no
threshold-invented finding classes. MosMed becomes a supporting second dataset
demonstrating the method transfers, and the paper should state the HU-split
caveat rather than present GGO/consolidation as annotated classes. This does not
cost the ISBI story (plan.md line 142 already scopes ISBI to one localisation
dataset), but it does mean MosMed cannot be the backup if LIDC disappoints.

## MosMed results (2026-08-08) — negative, and informative

Job 7215380, 6 arms × 3 seeds × 40 epochs, 4-class severity, DINOv2 ViT-B/14
cached features, K=8, patch-level bags.

| arm | test AUC | test ACC | active slots | max slot cos |
|---|---|---|---|---|
| mean pool | 0.7516 ± 0.0031 | 0.6961 | — | — |
| gated ABMIL | **0.8037 ± 0.0366** | 0.7238 | — | — |
| multi-head ABMIL (matched) | 0.7882 ± 0.0534 | 0.6906 | — | — |
| SlotMIL | 0.7758 ± 0.0463 | 0.6317 | 5.91 / 8 | **0.997** |
| SlotMIL + div 0.1 | 0.7485 ± 0.0678 | 0.6703 | 4.54 / 8 | **0.976** |
| SlotMIL + div 0.5 | 0.7063 ± 0.0231 | 0.6243 | 4.38 / 8 | **0.735** |

No pairwise difference is significant (all p ≥ 0.10). Best slot arm vs gated
ABMIL: −0.0279, p = 0.46.

**Slots collapse at every diversity weight tested.** The W1 setting that worked
at K=4 on MedMNIST (0.997 → 0.139) does not transfer to K=8 on real DINOv2
features: 0.1 barely moves it (0.976) and even 0.5 only reaches 0.735, while
costing 0.07 AUC. So the diversity weight is not a constant — it interacts with K
and with feature statistics, and needs a proper sweep rather than a single value
carried over from the toy dataset.

### Alignment (fit on 28 annotated val scans, frozen, scored on 22 test)

| metric | SlotMIL (div 0.5) | multi-head ABMIL | chance |
|---|---|---|---|
| assigned affinity | 0.5113 | 0.5032 | 0.5000 |
| lift over chance | **1.02×** | 1.01× | 1.0× |
| slot purity / NMI | 0.988 / **0.015** | 0.988 / 0.004 | — |
| best-slot Dice | **0.044 ± 0.041** | 0.007 ± 0.017 | — |
| pointing game | **0.000** | 0.000 | — |
| instance AUC | **0.601** | 0.332 | 0.5 |
| head redundancy (↓) | **0.073** | 0.221 | — |

**Read this as a clear negative on localisation.** Pointing game 0.000 means that
across 22 annotated test scans, the most-attended patch never once landed inside
a lesion. Affinity lift of 1.02× is chance. Dice 0.044 is nil.

Two things are nonetheless real and worth keeping:
- SlotMIL beats the matched control on every localisation metric (Dice 6×,
  instance AUC 0.601 vs 0.332 — the control is *below* chance), and its slots are
  markedly less redundant (0.073 vs 0.221). So the competitive mechanism is doing
  something the extra capacity alone does not.
- Instance AUC 0.601 is weakly above chance, i.e. there is signal, just far too
  little to call localisation.

### Two metric caveats that make these numbers look better than they are

**Purity 0.988 is an artefact.** With findings encoded as lesion-vs-background,
background is ~99% of patches, so a slot that attends nothing in particular is
"pure" by default. NMI (0.015) is the honest number and it is ~0.

**Consistency 1.000 is the same artefact.** Every slot's dominant finding is
trivially "background". Both metrics need re-specifying over annotated patches
only before they mean anything; as computed here they should not be reported.

### Why this is not yet a verdict on the method

MosMed was already downgraded to the *supporting* dataset by the data audit
above: consolidation is 2.6% of annotated lesion volume, the GGO/consolidation
split is our own HU threshold rather than an annotated class, and all 50 masked
scans are CT-1. Add that the task is *severity*, which plausibly does not require
attending to individual lesions at all — global lung appearance may suffice — and
a null localisation result is close to what a careful prior would predict.

LIDC is the real test: 4-radiologist consensus masks, a genuine per-nodule
finding structure, and a task (nodule presence) that cannot be solved without
looking at nodules.

## LIDC results (2026-08-11) — the decisive experiment, and it is negative

Jobs 7230368 (6 arms × 3 seeds × 40 epochs) and 7230369 (analysis). 999 series,
patient-grouped splits 702/149/148, label `nodule_present`, DINOv2 ViT-B/14
cached features, K=8, patch-level bags.

### Classification — nothing separates

| arm | test AUC | test ACC | active slots | max slot cos |
|---|---|---|---|---|
| **gated ABMIL** | **0.8512 ± 0.0071** | 0.8851 | — | — |
| mean pool | 0.8324 ± 0.0119 | 0.8851 | — | — |
| SlotMIL + div 0.5 | 0.8305 ± 0.0356 | **0.9009** | 5.28 / 8 | **0.345** |
| SlotMIL | 0.8180 ± 0.0509 | 0.8919 | 7.05 / 8 | 0.992 |
| multi-head ABMIL | 0.8074 ± 0.0525 | 0.8784 | — | — |
| SlotMIL + div 0.1 | 0.7555 ± 0.1030 | 0.8829 | 6.37 / 8 | 0.974 |

Best slot arm vs every other arm: **no comparison reaches significance**
(p = 0.42, 0.94, 0.57, 0.75, 0.34). SlotMIL is nominally *behind* gated ABMIL.

### Localisation — at chance

| metric | SlotMIL (div 0.5) | multi-head ABMIL | chance |
|---|---|---|---|
| affinity lift | 1.00× | 1.00× | 1.0× |
| best-slot Dice | 0.006 ± 0.007 | 0.002 ± 0.004 | — |
| pointing game | 0.008 | 0.000 | ~0.0005 |
| **instance AUC** | **0.489** | 0.295 | 0.5 |
| head redundancy (↓) | **0.041** | 0.927 | — |

Faithfulness is equally flat: deletion AUC 0.8566, insertion AUC 0.8617,
difference **+0.005**. The attention does not drive the prediction.

### What actually explains it — a resolution hypothesis, tested and refuted

The obvious suspect was spatial resolution. At 224 px input with patch 14, one
patch token covers **22.0 mm** (verified against the cache's own median pixel
spacing of 0.687 mm). A 5 mm nodule occupies 0.05 of a patch; a 10 mm nodule
0.21; even a 20 mm nodule fits inside a single patch. Nodule-positive patches are
0.05% of the grid. It looked like the representation simply could not express the
target.

**That hypothesis is wrong.** A supervised logistic probe trained directly on
patch labels, using the *identical* 224 px DINOv2 features and patient-disjoint
splits, separates nodule from non-nodule patches at:

> **patch-level nodule AUC = 0.9102**

So the features do encode nodules at this resolution. The ceiling for any pooling
method on this cache is ~0.91. SlotMIL's unsupervised instance AUC is **0.489**.

**The information is present and weakly-supervised slot attention does not find
any of it.** That is a far cleaner negative than "the resolution was too coarse",
and it is the experiment that makes the result publishable rather than merely
disappointing.

### What survives

- **Diversity reliably prevents collapse**: max slot cosine 0.345 with div 0.5 vs
  0.992 without. It just buys no accuracy and no localisation.
- **Slot competition genuinely differs from multi-head attention**: head
  redundancy 0.041 vs 0.927 for the parameter-matched control. The slots *are*
  distinct — they simply bind nothing anatomically meaningful. This answers
  reviewer objection #1 mechanistically while undercutting the headline claim.
- Collapse does **not** predict accuracy (r = −0.49, p = 0.27 over 7 slot runs);
  bare `slot` seed 1 scored 0.8625 while fully collapsed at cosine 0.993.

### Two live confounds, in priority order

1. **The bag label is nearly uninformative.** `nodule_present` is 87% positive
   (610 of 702 training bags). A label that almost every bag shares provides
   almost no gradient pressure to locate anything. The `malignancy` label
   (median-of-four-readers, indeterminates dropped) is balanced and cannot be
   predicted without characterising the nodule — `scan_label(mode="malignancy")`
   is already implemented. **This is the next experiment and it is cheap.**
2. **K=8 may be wrong for ~2 nodules per scan** (median 2). The K sweep on
   validation is still outstanding.

Until (1) is run, the honest claim is *"weakly-supervised slot attention fails to
localise under a near-degenerate bag label"*, not *"slot attention cannot
localise"*.

## LIDC malignancy (2026-08-11) — the confound test, and it is inconclusive

Job 7237846/7237847. Relabelled from `nodule_present` (87% positive) to malignancy
via the standard protocol: median of four readers per nodule, median==3 dropped as
indeterminate. Result: 734 of 999 scans, **50.4% positive**, test split 55/55.
Same cached features — only the label changed.

| arm | test AUC | test ACC | max slot cos |
|---|---|---|---|
| **mean pool** | **0.5949 ± 0.0100** | 0.5636 | — |
| gated ABMIL | 0.5803 ± 0.0088 | 0.5303 | — |
| multi-head ABMIL | 0.5709 ± 0.0210 | 0.5606 | — |
| SlotMIL + div 0.5 | 0.5645 ± 0.0082 | 0.5667 | 0.718 |

**Every method is near chance**, and SlotMIL is *significantly worse than mean
pooling* (Δ = −0.0304, p = 0.016).

### Why this does not answer the question

The malignancy probe ceiling explains it. Running the same supervised control,
now discriminating malignant from benign **nodule tissue**:

| task | supervised probe ceiling | best MIL arm |
|---|---|---|
| nodule presence | **0.9102** | — |
| **malignancy** | **0.6516** | 0.5949 (mean pool) |

At 22 mm per patch a nodule is sub-patch: its *presence* still perturbs patch
statistics enough to be detected (0.91), but its *margin, spiculation and
texture* — the features that determine malignancy — are averaged away (0.65).
The MIL arms reach 0.595 of a 0.652 ceiling, i.e. ~91% of what is achievable.
They are not failing to learn; there is almost nothing there to learn.

**So the experiment swapped one confound for another.** `nodule_present` had a
degenerate label but a supported task; `malignancy` has a balanced label but an
unsupported task. Neither isolates the question.

### What it did show

- Localisation moved off the floor: instance AUC **0.489 → 0.578**. But pointing
  game is still 0.000, Dice 0.002, affinity lift 1.00×, and the
  parameter-matched control scores *higher* (0.609) than SlotMIL.
- Faithfulness went **negative**: insertion − deletion = **−0.0216**. The
  attention is worse than useless as an explanation of the prediction.
- The redundancy dissociation persists: 0.145 (SlotMIL) vs 0.893 (multi-head
  ABMIL).

### The clean experiment that remains

Balanced label **and** a task the representation supports: subsample
nodule-presence to the 131 no-nodule scans plus 131 with-nodule scans (262 bags,
50/50). Small, but it is the only condition where the label is balanced and the
supervised ceiling is known to be 0.91. If SlotMIL still fails to localise there,
the confound defence is exhausted.

## Implementation findings

**Implicit differentiation freezes "learnable" slot queries.** With
`implicit=True, init="learnable", bo_qsa_straight_through=False` — which was the
natural default — `slots_query` sits inside the `no_grad` region and is then
detached, so it receives no gradient and stays at its random init for the entire
run. Faithful to Chang et al. (the fixed point is meant to be init-independent),
but it would have silently voided the random-vs-learnable init ablation
(plan.md line 114): both arms would have been frozen random inits, and the
conclusion "init doesn't matter" would have been an artefact. The constructor now
warns, and `TestQueryGradientFlow` pins the behaviour.

**Implicit mode must run `iters-1` no-grad steps, not `iters`.** The obvious
implementation gives implicit mode T+1 refinements against vanilla's T,
confounding the planned vanilla-vs-implicit ablation with a difference in
effective depth rather than gradient estimation.
