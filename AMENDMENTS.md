# Amendments to the pre-registration

Append-only. Newest entries at the bottom, so the file reads in the order things
actually happened.

Every entry must state **whether the affected results had already been seen**.
An amendment made before looking is a plan change; one made after is a
result-dependent choice, and the numbers it touches get labelled exploratory in
the paper. Recording that honestly costs a sentence. Not recording it costs the
whole mechanism.

Unblinding events are appended here automatically by
`scripts/prereg_unblind.py`.

Template:

```
## YYYY-MM-DD — short title

- **Kind:** amendment | unblinding | deviation
- **Config hash before → after:** `xxxx` → `yyyy`
- **What changed:**
- **Why:**
- **Results already seen?** yes / no — and which
- **Consequence for the paper:** e.g. "H5 now reported as exploratory"
```

---

## 2026-08-14 — pin the lung-mask method and the in-lung patch rule

- **Kind:** amendment
- **Config hash before → after:** `2b580fa93894d86f` → `20bdd93b781d950d`
- **What changed:** added `protocol.lung_mask` (`method: fill`,
  `lung_thresh: 0.0`) and replaced `estimands.secondary.in_lung_stratified_auc`'s
  `impl: 'pending WP4'` with a real reference. The original freeze declared that
  H8 depended on a lung mask but never said which one, so this closes a hole
  rather than changing a commitment.
- **Why:** the mask method and the "how much of a patch must be lung" cut are
  free parameters, and the first value tried for the latter — 0.5, chosen because
  it sounded neutral — turned out to delete 17% of annotated lesion patches. At a
  22 mm patch, a subpleural nodule sits in a patch that is part lung and part
  chest wall, and a majority rule discards exactly the lesions the control exists
  to keep. Measured on 30 series (26 lesion-bearing), sampled with seed 2027,
  **with the confirmatory test split excluded** (`--exclude-from
  data/lidc/splits_confirmatory.json --exclude-key test`):

  | method | containment @0.5 | @0.25 | @0.0 | lung frac @0.0 |
  |---|---|---|---|---|
  | air | 0.8151 | 0.9280 | **1.0000** | 0.24280 |
  | fill | 0.8167 | 0.9280 | **1.0000** | 0.24306 |
  | fill_close | 0.8200 | 0.9313 | **1.0000** | 0.24341 |
  | fill_hull | 0.8265 | 0.9607 | **1.0000** | 0.26893 |

  At threshold 0.0 every method reaches perfect containment with min-per-series
  1.000, so tightness decides. The stated rule picks `air`; it was overridden for
  `fill`, 0.00026 looser, because `air` is the deliberately unrepaired negative
  control and returns empty first and last slices. These 26 series happened to
  carry no lesion patch on an end slice; across 999 that would eventually delete
  positives. The override is recorded here rather than quietly applied.

- **Results already seen?** **No H8 outcome, and no result of any kind on
  confirmatory data.** What was seen: lesion-patch containment and lung-coverage
  diagnostics on 30 non-confirmatory-test series. No attention map, no AUC, no
  estimand was computed at any point during the selection.
- **Consequence for the paper:** none — H8 remains confirmatory. The selection
  run itself is labelled exploratory in `runs/lung_mask_sweep.json`
  (`analysis_role: exploratory`).

---

## 2026-08-15 — implement the two Harvey arms; rule what the freeze left open

- **Kind:** amendment
- **Config hash before → after:** `20bdd93b781d950d` → `33a55222c853a4cf`
- **What changed:** promoted `centre_gaussian` and `normal_guidance` from
  `planned` to `implemented`, and declared the choices each requires that the
  original freeze did not specify. Also declared `protocol.training.max_slices`,
  which was undeclared entirely, and ruled the scope of H2's "every trained arm"
  before any H2 number exists.

  **`centre_gaussian` — support, sigma, mu.** Harvey print
  `a_ij ∝ NormPDF(j | S_i/2, 1)` over the *raw* slice index. That is not
  computable on volumetric CT. The tail logit is −3612 nats at S=171 and −61075
  at S=700, against −103.3 for float32's smallest subnormal and −744.4 for
  float64's. The logits themselves survive — fp32 keeps all 86 distinct values —
  but the arm returns *normalised* attention, and the softmax collapses 142 of
  171 slices to exactly 0.0 (94 of 171 even in float64). That is a tie block over
  83% of the bag, and `roc_auc_score` credits ties at 0.5, so a literal
  implementation would report a floating-point artefact as Harvey's baseline.

  Declared instead: support `z = linspace(-1, 1, S_i)` over the normalised slice
  index, `sigma_z = 0.25`, `mu = (S-1)/2`.

  **`normal_guidance` — base arm, KL direction, moment source, variance floor.**
  The freeze said only "KL to a moment-matched Normal over slice index,
  recomputed under stop-gradient each step. Carries H6." It named neither the
  base arm that H6 compares against, nor the direction of the KL, nor what the
  Normal is matched to, nor any floor. Declared: `base_arm: gated_abmil`,
  `KL(attention_slice_marginal || moment_matched_normal)`,
  `moment_source: own_marginal_detached`, `var_floor_slices2: 1.0`,
  `moment_scope: per_bag_per_head`.

  **`protocol.training.max_slices: 48`.** Undeclared until now — it lived only as
  a `scripts/train_cached.py` argparse default.

  **H2 scope.** "Every trained arm" now means arms whose *attention* has learned
  parameters, explicitly excluding `centre_gaussian`.

- **Why:**

  *sigma.* Because sigma cannot move a reported number, the departure costs
  nothing. `NormPDF(·|mu,sigma)` is strictly monotone in `−|j−mu|` for every
  sigma > 0 and every estimand here is rank-based; measured, the slice AUC is
  identical to 12 decimal places across `sigma_z ∈ [0.25, 1.0]`. Sigma is
  therefore fixed on representability alone: 0.25 is the midpoint of the band
  where fp32 keeps the tie structure intact and fp16 still resolves every
  distinct value — the peakiest, most Harvey-like sigma that survives AMP. Below
  roughly 0.227 the fp32 tails lose precision and the AUC moves in the third
  decimal. Pinned by `tests/test_position_arms.py::TestSigma`.

  *mu.* `(S-1)/2` rather than Harvey's `S/2`: a half-slice offset has no
  anatomical basis and, for even S, breaks what should be a symmetric tie between
  slices `S/2 − 1` and `S/2`. It is also what makes the arm exactly rank-equal to
  the axial component of `nulls.centre_prior_scores`, which is a testable claim
  rather than a resemblance.

  *base arm.* `gated_abmil` has exactly one attention distribution over
  instances, so the slice marginal needs no additional convention; it carries no
  other regulariser, so H6's delta is attributable to the KL alone; and it is the
  study's strongest classifier, so a localisation gain cannot be dismissed as
  rescuing a weak arm. `slot:div=0.5` was rejected because it normalises over the
  slot axis (the marginal would need an undeclared renormalisation) and already
  carries a second regulariser; `mh_abmil` because its eight heads raise a second
  undeclared question and it already serves as the parameter-matched control.
  Implemented as a registry *alias*, so NG and its base are the same class and
  `lam=0` reproduces the base bit-for-bit at the same seed.

  *KL direction.* `KL(a || q)` decomposes as `−H(a) + CE(a, q)`, continuous with
  the entropy regulariser already in `losses.py`. The reverse diverges wherever a
  slice is starved and would scale with the model's peakedness rather than with
  prior mismatch.

  *moment source.* Matching the marginal's own moments under stop-gradient is the
  only reading in which "recomputed under stop-gradient each step" means
  anything: matching a fixed geometric centre leaves nothing to recompute, since
  only S varies and S carries no gradient. It also makes the term a Gaussian
  *shape* prior — unimodal, but silent about *where* — so it is structurally
  incapable of injecting information about the individual patient. That is what
  keeps H6 falsifiable rather than tautological.

  *variance floor, and it is the substantive addition here.* Unfloored, a
  single-slice attention is a **global minimum of the KL at exactly 0**, because
  a Dirac is the sigma→0 Normal. The term would therefore reward collapsing
  attention onto one slice, and the collapse would present as a spectacular
  localisation result. Floored at 1 slice² a delta pays 0.9189 nats while a true
  Gaussian pays 0. Harvey's sigma = 1 survives here, in the one place a raw-slice
  sigma is both meaningful and representable. Measured on S=48:

  | marginal | KL unfloored | KL floored at 1 |
  |---|---|---|
  | delta (1 slice) | **0.0000** | 0.9189 |
  | gaussian sigma=1 | 0.0000 | 0.0000 |
  | uniform | 0.0895 | 0.0895 |
  | bimodal | 1.0624 | 1.0624 |

  Bimodal pays most, so NG suppresses multi-focal attention. LIDC has
  multi-nodule cases: that is a real cost of the method and is reported as one.

  *max_slices.* Declared because `normal_guidance` made it material. Under
  subsampling, bag position *k* is true slice `sel[k]`, so the KL's 1-slice²
  floor would be applied in subsampled units — a random per-bag multiple of the
  anatomical scale, roughly 213× in variance for a 700-slice volume cut to 48.
  `collate_bags` therefore now carries the true `slice_index` through to the
  loss. Note the KL is invariant to affine rescaling of the coordinate (the
  target's moments rescale with it), so `slice_index` matters through exactly one
  channel — the variance floor — which is why it is routed to the loss and not to
  the model.

  *H2 scope.* `centre_gaussian` ranks slices by `−|z|` exactly as the
  `centre_prior` reference does, so it will **tie** that reference rather than be
  beaten by it — and a tie is not "exceeds". Without this ruling, an arm that *is*
  a centre prior would falsify a hypothesis about centre priors. Ruled now
  precisely so it cannot be decided once the numbers are in.

- **Results already seen?** **No.** No arm has been trained, no estimand
  computed, no attention dumped. Every number in this entry is either arithmetic
  on the floating-point format (the sigma argument) or a closed-form property of
  the KL evaluated on synthetic marginals (the floor table) — neither touches
  LIDC data of any split.

  A `lam` calibration pre-flight (`scripts/slurm/ng_lambda_preflight.sbatch`) was
  submitted before this entry was written and had not reported. It runs on the
  **discovery** split, which `PREREGISTRATION.md` already labels exploratory by
  construction, and its permitted scope is `loss_kl_prior` and `loss_bag`
  magnitudes — no localisation estimand, no attention dump, no test-split number,
  no comparison against the base arm. `lam` is **not** changed by this entry: it
  stands at the pre-registered 0.1. If that pre-flight shows the term inert and
  `lam` is subsequently amended, it gets its own entry disclosing exactly what
  was seen.
- **Consequence for the paper:** none. H6 and H2 remain confirmatory. The
  departure from Harvey's printed sigma must be stated in the methods section,
  together with the reason it cannot affect a rank-based metric.
