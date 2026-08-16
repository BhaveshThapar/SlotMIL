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

---

## 2026-08-15 — lam pre-flight reported; lam unchanged; a scope breach recorded

- **Kind:** deviation (record only — no config change)
- **Config hash before → after:** `33a55222c853a4cf` → `33a55222c853a4cf` (unchanged)
- **What changed:** nothing. `lam` stands at the pre-registered 0.1. This entry
  exists because the calibration pre-flight promised in the previous entry has
  now reported, and because it showed me more than I said it would.
- **What the pre-flight showed:** on the discovery split, seed 0, 8 epochs, the
  KL term is **not inert**. `train_loss_kl_prior` falls 0.0812 → 0.0387 (−52.3%)
  with validation tracking it (0.0707 → 0.0379). Its contribution to the total
  loss is small — `lam × KL` of 0.0081 → 0.0039, 1.2–2.0% — which matches the
  ~0.089-nat estimate for a near-uniform marginal that motivated the pre-flight.
  Small is not inert, and only inert would have justified changing `lam`.

  What this does **not** establish is attribution: whether the KL drove the drop
  or merely tracked a drift `gated_abmil` would have shown anyway. Settling that
  requires the base arm's KL at `lam=0`, which is the H6 comparison itself and
  belongs in the confirmatory sweep, not here.
- **The scope breach.** `scripts/slurm/ng_lambda_preflight.sbatch` declared the
  permitted scope as `loss_kl_prior` and `loss_bag` magnitudes, explicitly
  excluding "any test-split number". But `scripts/train_cached.py` prints
  `res["test"]["auc"]` unconditionally on arm completion, so two discovery-split
  test AUCs were emitted to the job log and seen: `centre_gaussian` 0.8311 and
  `normal_guidance:lam=0.1` 0.8511 (accuracy 0.8716 for both). A declared scope
  that the tooling does not enforce is a comment, not a control; recorded here
  rather than passed over.
- **Results already seen?** **Yes, and the above is the complete list.** Both
  numbers are on the **discovery** split, which this pre-registration labels
  exploratory by construction and which cannot carry confirmatory claims, so no
  confirmatory result was seen and none was influenced — `lam` is unchanged, and
  no other parameter was touched after seeing them. No localisation estimand was
  computed, no attention was dumped, and nothing on `splits_confirmatory.json`
  was read.
- **Consequence for the paper:** none. H6 remains confirmatory. Future
  calibration runs should suppress test metrics at the source rather than declare
  a scope the driver ignores.

---

## 2026-08-15 — the slice subsampler was correlated; fixed before the sweep

- **Kind:** deviation (record only — no config change)
- **Config hash before → after:** `33a55222c853a4cf` → `33a55222c853a4cf` (unchanged)
- **What changed:** no declared parameter. `FeatureBagDataset` held its
  subsampling RNG as a `np.random.Generator` built in `__init__`; it now derives
  the draw per item from `(seed, epoch, index)`, and `train.fit` calls
  `set_epoch` before each epoch's DataLoader iterator is created. This entry
  exists because the old form did not do what `protocol.training.max_slices`
  assumes it does, and because the fix changes what every future training run
  sees.

  **The defect.** `DataLoader` forks the Generator to every worker with
  identical state, and `utils.seed.seed_worker` reseeds only the legacy
  `np.random` global and `random` — it cannot reach a Generator object.
  `seed_worker`'s own docstring claimed it stopped subsampling correlating
  across workers. It never could. Three consequences ran at once:

  1. *Across workers.* Four workers drew from identical state, so bags at the
     same queue position received the same slice subset.
  2. *Across epochs.* Nothing in the repo sets `persistent_workers`, so workers
     are re-forked each epoch from a parent whose Generator never advances — the
     parent process never calls `__getitem__`. The streams reset every epoch.
     "Stochastic slice subsampling during training" resampled nothing.
  3. *Across seeds and arms.* `scripts/train_cached.py` never passed `seed`, so
     every arm at every training seed used the default stream 0.

  Affects only bags with `n_slices > 48` under `train=True`, which on LIDC is
  the majority (212,173 slices over 999 series). Evaluation is unaffected:
  `max_slices_applies_to: train_only` is already declared, val and test are
  constructed with `max_slices=None`, and every reported test number was
  computed on full bags.

- **Why it is recorded here rather than as a bug fix in a commit message:** it
  changes the *training distribution*, not just RNG hygiene. Under the old form a
  model saw approximately one fixed 48-slice view of each long volume for the
  whole run; it now sees a fresh view each epoch. Confirmatory arms will
  therefore not be reproducible from discovery checkpoints, and the numbers move
  for a reason that has nothing to do with any arm.

  **The prediction, stated before the sweep runs:** reported seed-to-seed std
  should come out **wider** on confirmatory than on discovery. Discovery
  variance covered initialisation and shuffle order but not the data view,
  because all five seeds shared stream 0. A wider std is the bug being removed,
  not a change in method, and must not be read as one after the fact.

- **Results already seen?** **No confirmatory result of any kind.** What was
  seen, and the complete list: the discovery-split results already published in
  `RESULTS.md`, all of which predate this entry and are labelled exploratory by
  construction; and `runs/untrained_floor{,_mh}.json`, which are untrained and
  built without `max_slices`, so the subsampler never ran for them and H3 is
  untouched. No arm was trained, no estimand computed and no attention dumped
  between finding the defect and fixing it. The fix was not chosen against any
  outcome — there is only one correct behaviour here and it is the one declared.

- **Consequence for the paper:** the 40 discovery arm/seed numbers in
  `RESULTS.md` stay valid *as reported* but are no longer bit-reproducible under
  the current code. The methods section states that slice subsampling is
  per-epoch and derived from `(seed, epoch, index)`, and that the discovery and
  confirmatory sweeps therefore differ in one respect beyond the split. No
  hypothesis changes status.

  Recorded alongside it, same commit boundary, two enforcement gaps the same
  review found — neither changes a declared parameter either: `train_cached.py`
  now stamps `analysis_role`, `splits`, `splits_hash` and a `prereg` block into
  every result file (`PREREGISTRATION.md` promised this and training wrote none
  of it), suppresses test metrics on stdout under `--role confirmatory` (the
  breach recorded in the entry above, fixed at the source), and refuses to
  resume a run produced on a different split hash. `merge_results.py` no longer
  discards the provenance block on rewrite.

- **The H3 floor, re-run under the current hash — and what reproduction actually
  bought.** `runs/untrained_floor{,_mh}.json` were stamped `20bdd93b781d950d`,
  two amendments stale, which by the literal rule in `PREREGISTRATION.md` lines
  189-191 demoted the project's only confirmatory results to exploratory over
  bookkeeping. Both were re-run on `splits_confirmatory.json` under
  `33a55222c853a4cf`, written to `runs/untrained_floor{,_mh}_rehash.json`; the
  originals are kept rather than overwritten so the comparison stays auditable.

  `slot` came back **byte-identical** outside the stamp. `mh_abmil` did not:
  4 of its 30 inits differ, by at most **2.97e-09**. Every frozen slot is the
  same, and median, p5, p95, min and max agree to at least 9 decimal places
  (p95 0.7678969514750311 both times), so H3 is unchanged on both arms. The
  cause is ordinary GPU reduction non-determinism across nodes —
  `utils.seed.set_seed` leaves `cudnn.deterministic` off by default, on the
  stated grounds that it costs throughput and that seed-to-seed variance is
  itself a reported quantity. Recorded because "reproduces bit-for-bit" is a
  claim this project makes in `slotmil/prereg.py`, and on this arm the true
  claim is "to 3e-9", which is a different sentence.

  Alongside it, `slotmil/prereg.py` gained `amendment_chain` and
  `classify_hash`, and `prereg_freeze.py --check` now places every stamped
  result on the recorded chain as current / superseded-by-<date> / UNKNOWN. It
  currently reports 16 stamps: 0 current, 16 superseded, 0 UNKNOWN. This does
  **not** relax the rule — a superseded stamp is still not a match, and the
  paper labels results by what they ran on, not by ancestry. It removes the
  situation where every amendment silently demoted every prior result with no
  record of which amendment did it.

---

## 2026-08-15 — declare H5's unit; restate H10 with three outcomes

- **Kind:** amendment
- **Config hash before → after:** `33a55222c853a4cf` → `2fa8286b9c4bc9d0`
- **What changed:** H5 gained a `unit` (`mean_over_seeds`), a `report_per_seed`
  flag and the rationale for both; H10 was added as a new secondary hypothesis
  with three outcomes rather than two. Both are holes in what the freeze
  declared, not changes to what it committed to.

  **H5's unit.** "No arm's prior-normalised skill exceeds 0.30" fixes a
  threshold and leaves the unit free: the arm's number could be a per-seed value
  or an aggregate over seeds, and the two do not agree. This is the same class of
  hole as H2's "every trained arm", ruled by amendment on 2026-08-15 for the same
  reason. Declared as the **mean over the arm's float32 seeds**, because that is
  the statement's own noun and because H4 is already per-arm; a per-seed-max rule
  takes five draws at the threshold instead of one, which is the best-of-k
  inflation `scripts/template_family.py` already refuses when it declines to
  promote the strongest family member to denominator. `report_per_seed: true` is
  part of the ruling: every per-seed value is reported whether or not it
  falsifies, so that fixing the unit cannot become a way of not printing a
  number.

  **H10.** Drafted as a two-outcome test — "the paired interval contains zero for
  a majority of seeds" — which mis-scores its own evidence. Across the five
  trained discovery dumps the `masks:separable − trained` interval crosses zero
  on three and puts the content-free oracle strictly ahead on two:

  | dump | separable − trained (flat) | outcome |
  |---|---|---|
  | f32_seed0 | +0.0078 [−0.0098, +0.0257] | indistinguishable |
  | f32_seed1 | −0.0171 [−0.0356, +0.0026] | indistinguishable |
  | f32_seed2 | +0.0993 [+0.0786, +0.1193] | oracle-wins |
  | real_seed0 (fp16) | +0.0077 [−0.0099, +0.0256] | indistinguishable |
  | real_seed1 (fp16) | −0.0178 [−0.0364, +0.0019] | indistinguishable |
  | real_seed2 (fp16) | +0.1327 [+0.1135, +0.1514] | oracle-wins |

  Both of those outcomes are *stronger* than the claim, not weaker, so a
  two-outcome form would have recorded an oracle win as a failure of a
  hypothesis about oracle wins. Declared instead as oracle-wins /
  indistinguishable / trained-wins, with only a trained-wins majority
  withdrawing the claim. The member is pinned to `separable` and all four are
  reported, for the same reason the denominator is pinned.

- **Why now, and the order it happened in — this is the part that matters.**
  Prior-normalised skill is a pre-registered primary estimand that had never
  been computed into any document; it existed only inside
  `runs/nulls/template_family.json`. Read out, the trained arm's skill was
  0.2615 (`f32_seed0`), −0.0876 (`f32_seed2`) and, on the three fp16 dumps that
  `protocol.dtype` excludes, 0.2619 / **0.3159** / −0.2335. The single value above
  the 0.30 threshold sat in an fp16 dump and so could not touch H5 — and
  `f32_seed1`, the float32 dump that would decide it, **had never been
  collected**.

  So the unit was ruled first, in that state, and then the missing dump was
  collected:

  1. The unit was chosen and committed while `f32_seed1` did not exist. What was
     known: two valid seeds at 0.2615 and −0.0876 (max 0.2615, mean 0.087), and
     that both candidate units passed on them.
  2. `f32_seed1` was then collected
     (`scripts/slurm/null_collect_f32.sbatch`, job 7256442, checkpoint
     `runs/lidc/slot_div=0.5/seed1/best.pt`, discovery split, float32).
  3. Its skill is **0.3121** — above 0.30.

  Under a per-seed-max unit that value falsifies H5 on the discovery split.
  Under the declared per-arm-mean unit the arm reads 0.162 and does not. The
  ruling therefore changes the verdict, which is exactly why the sequence is
  recorded rather than summarised: the choice was made before the number
  existed, and the number is printed here next to it rather than being
  discovered later by a reader.

  Two further things this settles. The fp16 exclusion never protected H5; it
  deferred it — fp16 read 0.3159 and float32 reads 0.3121, both above threshold,
  so the dtype rule was removing an inadmissible dump, not a favourable one. And
  the seed-1 direction was the opposite of seed 2's: fp16 ran 0.0007 *higher*
  than float32 here (0.8679 vs 0.8672) against 0.034 lower on seed 2, so the
  fp16 defect is seed-specific in sign as well as size.

- **Results already seen?** **Yes**, and this is the complete list, all on
  `data/lidc/splits.json` — the discovery split, which `PREREGISTRATION.md`
  labels exploratory by construction:
  the eight per-dump prior-normalised skill values and paired
  `separable − trained` intervals in `runs/nulls/template_family.json`; the
  axis-gate rows in `runs/nulls/axis_gate.json`; and the fp16/float32 pairs in
  `runs/nulls/fp16_check.json`. No confirmatory result of any kind was read: no
  arm has been trained on `splits_confirmatory.json` and no seed-2027 attention
  dump exists. Fixing a confirmatory unit against discovery numbers is the same
  move the freeze already made for every threshold in the document; it is
  legitimate because it is disclosed, and the disclosure is the sequence above.
- **Consequence for the paper:** H5 and H10 remain confirmatory, and are
  evaluated on the seed-2027 split. The discovery skill values are reported as
  exploratory, including `f32_seed1`'s 0.3121, and the methods section states
  the unit and the order in which it was fixed. If a confirmatory arm's mean
  clears 0.30, H5 fails on the declared unit; if only a seed does, that seed is
  reported and H5 stands.

---

## 2026-08-15 — implement CLAM-SB and DSMIL; rule what each paper leaves open

- **Kind:** amendment
- **Config hash before → after:** `2fa8286b9c4bc9d0` → `93e006fd8c7d8a57`
- **What changed:** promoted `clam_sb` and `dsmil` from `planned` to
  `implemented`, and declared the choices each requires that the original freeze
  did not specify. Same procedure as the two Harvey arms, and for the same
  reason: every one of these is a free parameter that could otherwise be settled
  after seeing a number.

  **`clam_sb` — base arm, B, loss, subtyping.** Implemented as a subclass of
  `gated_abmil`, so the attention path is bit-identical and the entire
  difference between the two arms is the clustering objective; `hidden` stays at
  the project's 128 rather than CLAM's 256 for the same reason. `B = 8`, CLAM's
  printed value, rather than a fraction of the bag: LIDC bags reach 43.8k
  instances against a WSI's few thousand, so a proportional *k* would make the
  term's effective weight scale with scan depth — roughly 12× across a 58–700
  slice cohort — and the branch exists to sharpen attention's extremes, not to
  label the bag. Where a bag cannot yield 2k disjoint instances, *k* falls to
  `n_valid // 2` for that bag, so the top and bottom sets cannot overlap and
  train the classifier on contradictory labels for one instance. Bag/instance
  weighting is CLAM's printed 0.7/0.3, run literally via `SlotMILLoss(w_bag=...)`
  rather than folded into the learning rate. `subtyping: false`, CLAM's default,
  so only the bag class's instance classifier is supervised — which is why the
  classifiers are per class rather than one shared head, since a shared head
  would invert the meaning of "top attention" on negative bags.

  The one departure: **cross-entropy on the pseudo-labels instead of the
  reference implementation's smooth top-1 SVM.** The SVM surrogate arrives as a
  third-party package for a margin that no pre-registered estimand reads — the
  branch shapes attention, and every estimand here scores attention by rank — so
  CE is the standard surrogate for the same 2k pseudo-labels at no cost to what
  is measured, and keeps the dependency set auditable.

  **`dsmil` — K, q width, dropout, and where the instance stream enters.**
  `K = num_classes`, not 1: DSMIL picks a critical instance per class and builds
  one bag embedding per class, so collapsing to a single token would be a
  different method. `build_model` therefore gives it `k_eff = num_classes`; the
  frozen-slot protocol already handles K > 1 (Hungarian on validation, argmax of
  the lesion column at F=2), so the published shape costs nothing.
  `q_hidden = 128` and `dropout_v = 0.0`, both DSMIL's own values. Instance-stream
  weight 0.5, matching the printed 0.5/0.5 split.

  Two departures, both to keep the arm comparison attributable: the bag score
  comes from the project's shared gated readout rather than DSMIL's per-class
  `Conv1d`, and the instance stream is supervised through the loss rather than
  averaged into the reported logits. Averaging the two streams would give DSMIL
  a classifier no other arm has and confound the classification side of H1 and
  H2. The localisation estimands are unaffected — they read `attn`, which the
  bag stream produces in full.

  **Both arms normalise attention over the instance axis**, like every other
  arm in `baselines.py` and unlike slot attention. Stated because `ENGINEERING.md`
  requires every new arm to state it, and because it decides whether
  `localization.instance_auc(slot=None)` is valid for them. It is — that
  function's max-over-slots is what read 0.4885 against the frozen-slot path's
  0.8423 on slot-normalised attention, the project's most expensive past bug.

  Two enforcement points came with the arms. Per-instance logits reach the loss
  through `out["instance_logits"]`, surfaced by `MILModel.forward` for any
  pooling marked `InstanceScoringPool` — the route `out["health"]` already takes,
  rather than widening the pooling contract for two arms. And
  `scripts/train_cached.py` refuses an auxiliary-stream arm whose stream weight
  is zero: `clam_sb` without its clustering term is plain `gated_abmil` and
  `dsmil` without its max term is a single-stream non-local pooling, and either
  would train, write a valid `result.json`, and be reported under a published
  method's name. That is the failure `lam` had.

- **Results already seen?** **No.** Neither arm has been trained, no estimand
  has been computed on either, and no attention has been dumped for either. The
  only numbers produced while implementing them are shape and gradient checks on
  random tensors.
- **Consequence for the paper:** the confirmatory sweep grows from seven arms to
  nine (`scripts/slurm/lidc_confirmatory_array.sbatch`, `--array=0-8`). The
  discovery array is deliberately left untouched — it must stay submittable
  exactly as it was or the published discovery numbers stop being reproducible.
  `transmil` remains `planned` under its 2026-09-22 drop rule. The CLAM and
  DSMIL departures above are stated in the methods section; a baseline that
  differs from its paper and does not say so is a worse baseline than one that
  does.

---

## 2026-08-15 — scope the hypothesis arm sets, name H10's arm, define the Holm family, and declare two more confirmatory conditions

- **Kind:** amendment
- **Config hash before → after:** `93e006fd8c7d8a57` → `7d0a43b7c301e332`
- **What changed:** six holes, all of the same class as H5's missing unit and
  H2's missing scope: a hypothesis that names a threshold but not the set it
  applies over, or a statistic but not how it is computed. Each is ruled below
  and each is ruled **before the confirmatory sweep is submitted**, which is the
  only reason any of it is admissible.

### 1. Arms carry a `scoring_class`, and hypothesis arm sets are declared

H2 already excluded `centre_gaussian` by amendment, via a prose
`trained_arm_definition`. That ruling was right and is unchanged; what it lacked
was a mechanism, so the next hypothesis with the same problem needed a second
prose exclusion beside it. Each arm now declares a `scoring_class`:

| class | arms | why |
|---|---|---|
| `learned_attention` | `abmil`, `gated_abmil`, `mh_abmil`, `slot:div=0.5`, `normal_guidance`, `clam_sb`, `dsmil` | attention path contains fitted parameters |
| `uniform_attention` | `mean` | `MeanPool.forward` returns exactly `1/n` |
| `fixed_geometric_attention` | `centre_gaussian` | attention is a function of slice index alone |

H1, H2, H4, H5 and H6 declare `arm_set: learned_attention`. H2's existing
`excluded_arms: [centre_gaussian]` is retained and is now redundant with the
class, which is the intended end state — the prose ruling and the mechanism
agree.

**Why `mean` and `centre_gaussian` are excluded, and it is arithmetic, not
empirical.** `MeanPool` (`slotmil/models/baselines.py:36-52`) returns uniform
attention, so every score in every bag ties: `roc_auc_score` credits 0.5 on the
flat, slice and within-slice axes alike, `|flat − within| = 0` exactly, and every
skill is exactly 0. It is a guaranteed H1 pass that measures nothing.
`CentreGaussianPool` attention depends only on slice index, so it is constant
within a slice; `per_bag_axes` therefore scores `within_slice = 0.5` on ties
while `flat` carries the axial prior, and the gap cannot be small. Neither
outcome is a measurement. This is the same defect as the `chance_affinity`
"1.00× lift" already recorded in `RESULTS.md` — a baseline that is 0.5 by
construction was reported as a result.

**The exclusion cannot change H1's verdict, and this is checkable.** H1
falsifies on a majority of arms failing. Over 9 arms a majority is 5; because
`centre_gaussian` contributes one guaranteed failure and `mean` one guaranteed
pass, falsification requires 4 failures among the other 7. Over the 7-arm set a
majority is 4 — **the same four**. The ruling removes two uninformative rows
and moves the bar by exactly zero. It could not have been chosen to change the
outcome, which is the property a scope ruling needs and the one H2's prose form
could not demonstrate.

Both arms stay in the sweep, in the classification table and in the axis table,
with their construction-fixed values printed and the reason given. Excluding an
arm from a hypothesis is not excluding it from the paper — the `report_per_seed`
precedent applies: a ruling must not become a way of not printing a number.

### 2. H1 declares positive controls and a tie-floor check

A test that cannot fail is ceremony, and H1 has never been shown to be able to
fail. Three references already scored by `scripts/template_family.py` carry
axial information and therefore must exceed the 0.02 threshold:
`masks:axial`, `masks:separable`, `centre_prior`. If any of the three comes in
**under** 0.02 on the confirmatory split, the H1 harness is broken and the H1
verdict is `void` rather than supported. Separately, `mean` must return exactly
`0.0000`; anything else means the tie handling changed underneath the estimand.

Both cost nothing — every one of these rows is already computed — and they
convert "can your test fail?" from a reviewer's question into a printed row.

### 3. H10 names its arm

H10 reads "the paired per-bag difference between the mask-fitted separable
template and **the trained arm**" — singular, and never specified. With nine
arms in the sweep that is a best-of-9 selection sitting inside a hypothesis
whose own `member_rationale` refuses best-of-k for the template family. Ruled:
the verdict is taken on **`slot:div=0.5`**, the arm every discovery number and
every threshold in this document came from. All `learned_attention` arms are
scored and reported; only that one carries the pre-registered verdict.

### 4. Patient-specific skill is operationalised

H4 and H6 both read `patient_specific_skill` (`auc_real − auc_cross_patient`)
and neither says on which axis, over how many derangements, or with what RNG —
so three people would compute three different numbers. Ruled: **flat axis;
R = 100 derangements through `nulls.shuffle_masks_across_bags`; numpy
`default_rng(0)`; mean over derangements; paired per bag against the real score;
patient-clustered bootstrap at the pre-registered 10 000 reps and seed 0.**

### 5. The Holm family is enumerated, and verdicts are scored on point estimates

`statistics.multiplicity: holm` declared a correction over a family the config
never defined, and eight of the ten hypotheses emit no p-value at all — their
falsifiers are written as comparisons of a point estimate to a threshold. Two
rulings, and the order matters:

- **Hypothesis verdicts are scored exactly as their falsifiers are written**, on
  point estimates, with the patient-clustered 95% CI reported beside every one
  and never changing a verdict. Rewriting point-estimate falsifiers as interval
  rules now would loosen or tighten every one of them, in a direction chosen
  with the discovery numbers already in hand. H9 and H10 state CI-based
  falsifiers and keep them.
- **Holm therefore applies to a separately named family** of paired DeLong
  comparisons supporting the secondary arm-vs-reference tests, listed by name in
  `statistics.holm_family`. `classification.holm_reject` and `delong_test` are
  implemented and unit-tested and have had no caller outside `tests/`; this is
  what they are for.

### 6. Two more confirmatory conditions, and how their families are counted

`malignancy` was declared in `conditions` with no confirmatory split, and
`balanced_presence` was not declared at all despite `runs/lidc_balanced/`
holding five seeds of discovery results. Both now carry seed-2027
patient-grouped splits drawn by `scripts/make_splits.py`, on the same argument
that made the LIDC re-split clean: the DINOv2 cache is frozen and unsupervised,
so no label information crosses the boundary.

| condition | split | series | train/val/test | hash |
|---|---|---|---|---|
| `malignancy` | `data/lidc/splits_malignancy_confirmatory.json` | 734, 726 patients | 512/111/111 | `8cb89c9a0b70b759` |
| `balanced_presence` | `data/lidc/splits_balanced_confirmatory.json` | 262, 262 patients | 184/40/38 | `554390c2a05a663b` |

**Their power limits are stated here rather than discovered at analysis time.**
`balanced_presence` has **38 test bags over 38 patients**; a 10 000-rep
patient-clustered bootstrap over 38 clusters produces intervals wide enough that
most outcomes will land `indistinguishable` by default, and that is a property
of the design, not a finding. `malignancy` has a supervised patch-probe ceiling
of 0.6516 against nodule presence's 0.9102, and `RESULTS.md` already records the
MIL arms reaching ~91% of that ceiling — the condition tests the label, not the
method.

Ruled accordingly: **`nodule_present` carries the pre-registered hypothesis
family and the Holm correction. `malignancy` and `balanced_presence` are
reported per-condition as consistency checks and do not join that family.**
Correcting over 27 tests instead of 9 would spend the power of the condition
that carries the claim to protect two that cannot carry it, and choosing which
condition leads after seeing three verdict tables would be the same best-of-k
error refused everywhere else in this document.

- **Why now:** the confirmatory sweep is submitted next. Every one of these six
  is a choice that, left open, would be made with a confirmatory number already
  on screen — which is precisely the failure this document exists to prevent.
  Section 1's arithmetic neutrality is offered because "we excluded two arms
  from the lead hypothesis" is otherwise exactly the sentence a reader should
  distrust.

- **Results already seen?** **Yes, all on the discovery split**, and this is the
  complete list. The per-scorer axis rows in `runs/nulls/template_family.json`,
  from which sections 1 and 2 quote `masks:axial` 0.0978, `centre_prior` 0.0401,
  `masks:separable` 0.0259 and `attention:inplane` 0.0020; the eight trained-dump
  gaps in `runs/nulls/axis_gate.json` (0.0014–0.0137, all under 0.02), already
  published in `RESULTS.md` before this entry; and the split sizes printed by
  `make_splits.py` above, which are counts rather than outcomes. **No
  confirmatory result of any kind was read**: `runs/lidc_confirmatory/` does not
  exist, no arm has been trained on any seed-2027 split, and no seed-2027
  attention dump exists. The discovery numbers are what make sections 1 and 2
  *checkable* — the exclusions are provable from the code alone, and the
  positive controls are declared because the data shows the test can fail.

- **Consequence for the paper:** H1's denominator is 7 arms and the falsification
  bar is unchanged at 4 failures; `mean` and `centre_gaussian` appear in every
  table with their construction-fixed values and the reason. H10's verdict is
  `slot:div=0.5`, all arms reported. H4 and H6 have a computable estimand for the
  first time. The verdict table reports point estimates with CIs beside them, and
  a separately named DeLong family carries the Holm correction. Three conditions
  run; one carries the family. The methods section states that the arm scoping
  was ruled before submission and that it does not move H1's bar.

---

## 2026-08-15 — define H7's denominator and its content-free set, and fix the probe's protocol

- **Kind:** amendment
- **Config hash before → after:** `7d0a43b7c301e332` → `de43c8bcdc5053b6`
- **What changed:** H7 gained a denominator rule for the probe, an enumerated
  `content_free_set`, and a `probe_protocol`. Nothing about the estimand itself
  moves.

**This is a separate entry from the scoping amendment above, on purpose.** That
one answers "results already seen?" with discovery numbers that only *check* its
rulings; this one answers with discovery numbers that would otherwise *decide*
one of them. Merging the two would launder the difference.

### The defect: H7 is not computable as written, in three different ways

H7 reads: *the supervised patch probe's prior-normalised skill exceeds 0.50
while every content-free baseline's is below 0.05.* Working it against what is on
disk:

**1. The probe has no declared denominator.** `reference_baselines.fitted_template`
is fit to *the scorer's own* validation attention (`nulls.global_template(attns,
masks, slot)` accumulates `a[slot]`, not the masks). Every arm therefore gets its
own denominator. The probe is not an arm and nobody said which template it is
scored against. Borrowing another arm's is arbitrary, and the arbitrariness is
not small: carrying the published probe AUC of 0.9102 through each of the eight
discovery denominators gives skill **0.4720 to 0.7224**, and the lowest of those
is *below the 0.50 threshold*. The clause H7 leads with can be passed or failed
by a choice nobody made.

**2. `probe_ceiling.py` cannot produce the number at all.** It trains on 60 scans,
tests on 40, subsamples negatives at `neg_per_pos=20`, and returns a bare patch
AUC. A skill needs the probe scored on the same bags and the same patch grid as
the template, or the numerator and denominator do not correspond. 0.9102 is not
divisible into a skill.

**3. "Every content-free baseline" is an undefined set, and it contains a
falsifier.** The mask-fitted template family in `slotmil/eval/templates.py`
post-dates the freeze and reads no test image, so on the plain reading of
"content-free" it qualifies. Its skill against the declared denominator is
**+0.298 / +0.224 / +0.346** on the three float32 dumps, up to **+0.537** on an
untrained init: `masks:separable` is above 0.05 on 8 of 8 dumps and
`masks:inplane` on 7 of 8. Under that reading H7 is falsified on discovery data
and the constructive half of the paper is blocked.

### Rulings

**(a) The probe is scored against its own self-fitted in-plane template**, exactly
as every arm is. Its per-patch scores take the place of attention, so
`templates.inplane_template(source="attention")` applies unchanged. This is not a
new rule — it is the existing `fitted_template` rule applied to a scorer that was
never explicitly handed to it. It also removes the 0.4720–0.7224 range above:
there is one denominator, and it is the probe's own.

**(b) The mask-fitted family is excluded from the content-free set, as oracle
references.** "Content-free" in this document has meant *reads no test image*
(`PREREGISTRATION.md`: "a 256-number in-plane map fit on validation that never
reads a test image"). The `masks:*` family satisfies that and still fails the
sense H7 needs, because it is fit to validation **lesion masks** — it is
content-free but not *label-free*. H7's whole purpose is to show the estimand
separates a supervised scorer from unsupervised ones; requiring a reference that
was itself fit to lesion labels to score near zero contradicts the clause
immediately before it, which requires a supervised scorer to score high. A set
containing both demands is not strict, it is inconsistent.

**(c) `centre_prior` is reported as its own row and is not in the gate's set.**
It is axially informative *by construction* — it ranks by distance from the
volume centre — while the denominator is in-plane and, as
`slotmil/eval/templates.py` records, has a slice AUC of exactly 0.5 by
construction. Skill above zero for `centre_prior` therefore measures something
the denominator cannot represent, which is this paper's own thesis, not a failure
of the instrument. Using the finding to falsify the gate that measured it would
be circular.

  **`centre_prior`'s prior-normalised skill has not been computed.** It is not in
  `runs/nulls/template_family.json`, which stores the family per trained dump and
  scores `centre_prior` only as a scorer row (flat 0.7752, slice 0.6026,
  within-slice 0.7351). This ruling is therefore made **before** the number
  exists, which is the same order the H5 unit was ruled in and the reason this
  clause is admissible at all. Whatever it comes out as, it is reported.

**(d) The gate's content-free set is enumerated by name:** `chance`,
`roll_permutation`, `entropy_matched_random`, `fitted_template`. All four are in
the pre-freeze `reference_baselines` block; none reads a label; `fitted_template`
scores exactly 0 against itself by construction and is included so the set is
never empty.

**(e) The probe protocol.** Fit on the confirmatory **train** split — all lesion
patches plus `neg_per_pos=20` sampled negatives, `default_rng(0)`. Score **every
patch of every confirmatory test bag**, emitted in the `(attns, masks)` shape the
dumps use so it flows through `per_bag_axes` and `fit_family` unchanged.
Subsampling at fit time is a fitting choice and touches no estimand; subsampling
at score time changes the estimand and is forbidden.

**(f) Rejected alternative, recorded.** Scoring H7 on the within-slice axis, where
the in-plane denominator is commensurable by construction, also removes the
inconsistency. Rejected because H5 reads the same estimand on the flat axis, and
two hypotheses reading one estimand on two axes is a worse trap than the one
being closed.

### The cost, paid in the paper rather than hidden

This ruling makes H7 easier to pass. Three things are therefore committed to
here, and the paper reports them whether or not H7 clears:

1. `masks:separable`'s skill (0.119–0.537 on discovery) as evidence that **the
   declared denominator is not the strongest available reference** — a fitted
   oracle beats it, and the honest reading of prior-normalised skill against an
   in-plane template is an **upper bound** for any scorer carrying axial
   structure.
2. `centre_prior`'s skill, once computed, in the same table.
3. The sentence that an in-plane denominator under-credits axial content-free
   structure, stated as a limitation of the estimand we are recommending.

- **Results already seen?** **Yes**, all on the discovery split and all
  exploratory by construction. The complete list: the per-dump
  `prior_normalised_skill` block in `runs/nulls/template_family.json`, from which
  every `masks:*` figure above is quoted; the scorer axis rows in the same file;
  and the probe AUC 0.9102 published in `RESULTS.md` since 2026-08-11, carried
  through each stored denominator by arithmetic in this entry. **Not seen, and
  deliberately not computed before ruling (c):** `centre_prior`'s
  prior-normalised skill. **No confirmatory result of any kind:** no arm is
  trained on any seed-2027 split and no seed-2027 dump exists.

- **Consequence for the paper:** H7 remains the power gate and remains
  confirmatory, evaluated on the seed-2027 split with the probe scored against
  its own fitted template over the full test set. Its content-free set is the
  four named references. The mask-fitted family and `centre_prior` are reported
  beside it, not inside it, and the limitation in (c) is stated in the discussion
  rather than left for a reviewer to find. If the probe still fails to separate,
  the pre-registration's original consequence stands: the protocol is not
  recommended.

---

## 2026-08-15 — the H3 floor re-run at the current head; both arms bit-identical

- **Kind:** deviation (record only — no config change)
- **Config hash before → after:** `de43c8bcdc5053b6` → `de43c8bcdc5053b6` (unchanged)
- **What changed:** no declared parameter. `runs/untrained_floor{,_mh}_rehash.json`
  were stamped `33a55222c853a4cf`, which the two amendments above left two
  transitions behind, so by the literal rule in `PREREGISTRATION.md`'s "Verifying
  the chain" the project's only confirmatory results were demoted to exploratory
  for the **third** time over bookkeeping. Both poolings were re-run on
  `data/lidc/splits_confirmatory.json` under `de43c8bcdc5053b6` and written to
  `runs/untrained_floor{,_mh}_rehash2.json`, jobs 7256783 and 7256784. The
  earlier files are kept rather than overwritten so the comparison stays
  auditable — same handling as the 2026-08-15 re-run.

- **H3 is unchanged and passes.** `slot` p95 **0.7665760928074817**, `mh_abmil`
  p95 **0.7678969514750311**, both above the pre-registered 0.70 over 30 inits.

- **Both arms reproduced bit-identically this time, and that is worth stating
  precisely because last time one did not.** 0 of 30 inits differ on either
  pooling; every frozen slot is the same; median, p5, p95, min and max are equal
  to every printed digit. The previous re-run had `mh_abmil` differing on 4 of 30
  inits by at most 2.97e-09, attributed to GPU reduction non-determinism across
  nodes with `cudnn.deterministic` left off. That attribution survives: the
  discrepancy is node-dependent, not run-dependent, so "reproduces bit-for-bit"
  is a claim about a run that happened to land on comparable hardware and **"to
  3e-9" remains the honest general claim**. Recorded so that a reader comparing
  the two entries does not conclude the earlier one was wrong.

- **Why the treadmill is being paid down rather than argued away.** Three
  re-stampings in two days is the visible cost of a rule that reads a stamp
  against the head hash. The rule is not being relaxed: `classify_hash` already
  distinguishes `superseded` from `UNKNOWN`, and only `UNKNOWN` fails `--check`,
  which is the correct division. The cost is ~7 GPU-minutes per pooling, and
  paying it is strictly cheaper than arguing in the paper about whether an
  ancestor stamp counts. Both amendments above were deliberately bundled so this
  happens once rather than twice.

- **Results already seen?** **Yes** — the H3 floor itself, on the confirmatory
  split, both before and after the re-run, and they are equal. Nothing was chosen
  against them: this entry changes no parameter, and the re-run was mechanical.
  No other confirmatory result existed when it ran; the three sweeps submitted
  the same day (7256785, 7256799, 7256800) had produced no `result.json` yet.

- **Consequence for the paper:** H3 is reported as confirmatory, stamped
  `de43c8bcdc5053b6`, with the floor as a distribution (median, 5th–95th
  percentile) rather than a point, as `reference_baselines.untrained_fleet`
  requires. `prereg_freeze.py --check` now reports 2 current stamps where it
  reported 0.
