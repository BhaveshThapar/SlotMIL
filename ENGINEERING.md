# SlotMIL — working notes

Contracts and procedures only. **Results live in `RESULTS.md`; do not duplicate
them here.** A second copy of a number goes stale silently, and this file is
loaded into every session, so a stale line here is worse than no line.

Run everything through the venv: `source .venv/bin/activate`. Either `pytest` or
`python -m pytest` works as of 2026-08-15: `tests/test_lung_mask_io.py` imports
from `scripts/`, which has no `__init__.py`, so the repo root has to be on
`sys.path` — `python -m pytest` puts it there and a bare `pytest` does not, so
`pyproject.toml` now sets `pythonpath = ["."]` and both forms collect. CI runs
the bare form; do not remove that line or CI fails at collection with 0 passed
rather than 1 failed.

`ruff check .` and the scoped `mypy` invocation in `.github/workflows/ci.yml`
must both stay clean. Ruff's ignore list carries a reason per rule; `B905`
(`zip()` without `strict=`) is ignored **pending an audit, not permanently** —
several sites zip per-bag `attns` against `masks`, where a length mismatch would
truncate the analysis silently rather than raise. Mypy is scoped to the nine
modules that compute the reported numbers, and that scope is a floor to raise,
not a convenience to preserve — `slotmil/eval/verdict.py` joined it when the
hypothesis scorers landed, because it decides every reported PASS/FAIL/VOID.

## The pooling contract

Every pooling module maps `(feats [B, N, D_in], pad_mask [B, N]) -> (tokens [B, K, D], attn [B, K, N])`,
with `pad_mask` True for real instances. `MILModel.forward` wraps that into
`{"logits" [B,C], "attn" [B,K,N], "tokens" [B,K,D], "attribution" [B,K]}`, plus
two conditional keys: `"health"` if and only if the pooling is a `SlotAttention`,
and `"instance_logits"` if and only if it is an `InstanceScoringPool`
(`clam_sb`, `dsmil` — their auxiliary streams are supervised per instance, which
the two-tuple contract has no room for). Both go through `out` rather than
widening the contract, and `instance_logits` is **recomputed from `feats`, never
cached on the module**: a cache would make the value depend on whether `forward`
ran first, which produces a wrong number rather than a crash.

Two invariants that are tested and that silent breakage would hide:

- **Padded columns of `attn` must be exactly 0.0**, not merely small. Every eval
  consumer slices `attn[i, :, :length]` and renormalises, so a nonzero pad steals
  mass. `_masked_softmax` in `slotmil/models/baselines.py` fills with `-inf` and
  gets this for free — use it rather than a second masking convention.
- **`attn` normalisation differs by family.** Slot attention softmaxes over the
  *slot* axis; ABMIL, gated ABMIL, MH-ABMIL, `centre_gaussian` and `transmil`
  softmax over the *instance* axis. This is not cosmetic:
  `localization.instance_auc(slot=None)` takes a max over slots and is
  **invalid** for slot-normalised attention — it reads 0.4885 where the
  frozen-slot path reads 0.8423. That discrepancy is the project's most
  expensive past bug. Any new arm must document which axis it normalises on.

`transmil` adds a third thing a reader has to know, because its reported
attention is **not** the tensor its own forward pass used. The block aggregates
with the Nystrom approximation; `attn` is the class token's *exact* softmax row.
The contract requires a distribution — padded columns exactly 0, real ones
summing to 1 — and a Nystrom row is neither normalised nor non-negative, quite
apart from costing a 1.9e9-entry matrix per head to extract at LIDC bag size.
The exact row is O(N d) and is the quantity the approximation approximates;
`tests/test_transmil.py` pins them as the same thing by showing the gap shrinks
with the pseudo-inverse iteration count rather than by picking a tolerance.
Its squaring pad is masked, not filled with repeated instances as the reference
does, and PPEG's convolution therefore has to zero the pad *before* convolving:
a 7x7 depthwise kernel carries a nonzero pad into real positions, where the
attention mask can no longer remove it.

## There is no model registry

Arm construction is **three hand-written `if/elif` switches**, each raising
`ValueError` on an unknown name. There is no dict to append to; adding an arm
means editing all the relevant ones:

- `slotmil/models/baselines.py::build_pooling` — the non-slot poolings
- `slotmil/models/mil.py::build_model` — special-cases `slot` and `mh_abmil`;
  everything else falls through to `build_pooling` with `k_eff = 1`. An arm that
  produces K > 1 tokens needs its own branch or the readout is sized wrong.
- `slotmil/models/heads.py::build_readout`

Arm *specs* are strings like `slot:div=0.5` or `normal_guidance:lam=0.1`, parsed
by `scripts/train_cached.py::parse_arm`, which coerces **every override to
`float`** — no bare flags, no string values. Run directories are the spec with
`:` replaced by `_`, so `normal_guidance:lam=0.1` writes to
`runs/<cond>/normal_guidance_lam=0.1/seed<N>/`.

An arm whose method *is* an auxiliary loss term needs a guard, not just wiring.
`clam_sb` without its clustering term is bit-identical to `gated_abmil`, `dsmil`
without its max term is a single-stream pooling, and `normal_guidance` without
`lam` is its own base arm — all three would train, write a valid `result.json`
and be reported under the published name. `train_cached.py` refuses each of them
rather than trusting the spec.

**A parsed override that is never wired reaches nothing and warns about nothing.**
`lam` was parsed and silently dropped for as long as the arm was unbuilt; the arm
would have trained as its base and been reported as Normal Guidance. If you add
an override, wire it where the criterion is constructed and add a guard that
refuses the run if it is missing.

Hard allow-lists that a new arm must also be added to, or it fails only at
submission time: `scripts/untrained_floor.py` and `scripts/null_collect_attn.py`
(`--pooling choices=[...]`), `scripts/reeval_all_alignment.py` (`ARMS`), and
**four** sbatch files carrying `ARMS=(...)` **and** `--array=0-N` that must be
raised together — `lidc_train_array.sbatch`, `lidc_confirmatory_array.sbatch`,
`lidc_condition_confirmatory_array.sbatch`, `mosmed_confirmatory_array.sbatch`,
plus `confirmatory_collect_array.sbatch`. Each has a bounds check that exits 1
when the two drift, which is the only reason this is a nuisance rather than a
silently short sweep.

Two test guards used to be literal counts of "7 learned_attention arms" and went
red on the tenth. They now compare against `scoring_class`, because bumping a
literal is exactly the move that would let a misclassified arm through.

## Auxiliary losses

`slotmil/train.py` calls `criterion(out, labels, pad_mask=..., slice_index=...)`
and passes nothing else off the batch. So a regulariser gets `out`, `pad_mask`
and `slice_index` — anything else must either be stashed in `out` by the model
(the `health` precedent) or plumbed explicitly.

Prefer the loss. `normal_guidance_loss` needs the slice index, and routing that
through `collate_bags -> train.py -> criterion` left `MILModel.forward` untouched;
stashing it in `out` instead would have forced a signature change across all nine
`model(feats, pad_mask)` call sites.

The pattern in `SlotMILLoss.forward` is `if self.w_X > 0: v = f(...); loss = loss
+ self.w_X * v; comps["X"] = v.detach()`. Every `comps` key is auto-logged as
`loss_<key>` in `history.json` — no extra wiring.

## Changing the pre-registration

`configs/prereg/isbi2027.yaml` is hashed into `PREREGISTRATION.md`, and the test
suite enforces the chain deliberately. Editing the YAML **will** turn the suite
red, and the sequence out is fixed:

1. Draft the `AMENDMENTS.md` entry first, hash `before → PENDING`.
2. Make the config and code changes together.
3. `python scripts/prereg_freeze.py --amend` — it refuses without `--amend` when
   the doc already records a different hash.
4. Fill the real after-hash into the entry.
5. Prose edits to `PREREGISTRATION.md` are safe *after* the freeze:
   `canonical_hash` covers the YAML only.
6. Suite green + `prereg_freeze.py --check` reporting MATCH, then **one commit**.

Two traps worth knowing before you start:
- `test_planned_arms_are_not_yet_constructible` asserts every `status: planned`
  arm **raises** when built. Implementing one turns the suite red until the YAML
  is promoted — by design, so an arm cannot be quietly skipped.
- `test_implemented_arms_are_constructible` builds each arm with
  `input_dim=64, dim=32, num_slots=4` on a `(2, 15, 64)` bag. **15 is not a
  multiple of 256**, so a position-aware arm sees `S = ceil(15/256) = 1` and must
  survive it — a bare `(S-1)` divisor divides by zero on the first test.

Every entry must state **whether the affected results had already been seen**.
That sentence is the whole mechanism.

## SLURM

Training runs on `scavenger` (own quota, preemptible with `PreemptMode=REQUEUE`,
which is safe because `train_cached.py` resumes per seed on the presence of
`result.json`).

**Pin the GPU type in `--gres` itself**: `--gres=gpu:rtxa5000:1`. A node-level
`--constraint=Turing|Ampere|Hopper` is *not* sufficient — nodes like
legacygpu12/15 carry both Pascal and Turing cards, so the node matches the
feature and SLURM still hands out a card this torch build has no kernels for.

Standard preamble: `set -euo pipefail`, `module load Python3/3.11.11`, activate
`.venv`, and export `TORCH_HOME=/fs/nexus-scratch/bthapar/.cache/torch`,
`TMPDIR=/fs/nexus-scratch/bthapar/.tmp`, `PYTHONUNBUFFERED=1`.

## Memory, and the shape the OOM takes

24 GB card. The work is **I/O bound, not compute bound** — profiling showed 0%
GPU utilisation, which is why arms moved off `tron`.

The failure mode to expect is a `[B, N, big]` intermediate over a 45k-instance
bag. Parameter-matched MH-ABMIL bisects to `hidden = 288`, so `V(x)` and `U(x)`
each materialise `[4, 45312, 2304]` ≈ 1.67 GB, and with `tanh`, `sigmoid` and
their product that is ~6 GB per forward. Thirty of those resident OOMs a 24 GB
card.

The remedy is at the **driver** level, not inside the model: chunk the resident
fleet (`--model-chunk`), `del` and `torch.cuda.empty_cache()` between chunks, and
`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the sbatch. When you
chunk, **seed inside the builder** so init *i* is reproducible on its own —
chunking must not be able to change a number that has already been reported.

Note eval-time collection (`untrained_floor.py`, `null_collect_attn.py`) runs
fp32 with no autocast, so it is the tighter budget; training gets AMP for free.
`protocol.dtype` is pre-registered float32, so you cannot buy memory back by
dumping confirmatory attention in fp16.

## Provenance and scope, enforced by the driver

`train_cached.py` takes `--role {exploratory,confirmatory}`. It has three effects
and you want all of them:

- Every `result.json` and `summary.json` opens with `analysis_role`, `splits`,
  `splits_hash`, `splits_file_sha256` and a `prereg` stamp. Without this a
  confirmatory result is indistinguishable from a discovery one on inspection,
  which `PREREGISTRATION.md`'s closing "Verifying the chain" section says decides
  whether it counts. (Cited by section, not line number — the line moved once
  already and a stale pointer to the rule that decides what is confirmatory is a
  bad thing to have.)
- **Test metrics are suppressed on stdout under `--role confirmatory`** unless
  `--report-test` is passed. A declared scope the tooling ignores is a comment,
  not a control — that is exactly how two discovery test AUCs reached a job log
  that had declared them out of scope (`AMENDMENTS.md`, 2026-08-15).
- **Per-seed resume refuses to cross splits.** It compares the recorded
  `splits_hash`; a run predating stamping has none and is refused rather than
  assumed. Pointing a confirmatory job at `runs/lidc` would otherwise have
  skipped all 18 discovery seeds as "already complete" and reported them as
  confirmatory. Use a fresh run root.

`merge_results.py` carries the block up and refuses a directory that mixes roles
or split hashes.

**Know when your driver samples the stamp, because editing during a run poisons
it.** `git_state()` shells out to `rev-parse` and `status --porcelain` at the
moment `stamp()` is called, and the drivers call it at opposite ends:

- `train_cached.py` stamps **once per task, before the seed loop**. Every seed of
  that task carries the tree as it stood when the task *started*. Editing during
  the run is therefore safe, but submitting from an uncommitted tree brands all
  five seeds `git_dirty: true` — which is what happened on 2026-08-15, when the
  arrays were submitted moments before the commit describing them landed and 17
  tasks stamped `afebaf6` with a dirty flag they could never shed.
- `probe_gate.py`, `template_family.py`, `untrained_floor.py` and the `null_*`
  drivers stamp **at the end**, when they write. An edit made at any point while
  they run lands in their stamp. Three artefacts were lost to this in one
  session before the pattern was noticed.

So: commit before you submit, and hold edits while an end-stamping driver runs.
The check is `prereg_freeze.py --check` — it prints `current` / `superseded` per
stamp, and `0 current` after an amendment means every confirmatory result on
disk needs re-running.

This is not an NFS artefact and it is worth not re-diagnosing as one. A probe job
on a scavenger node reads `rev-parse HEAD` as the live head with an empty
porcelain status; compute nodes see the repository correctly.

Related, from the same session: **an amendment supersedes every stamp on disk.**
Cheap for H3 (two jobs, 3m22s and 6m59s), expensive for the sweep (27 tasks x 5
seeds x ~8h). Land pre-registration changes *before* submitting long jobs, not
during them.

## Verdicts

`scripts/prereg_verdict.py` is the pre-registration scorer. `scripts/final_verdict.py`
is a **discovery-era table** over `runs/control_*.npz` with three conditions
hardcoded; do not extend it into a scorer. The falsifiers themselves are
`slotmil/eval/verdict.py`, which takes already-extracted numbers so that the rules
are testable without data, and **no threshold is retyped into Python** — every bound
comes from `configs/prereg/isbi2027.yaml` through `Prereg.hypothesis`/`arm_set`.

Four outcomes, and the two extra ones are the point. `VOID` is a falsifier that
could not be *trusted* — H1's positive controls came in under threshold, the tie
floor moved, or an input is off the amendment chain — and `NOT_RUN` is a hypothesis
with no artefact. Collapsing either into PASS makes an unsupported hypothesis read
as supported.

**Build a dump tag from the arm `spec`, never the `name`.** `slot:div=0.5` has name
`slot_div0.5` and directory `slot_div=0.5`; a tag from the name looks for a file
that does not exist, and the symptom is a hypothesis that reads NOT_RUN rather than
an error. `verdict.arm_tag` is the one place that conversion lives.

Blinding is only as good as its coarsest spelling. An arm reaches the analysis layer
as a name, a spec **and** a tag, and every artefact keys its rows by tag — so a
verdict table keyed by tag is unblinded regardless of what the code column says.
`prereg_verdict.blind_substitutions` covers all three. Related: **keep arm names out
of message strings.** Structured fields get substituted; prose does not.

### A second dataset, and the two things that assumed there was only one

`mosmed_severity` became a confirmatory condition on 2026-08-17 because H9 needs
it. Both analysis arrays are now dataset-generic and resolve the split, the
feature cache and the run root from the same config stanza, so a third dataset
needs a condition block and nothing else. Two latent defects had to go first,
and both had the same shape — a path hardcoded to LIDC next to a value resolved
from the config:

- `confirmatory_collect_array.sbatch` **refused** non-LIDC conditions. What that
  guard protected was the hardcoded LIDC feature cache beneath it, which would
  have paired MosMed checkpoints with LIDC features and written dumps that
  looked entirely normal. The cache now resolves beside the split, so the
  pairing cannot come apart.
- `probe_gate.py` is invoked with a cache, and the analyse array passed LIDC's
  unconditionally.

MosMed runs a **strict subset** of the analysis: `template_family` and nothing
else. The config's split note says it "supports the stereotypy contrast (H9) and
nothing else, and the paper says so", and a number that exists is a number
someone eventually quotes. The guards are on `dataset`, not on the condition
name.

`runs/_audit_meta.json` is the uid→patient map every cluster bootstrap resamples
on, and it is **per dataset**: LIDC groups several series under one patient (999
series, 991 patients), MosMed is one study per patient so the map is the
identity. It was read by seven drivers and written by nothing committed until
`scripts/make_audit_meta.py`; `--check` rebuilds it from the feature cache and
diffs rather than overwriting. Pass `--meta` explicitly — the default is LIDC's.

### H7's content-free members are drawn, not computed once

Since 2026-08-17 the two stochastic members are drawn `content_free_draws` (30)
times per tag and each tag's skill is the **mean over its draws**. Before that
they were computed once: `default_rng(cf_seed)` was rebuilt per tag with
`cf_seed` fixed at 0, so every tag in a condition shared one realisation and
draw-to-draw variance was never estimable. That mattered because the gate is a
**maximum** over tags, which over a single realisation cannot distinguish a
member above the threshold from a member whose one draw was — the measured
draw-to-draw sd is larger than the threshold itself.

The aggregation is unchanged and is declared as `content_free_unit`. Draw 0
keeps the original seed so the published value stays locatable in its own
distribution. `cluster_bootstrap(reps=0)` returns the point estimate with no
interval, which is what 29 of every 30 draws want; do not reimplement "drop
non-finite, then take the mean" in a driver.

### The `--out` trap in the analysis drivers

`axis_gate.py`, `template_family.py` and `h8_in_lung.py` all default to
`--out runs/nulls/<name>.json` — the *exploratory* artefacts that set every
threshold in `PREREGISTRATION.md`. A confirmatory run that omits `--out` overwrites
them. Always pass `--dir` **and** `--out`;
`scripts/slurm/confirmatory_analyse_array.sbatch` does, and says why.

### Three places the declared analysis is not computable as written

Found by writing the callers, recorded in the artefacts rather than resolved by
amendment (an amendment supersedes every stamp on disk):

- **`cross_patient` cannot be a paired DeLong member.** It is a label-side null —
  `shuffle_masks_across_bags` changes the target, not the score — so there is no
  second score vector to pair. Those members enter the family at `p = NaN`, which is
  what `holm_adjust` documents: NaN still counts toward the family size, so the size
  the `note` pins is preserved and the member is never rejected.
- **`holm_family.axis` and DeLong measure different things.** `flat_instance_auc`'s
  declared impl (`null_battery.score`) is a *mean of per-bag AUCs*; DeLong is defined
  for one AUC over one sample set. `holm_family.py` computes z and p on the pooled
  form and reports the declared mean-per-bag statistic beside every comparison,
  under its own name.
- **Holm is not load-bearing at pooled-instance scale.** ~7.4M paired instances put
  the DeLong p at exactly 0.0 by underflow, so every pairable member is rejected
  whatever the multiplier. `n_p_underflowed_to_zero` is emitted per seed so "n of m
  rejected" is not read as discrimination.

### Units the config does not declare

H1 and H4 state a per-arm bound with no unit over seeds; H9 states an ordering with
no aggregation over arms and seeds. Taken as **mean over seeds** for H1/H4 — H5's
`unit_rationale` makes exactly this argument for the identical hole, and a per-seed
maximum takes five draws at the threshold instead of one — and as **every matched
tag** for H9, the strictest available reading. All three are recorded in the verdict
artefact under `undeclared_units_taken`, not left implicit.

## Slice subsampling

`FeatureBagDataset` derives its subsample from `(seed, epoch, index)` via
`_subsample_rng`, and `train.fit` calls `set_epoch` before each epoch's iterator
is created. Do not reintroduce a `Generator` held on the dataset: DataLoader
forks it to every worker with identical state, and `seed_worker` reseeds only the
legacy `np.random` global, never a Generator object. That earlier form correlated
across workers, reset every epoch (nothing sets `persistent_workers`), and — since
`train_cached.py` never passed `seed` — was shared by every arm at every seed.

Consequence to state once and not relitigate: **the confirmatory seed-to-seed
std should come out wider than the discovery std**, because discovery variance
excluded the data view. That is the bug being removed, not a change in method.
