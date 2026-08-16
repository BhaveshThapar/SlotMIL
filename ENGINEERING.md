# SlotMIL — working notes

Contracts and procedures only. **Results live in `RESULTS.md`; do not duplicate
them here.** A second copy of a number goes stale silently, and this file is
loaded into every session, so a stale line here is worse than no line.

Run everything through the venv: `source .venv/bin/activate`. Tests need
`python -m pytest`, not bare `pytest` — `tests/test_lung_mask_io.py` imports from
`scripts/`, which has no `__init__.py`, so the repo root must be on `sys.path`.

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
  *slot* axis; ABMIL, gated ABMIL, MH-ABMIL and `centre_gaussian` softmax over the
  *instance* axis. This is not cosmetic: `localization.instance_auc(slot=None)`
  takes a max over slots and is **invalid** for slot-normalised attention — it
  reads 0.4885 where the frozen-slot path reads 0.8423. That discrepancy is the
  project's most expensive past bug. Any new arm must document which axis it
  normalises on.

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
`scripts/slurm/lidc_train_array.sbatch` (`ARMS=(...)` **and** `--array=0-N`).

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
  which `PREREGISTRATION.md` line 189 says decides whether it counts.
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
