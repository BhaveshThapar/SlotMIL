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
