#!/usr/bin/env python
"""Derive per-patch lung masks for LIDC, staged against the disk budget.

The radial-bin stratification used so far is a proxy. The rigorous version of
"is the attention inside the lung, or just in the middle of the image?" needs a
real lung mask on the same 16x16 patch grid as the features -- and it has never
been run, because ``data/lidc/dicom/`` is empty: the staged extraction pipeline
deleted the DICOM by design once features and lesion masks were cached, and the
HDF5 keeps only features, the lesion mask and ``slice_index``.

So this re-downloads, computes masks only, and deletes -- the same staged rhythm
as ``download_lidc_staged.py`` but with no DINOv2 pass, so it is network- and
CPU-bound rather than GPU-bound.

Alignment is exact rather than approximate: the stored ``slice_index`` selects
the same slices the features came from, and ``mask_to_patch_grid`` is the same
function that rasterised the lesion mask. The lung mask therefore lands on the
identical grid, and "lesion patch" and "lung patch" are directly comparable.

**The gate.** Every series reports the fraction of its annotated lesion patches
that fall inside the lung mask. If that is not ~1.0 the mask is removing the
targets it is supposed to bound, and the whole control inverts -- see
``lung_mask_for_evaluation`` for why an air threshold does exactly that. Series
below ``--min-containment`` are written with a flag, and the summary refuses to
declare success if the aggregate misses the bar.

    # method selection on a handful of discovery series
    python scripts/lung_mask_lidc.py --limit 12 --compare-methods

    # the real run
    python scripts/lung_mask_lidc.py --method fill_hull
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

METHODS = ("air", "fill", "fill_close", "fill_hull")


def stored_series(cache: str) -> dict[str, np.ndarray]:
    """uid -> slice_index, for every series already in the feature cache."""
    with h5py.File(cache, "r") as f:
        return {k: f[k]["slice_index"][:] for k in f.keys() if "features" in f[k]}


def stored_lesion_patches(cache: str, uid: str) -> np.ndarray:
    with h5py.File(cache, "r") as f:
        return f[uid]["mask"][:]


def done_uids(out_h5: str) -> set[str]:
    """Series that are *complete*, not merely started.

    A group is created, then its dataset, then its attributes. A crash between
    those steps leaves a group that exists and looks finished -- and the failed
    array run produced 828 of exactly that: valid lung masks with no containment
    attributes, which a bare ``set(f.keys())`` would have skipped forever on
    resume, silently shipping a mask store whose gate could never be evaluated.

    So completeness means the dataset *and* the attribute the gate reads. Mirrors
    ``features.extract.cached_uids``, which checks for the dataset rather than
    trusting the group.
    """
    p = Path(out_h5)
    if not p.exists():
        return set()
    try:
        with h5py.File(p, "r") as f:
            return {k for k in f.keys()
                    if "lung" in f[k] and "contained" in f[k].attrs}
    except OSError:
        return set()


def load_prior_stats(path: str) -> dict:
    """Per-series stats from an earlier run, so a killed job is not a total loss.

    Downloads are the expensive part here and this job runs on scavenger, where
    preemption is normal rather than exceptional. An earlier run lost 30 series'
    worth of work to a teardown because everything lived in memory until the
    final write; hence flushing per batch and resuming from what landed.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text()).get("per_series", {})
    except (OSError, json.JSONDecodeError):
        return {}


def write_attrs(group, stats: dict) -> None:
    """Store a per-series stats dict as HDF5 attributes.

    h5py attributes take scalars and arrays, not mappings -- and the containment
    stats grew a nested ``contained_at`` / ``lung_frac_at`` when the threshold
    sweep was added, which made every write in the production path raise. The
    sweep never caught it because ``--compare-methods`` skips the HDF5 write
    entirely, so the two paths had diverged silently.

    ``None`` becomes -1.0 because HDF5 has no null; mappings are JSON-encoded so
    the sweep travels with the mask rather than living only in a sidecar file.
    """
    for k, v in stats.items():
        if isinstance(v, dict):
            group.attrs[k] = json.dumps(v)
        elif v is None:
            group.attrs[k] = -1.0
        else:
            group.attrs[k] = v


def clean_dicom_root(dicom_root: Path) -> None:
    """Remove anything a previous run left behind mid-batch.

    DICOM is only deleted at the end of a batch, so a kill leaves a partial
    staging directory that pylidc would happily half-read on the next run.
    """
    n = 0
    for p in dicom_root.iterdir():
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            n += 1
    if n:
        print(f"[lung] cleared {n} stale DICOM directories from {dicom_root}",
              flush=True)


THRESHOLDS = (0.0, 0.05, 0.1, 0.25, 0.5)


def containment(lung_patch: np.ndarray, lesion_patch: np.ndarray,
                lung_thresh: float = 0.5) -> dict:
    """Fraction of annotated lesion patches that land inside the lung mask.

    The number that decides whether this control is usable at all: the
    restriction removes patches from the candidate pool, so any lesion patch it
    drops is a positive silently deleted from the metric.

    Swept over the "how much of a patch must be lung" cut rather than reported at
    one value, because that cut is a free parameter and the first run set it to
    0.5 for no reason beyond it sounding neutral. A patch is 22 mm across; a
    subpleural nodule -- common in LIDC -- sits in a patch that is part lung and
    part chest wall, so a strict majority rule discards exactly the lesions the
    control is supposed to keep. For an *inclusion* criterion the permissive end
    is the principled one: drop a patch only when it contains no lung at all.
    """
    lesion = lesion_patch > 0
    n_les = int(lesion.sum())
    out = {
        "n_lesion_patches": n_les,
        "contained_at": {},
        "lung_frac_at": {},
    }
    for t in THRESHOLDS:
        in_lung = lung_patch > t
        out["lung_frac_at"][str(t)] = float(in_lung.mean())
        out["contained_at"][str(t)] = (
            None if n_les == 0 else float((in_lung & lesion).sum()) / n_les
        )
    out["contained"] = out["contained_at"][str(lung_thresh)]
    out["lung_frac"] = out["lung_frac_at"][str(lung_thresh)]
    return out


def process(uid, meta_row, slice_index, cache, dicom_root, methods, lung_thresh):
    """Download one series, build every requested mask, return patch grids + stats."""
    from slotmil.data.lidc import arrange_for_pylidc, download_series, load_scan_and_masks
    from slotmil.features.ct_preprocess import lung_mask_for_evaluation, mask_to_patch_grid

    flat = dicom_root / f"_staging_{uid}"
    download_series(uid, flat)
    arranged = arrange_for_pylidc(
        flat, meta_row["PatientID"], meta_row["StudyInstanceUID"], uid)
    shutil.rmtree(flat, ignore_errors=True)

    volume, _, _ = load_scan_and_masks(uid)
    lesion_patch = stored_lesion_patches(cache, uid)

    grids, stats = {}, {}
    for m in methods:
        lung3d = lung_mask_for_evaluation(volume, method=m)
        sel = lung3d[slice_index].astype(np.float32)
        lp = mask_to_patch_grid(sel, lesion_patch.shape[-1], mode="mean")
        grids[m] = lp
        stats[m] = containment(lp, lesion_patch, lung_thresh)
    return grids, stats, arranged


def summarise(methods, all_stats, min_containment: float) -> dict:
    """Aggregate containment per method.

    Weighted by lesion patches, because a series with one small nodule and a
    series with twenty should not count equally toward "does this mask keep the
    targets". ``min_per_series`` is reported alongside: a good average with one
    catastrophic series is still a broken control.
    """
    summary = {}
    for m in methods:
        rows = [v[m] for v in all_stats.values()
                if m in v and v[m]["contained"] is not None]
        if not rows:
            continue
        weights = np.array([r["n_lesion_patches"] for r in rows], float)
        contained = np.array([r["contained"] for r in rows], float)
        summary[m] = {
            "aggregate_containment": float((contained * weights).sum() / weights.sum()),
            "mean_per_series_containment": float(contained.mean()),
            "min_per_series_containment": float(contained.min()),
            "frac_series_fully_contained": float((contained >= 0.999).mean()),
            "frac_series_above_min": float((contained >= min_containment).mean()),
            "mean_lung_frac": float(np.mean([r["lung_frac"] for r in rows])),
            "n_series": len(rows),
            "by_threshold": {},
        }
        # The lung_thresh sweep is the point of the comparison: a method is only
        # usable where containment is ~1, and among those the smallest lung_frac
        # wins, because a looser mask restricts less and therefore proves less.
        for t in rows[0].get("contained_at", {}):
            c = np.array([r["contained_at"][t] for r in rows], float)
            lf = np.array([r["lung_frac_at"][t] for r in rows], float)
            summary[m]["by_threshold"][t] = {
                "aggregate_containment": float((c * weights).sum() / weights.sum()),
                "min_per_series_containment": float(c.min()),
                "frac_series_fully_contained": float((c >= 0.999).mean()),
                "mean_lung_frac": float(lf.mean()),
            }
    return summary


def write_stats(args, methods, all_stats) -> dict:
    from slotmil import prereg

    summary = summarise(methods, all_stats, args.min_containment)
    Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_out).write_text(json.dumps(prereg.load().stamp({
        # Method selection runs on non-confirmatory series by construction
        # (--exclude-from), so it never tunes the mask on data carrying H8.
        "analysis_role": "exploratory",
        "method": None if args.compare_methods else args.method,
        "lung_thresh": args.lung_thresh,
        "min_containment": args.min_containment,
        "excluded_from": args.exclude_from,
        "summary": summary,
        "per_series": all_stats,
    }), indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/lidc/features_dinov2_vitb14.h5")
    ap.add_argument("--out", default="data/lidc/lung_masks.h5")
    ap.add_argument("--stats-out", default="runs/lung_mask_stats.json")
    ap.add_argument("--dicom-root", default="data/lidc/dicom")
    ap.add_argument("--method", default="fill_hull", choices=METHODS)
    ap.add_argument("--compare-methods", action="store_true",
                    help="build every method and report containment for each; for "
                         "choosing the method on discovery series, not for the run")
    ap.add_argument("--lung-thresh", type=float, default=0.0,
                    help="a patch counts as lung when more than this fraction of "
                         "it is lung. Default 0.0 = 'contains any lung at all'. "
                         "This is an INCLUSION criterion for an evaluation region, "
                         "so the permissive end is correct: a stricter cut drops "
                         "boundary patches, and subpleural nodules live in exactly "
                         "those. Every value in THRESHOLDS is reported regardless.")
    ap.add_argument("--save-grids", default=None, metavar="H5",
                    help="persist the per-method lung grids, so threshold and "
                         "method questions can be re-answered without re-downloading")
    ap.add_argument("--min-containment", type=float, default=0.95,
                    help="aggregate lesion-patch containment required to pass")
    ap.add_argument("--batch-series", type=int, default=25)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--exclude-from", default=None, metavar="SPLITS.JSON",
                    help="drop series listed under --exclude-key in this splits "
                         "file. Use it when selecting the mask method, so the "
                         "choice is never tuned on confirmatory test series.")
    ap.add_argument("--exclude-key", default="test")
    ap.add_argument("--sample-seed", type=int, default=None,
                    help="shuffle before --limit, so a subset is a sample rather "
                         "than whatever sorts first by UID")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1,
                    help="split the series list across a SLURM array. One pass "
                         "over 999 series is ~24h, which is the walltime limit "
                         "and SLURM does not requeue on TIMEOUT -- 8 shards make "
                         "it ~3h. Each shard needs its own --out and --stats-out; "
                         "h5py has no safe concurrent writer.")
    args = ap.parse_args()

    if not 0 <= args.shard_index < args.shard_count:
        print(f"[lung] shard-index {args.shard_index} out of range for "
              f"shard-count {args.shard_count}", file=sys.stderr)
        return 2

    dicom_root = Path(args.dicom_root)
    dicom_root.mkdir(parents=True, exist_ok=True)
    if args.shard_count == 1:
        clean_dicom_root(dicom_root)
    else:
        print(f"[lung] shard {args.shard_index}/{args.shard_count}: skipping the "
              f"startup sweep of {dicom_root}, siblings are using it", flush=True)

    from slotmil.data.lidc import configure_pylidc, free_gb, list_ct_series
    configure_pylidc(dicom_root)

    methods = list(METHODS) if args.compare_methods else [args.method]
    todo_index = stored_series(args.cache)
    all_stats = load_prior_stats(args.stats_out)
    already = set(all_stats) if args.compare_methods else done_uids(args.out)
    print(f"[lung] {len(todo_index)} series in cache, {len(already)} already done",
          flush=True)

    series = {s["SeriesInstanceUID"]: s for s in list_ct_series()}
    candidates = [u for u in todo_index if u in series]

    if args.exclude_from:
        held = set(json.loads(Path(args.exclude_from).read_text())[args.exclude_key])
        before = len(candidates)
        candidates = [u for u in candidates if u not in held]
        print(f"[lung] excluded {before - len(candidates)} series in "
              f"{args.exclude_from}:{args.exclude_key}", flush=True)

    # Sample and truncate BEFORE removing what is already done, so a resumed run
    # targets the same set rather than drawing 30 fresh series each time.
    if args.sample_seed is not None:
        np.random.default_rng(args.sample_seed).shuffle(candidates)
    if args.limit:
        candidates = candidates[: args.limit]

    # Stride rather than contiguous blocks, so every shard gets a mix of large
    # and small series and they finish at roughly the same time.
    if args.shard_count > 1:
        candidates = candidates[args.shard_index :: args.shard_count]

    todo = [u for u in candidates if u not in already]
    print(f"[lung] {len(candidates)} in scope, {len(todo)} still to do, "
          f"methods={methods}", flush=True)
    if not todo:
        print("[lung] nothing to do")
        return 0

    n_ok = n_fail = 0
    for start in range(0, len(todo), args.batch_series):
        chunk = todo[start : start + args.batch_series]
        free = free_gb(dicom_root)
        if free < args.min_free_gb:
            print(f"[lung] ABORT: {free:.1f} GB free < {args.min_free_gb}",
                  file=sys.stderr)
            break
        print(f"\n[lung] batch {start // args.batch_series + 1} "
              f"({len(chunk)} series, {free:.1f} GB free)", flush=True)

        for uid in chunk:
            try:
                grids, stats, arranged = process(
                    uid, series[uid], todo_index[uid], args.cache,
                    dicom_root, methods, args.lung_thresh)
                # Delete this series' DICOM immediately rather than at end of
                # batch. Peak transient drops from a whole batch (~4.5 GB) to one
                # series (~180 MB), and -- the reason it matters for sharding --
                # concurrent array tasks never touch each other's directories.
                shutil.rmtree(arranged, ignore_errors=True)
                if args.save_grids:
                    # The download is the expensive part; keeping the grids means
                    # any later question about thresholds or methods is answered
                    # offline instead of by re-fetching 30 series from TCIA.
                    with h5py.File(args.save_grids, "a") as f:
                        g = f.require_group(uid)
                        for m, arr in grids.items():
                            if m in g:
                                del g[m]
                            g.create_dataset(m, data=arr.astype(np.float16),
                                             compression="lzf")
                if not args.compare_methods:
                    with h5py.File(args.out, "a") as f:
                        if uid in f:
                            del f[uid]
                        g = f.create_group(uid)
                        g.create_dataset("lung", data=grids[args.method],
                                         compression="lzf")
                        g.attrs["method"] = args.method
                        g.attrs["lung_thresh"] = args.lung_thresh
                        write_attrs(g, stats[args.method])
                all_stats[uid] = stats
                n_ok += 1
                s = stats[methods[0]]
                c = "n/a" if s["contained"] is None else f"{s['contained']:.3f}"
                print(f"[lung]   ok {uid[-12:]}  contained={c}  "
                      f"lung_frac={s['lung_frac']:.3f}", flush=True)
            except Exception as e:
                n_fail += 1
                print(f"[lung]   FAILED {uid}: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)

        # No whole-root sweep here: with a sharded array every task shares this
        # directory (pylidc reads a single global ~/.pylidcrc, so per-shard roots
        # would race), and deleting everything would pull the floor out from
        # under a sibling mid-download. Per-series cleanup above covers it.

        # Flush after every batch. Downloads are the expensive part and this runs
        # on a preemptible queue, so losing a batch must not mean losing the run.
        write_stats(args, methods, all_stats)
        print(f"[lung] batch flushed: {len(all_stats)} series in {args.stats_out}",
              flush=True)

    # ------------------------------------------------------------ verdict
    print(f"\n[lung] {n_ok} ok, {n_fail} failed")
    summary = write_stats(args, methods, all_stats)
    for m in methods:
        if m not in summary:
            continue
        s = summary[m]
        print(f"[lung] {m:11s} containment {s['aggregate_containment']:.4f} "
              f"(min/series {s['min_per_series_containment']:.3f}, "
              f"fully contained {s['frac_series_fully_contained']:.2f}, "
              f"lung covers {s['mean_lung_frac']:.3f} of grid)  "
              f"{'PASS' if s['aggregate_containment'] >= args.min_containment else 'FAIL'}")
    print(f"[lung] wrote {args.stats_out}")

    if not args.compare_methods:
        agg = summary.get(args.method, {}).get("aggregate_containment", 0.0)
        if agg < args.min_containment:
            print(f"[lung] GATE FAILED: containment {agg:.4f} < "
                  f"{args.min_containment}. The mask is deleting the targets it "
                  f"is meant to bound; do not use it as a control.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
