#!/usr/bin/env python
"""Merge sharded lung-mask HDF5s into one, and re-check the containment gate.

The shards exist because h5py has no safe concurrent writer, not because the
split means anything -- so this puts them back together and, importantly,
re-evaluates the gate on the *pooled* set. A per-shard PASS says nothing about
the whole: eight shards can each clear 0.95 on their own subset while a handful
of catastrophic series hide in the aggregate.

    python scripts/merge_lung_masks.py --shards 'data/lidc/lung_masks_shard*.h5' \\
        --out data/lidc/lung_masks.h5
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/lidc/lung_masks_shard*.h5")
    ap.add_argument("--stats-shards", default="runs/lung_mask_shard*.json")
    ap.add_argument("--out", default="data/lidc/lung_masks.h5")
    ap.add_argument("--stats-out", default="runs/lung_mask_stats.json")
    ap.add_argument("--min-containment", type=float, default=0.95)
    args = ap.parse_args()

    shards = sorted(glob.glob(args.shards))
    if not shards:
        print(f"[merge] no shards matched {args.shards}", file=sys.stderr)
        return 1
    print(f"[merge] {len(shards)} shards")

    seen: dict[str, str] = {}
    with h5py.File(args.out, "w") as out:
        for sh in shards:
            with h5py.File(sh, "r") as f:
                for uid in f.keys():
                    if uid in seen:
                        # A disjoint stride cover should make this impossible; if
                        # it happens the shard arithmetic is wrong and silently
                        # keeping one copy would hide it.
                        print(f"[merge] DUPLICATE {uid} in {sh} and {seen[uid]}",
                              file=sys.stderr)
                        return 1
                    seen[uid] = sh
                    f.copy(f[uid], out, name=uid)
            print(f"[merge]   {sh}: {len(seen)} cumulative")
    print(f"[merge] wrote {args.out} with {len(seen)} series")

    per_series: dict[str, dict] = {}
    for sp in sorted(glob.glob(args.stats_shards)):
        try:
            per_series.update(json.loads(Path(sp).read_text()).get("per_series", {}))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[merge] could not read {sp}: {e}", file=sys.stderr)

    if not per_series:
        print("[merge] no containment stats found; gate not re-checked",
              file=sys.stderr)
        return 0

    methods = {m for v in per_series.values() for m in v}
    if len(methods) != 1:
        # A production run records exactly one method. More than one means these
        # shards came from --compare-methods, and pooling them would silently mix
        # masks built by different rules into one store.
        print(f"[merge] shards disagree on method: {sorted(methods)}",
              file=sys.stderr)
        return 1
    method = methods.pop()

    scored = [v[method] for v in per_series.values()
              if v[method].get("contained") is not None]
    if not scored:
        print(f"[merge] no series with lesions under {method!r}", file=sys.stderr)
        return 0
    c = np.array([r["contained"] for r in scored], float)
    w = np.array([r["n_lesion_patches"] for r in scored], float)
    agg = float((c * w).sum() / w.sum())

    from slotmil import prereg
    Path(args.stats_out).write_text(json.dumps(prereg.load().stamp({
        "analysis_role": "confirmatory",
        "method": method,
        "n_series": len(c),
        "aggregate_containment": agg,
        "min_per_series_containment": float(c.min()),
        "frac_series_fully_contained": float((c >= 0.999).mean()),
        "n_series_below_gate": int((c < args.min_containment).sum()),
        "per_series": per_series,
    }), indent=2))

    print(f"[merge] pooled containment {agg:.4f} over {len(c)} series "
          f"(min {c.min():.3f}, {int((c < args.min_containment).sum())} below gate)")
    if agg < args.min_containment:
        print(f"[merge] GATE FAILED on the pooled set: {agg:.4f} < "
              f"{args.min_containment}", file=sys.stderr)
        return 1
    print(f"[merge] wrote {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
