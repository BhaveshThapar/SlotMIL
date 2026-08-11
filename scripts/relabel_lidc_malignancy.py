#!/usr/bin/env python
"""Recompute LIDC bag labels as malignancy, without re-extracting features.

Why this exists: the LIDC cache was built with `nodule_present`, which is true
for 87% of bags. A label almost every bag shares exerts almost no pressure to
localise anything, and that is the leading confound for the null localisation
result (see RESULTS.md). Malignancy is balanced and cannot be predicted without
characterising the nodule.

The features do not change -- only the label -- so there is no need to repeat the
9-hour extraction. Malignancy ratings live in pylidc's bundled annotation
database rather than in the DICOM headers, so this works even though the staged
pipeline deleted the DICOM.

Protocol (the standard one): each nodule's malignancy is the median of its
radiologists' 1-5 ratings; nodules with median exactly 3 are indeterminate and
excluded; a scan is positive if any remaining nodule is >3, negative if all are
<3, and dropped if nothing decisive remains.

Output is a label map consumable by make_splits.py --labels-from.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

INDETERMINATE = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/fs/nexus-scratch/bthapar/SlotMIL/data/lidc/features_dinov2_vitb14.h5")
    ap.add_argument("--dicom-root", default="/fs/nexus-scratch/bthapar/SlotMIL/data/lidc/dicom")
    ap.add_argument("--out", default="/fs/nexus-scratch/bthapar/SlotMIL/data/lidc/labels_malignancy.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from slotmil.data.lidc import _restore_numpy_aliases, configure_pylidc

    configure_pylidc(args.dicom_root)
    _restore_numpy_aliases()
    import pylidc as pl

    with h5py.File(args.cache, "r") as f:
        uids = sorted(f.keys())
    if args.limit:
        uids = uids[: args.limit]
    print(f"[relabel] {len(uids)} cached series", flush=True)

    labels: dict[str, int] = {}
    dropped: dict[str, str] = {}
    t0 = time.time()

    for i, uid in enumerate(uids):
        scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == uid).first()
        if scan is None:
            dropped[uid] = "not in pylidc DB"
            continue
        try:
            clusters = scan.cluster_annotations()
        except Exception as e:  # noqa: BLE001
            dropped[uid] = f"cluster failed: {type(e).__name__}"
            continue

        medians = [float(np.median([a.malignancy for a in c])) for c in clusters]
        decisive = [m for m in medians if m != INDETERMINATE]
        if not decisive:
            dropped[uid] = "no nodules" if not medians else "all indeterminate"
            continue
        labels[uid] = int(any(m > INDETERMINATE for m in decisive))

        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"[relabel] {i+1}/{len(uids)}  kept={len(labels)} dropped={len(dropped)}"
                  f"  {el/(i+1):.2f}s/scan  eta {(len(uids)-i-1)*el/(i+1)/60:.0f}min",
                  flush=True)

    dist = Counter(labels.values())
    reasons = Counter(dropped.values())
    print(f"\n[relabel] kept {len(labels)} / {len(uids)}")
    print(f"[relabel] label balance: {dict(sorted(dist.items()))} "
          f"({100*dist[1]/max(len(labels),1):.1f}% malignant)")
    print(f"[relabel] dropped {len(dropped)}: {dict(reasons)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"labels": labels, "dropped": dropped, "mode": "malignancy",
         "balance": {str(k): v for k, v in dist.items()}}, indent=2))
    print(f"[relabel] wrote {args.out}")

    if len(labels) < 100:
        print("[relabel] WARNING: fewer than 100 usable scans; the malignancy "
              "experiment may be underpowered", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
