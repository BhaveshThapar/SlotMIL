#!/usr/bin/env python
"""Staged LIDC-IDRI download -> mask extraction -> feature cache -> delete.

LIDC is 1,018 CT series / 243,958 slices / ~124 GB of DICOM (measured against the
TCIA NBIA API), and scratch has ~150 GB. Holding the raw collection and a feature
cache simultaneously does not fit, so this processes in batches and deletes each
batch's DICOM once its features and masks are safely in the cache. Peak transient
usage stays around 15 GB.

Ordering is not negotiable::

    download -> pylidc masks -> features -> h5 -> DELETE

pylidc builds a nodule mask by placing annotation contours into the volume using
per-slice z positions read from the DICOM headers. Those headers are not in its
bundled annotation database. Delete the DICOM before extracting masks and the
localisation ground truth is simply gone -- 124 GB of re-download to recover.

Resumable: series already in the cache are skipped, so preemption or a wall-clock
limit costs one batch, not the run.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/fs/nexus-scratch/bthapar/SlotMIL/data/lidc")
    ap.add_argument("--cache", default="/fs/nexus-scratch/bthapar/SlotMIL/data/lidc/features_dinov2_vitb14.h5")
    ap.add_argument("--backbone", default="dinov2_vitb14")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-series", type=int, default=25, help="series per download/delete cycle")
    ap.add_argument("--extract-batch", type=int, default=32, help="slices per GPU forward")
    ap.add_argument("--limit", type=int, default=None, help="stop after N series (smoke test)")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--label-mode", default="nodule_present", choices=["nodule_present", "malignancy"])
    ap.add_argument("--consensus-level", type=float, default=0.5)
    ap.add_argument("--no-lung-filter", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_root = Path(args.out)
    dicom_root = out_root / "dicom"
    dicom_root.mkdir(parents=True, exist_ok=True)

    # pylidc reads ~/.pylidcrc at import time, so configure before importing it.
    from slotmil.data.lidc import (
        arrange_for_pylidc,
        configure_pylidc,
        download_series,
        free_gb,
        list_ct_series,
        load_scan_and_masks,
        scan_label,
    )

    configure_pylidc(dicom_root)

    from slotmil.features.backbones import build_backbone
    from slotmil.features.extract import cache_size_report, cached_uids, extract_volume, write_series

    print(f"[lidc] querying TCIA for LIDC-IDRI CT series ...", flush=True)
    series = list_ct_series()
    print(f"[lidc] {len(series)} CT series available", flush=True)

    done = cached_uids(args.cache)
    todo = [s for s in series if s["SeriesInstanceUID"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[lidc] {len(done)} already cached, {len(todo)} to process", flush=True)
    if not todo:
        print("[lidc] nothing to do")
        return

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[lidc] loading {args.backbone} on {device} ...", flush=True)
    backbone = build_backbone(args.backbone, image_size=args.image_size).to(device).eval()
    grid = backbone.grid
    print(f"[lidc] patch grid {grid}x{grid}, dim {backbone.dim}", flush=True)

    n_ok = n_fail = 0
    for start in range(0, len(todo), args.batch_series):
        chunk = todo[start : start + args.batch_series]

        free = free_gb(out_root)
        if free < args.min_free_gb:
            print(
                f"[lidc] ABORT: {free:.1f} GB free < --min-free-gb {args.min_free_gb}. "
                f"Cache so far: {cache_size_report(args.cache)}",
                file=sys.stderr,
            )
            break

        print(
            f"\n[lidc] batch {start//args.batch_series + 1} "
            f"({len(chunk)} series, {free:.1f} GB free)",
            flush=True,
        )

        staged = []
        for s in chunk:
            uid = s["SeriesInstanceUID"]
            try:
                flat = dicom_root / f"_staging_{uid}"
                download_series(uid, flat)
                arrange_for_pylidc(
                    flat, s["PatientID"], s["StudyInstanceUID"], uid
                )
                shutil.rmtree(flat, ignore_errors=True)
                staged.append(s)
            except Exception as e:
                n_fail += 1
                print(f"[lidc]   download FAILED {uid}: {e}", file=sys.stderr, flush=True)

        for s in staged:
            uid = s["SeriesInstanceUID"]
            try:
                # Masks first -- they need the DICOM that is about to be deleted.
                volume, mask, meta = load_scan_and_masks(
                    uid, consensus_level=args.consensus_level
                )
                label = scan_label(meta, mode=args.label_mode)
                if label is None:
                    print(f"[lidc]   skip {uid}: indeterminate under {args.label_mode}", flush=True)
                    continue

                feats, slice_idx = extract_volume(
                    backbone, volume, device=device,
                    batch_size=args.extract_batch,
                    image_size=args.image_size,
                    restrict_to_lung=not args.no_lung_filter,
                )
                write_series(
                    args.cache, uid, feats, label, slice_idx, grid,
                    lesion_mask=mask[slice_idx],
                    extra_attrs={
                        "patient_id": meta["patient_id"],
                        "n_nodules": meta["n_nodules"],
                        "slice_thickness": meta["slice_thickness"],
                        "pixel_spacing": meta["pixel_spacing"],
                        "backbone": args.backbone,
                        "label_mode": args.label_mode,
                    },
                )
                n_ok += 1
                print(
                    f"[lidc]   ok {uid}  slices {volume.shape[0]}->{len(slice_idx)}  "
                    f"nodules={meta['n_nodules']}  label={label}",
                    flush=True,
                )
            except Exception as e:
                n_fail += 1
                print(f"[lidc]   process FAILED {uid}: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)

        # Batch is cached; reclaim the DICOM.
        for p in dicom_root.iterdir():
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)

        print(f"[lidc] batch done. cache: {cache_size_report(args.cache)}", flush=True)

    print(f"\n[lidc] finished: {n_ok} cached, {n_fail} failed")
    print(f"[lidc] {cache_size_report(args.cache)}")
    print(f"[lidc] free space: {free_gb(out_root):.1f} GB")


if __name__ == "__main__":
    main()
