#!/usr/bin/env python
"""MosMed feature extraction into the HDF5 cache.

Simpler than the LIDC pipeline: the volumes are already NIfTI on disk, so there
is no download/delete staging. Two MosMed-specific choices:

**Lung filtering is off by default.** The public release keeps only every 10th
slice (~8 mm spacing), giving ~38 slices per volume. Dropping any of those costs
z-coverage the bags cannot spare, and the storage saving is negligible at this
scale.

**Masks are written for the 50 annotated scans only.** The other 1,060 get
features and a severity label but no localisation target, which is what
``FeatureBagDataset(return_mask=True)`` expects -- it skips groups without a
mask rather than fabricating an empty one.
"""

from __future__ import annotations

import argparse
import sys
import traceback

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/fs/nexus-scratch/bthapar/SlotMIL/data/mosmed/full")
    ap.add_argument("--cache", default="/fs/nexus-scratch/bthapar/SlotMIL/data/mosmed/features_dinov2_vitb14.h5")
    ap.add_argument("--backbone", default="dinov2_vitb14")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--extract-batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lung-filter", action="store_true",
                    help="off by default; the every-10th-slice release cannot spare slices")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from slotmil.data.mosmed import MosMedIndex, load_mask, load_volume
    from slotmil.features.backbones import build_backbone
    from slotmil.features.extract import (
        cache_size_report,
        cached_uids,
        extract_volume,
        write_series,
    )

    ix = MosMedIndex(args.root)
    print(f"[mosmed] {ix.summary()}", flush=True)

    done = cached_uids(args.cache)
    todo = [u for u in sorted(ix.volumes) if u not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[mosmed] {len(done)} cached, {len(todo)} to process", flush=True)
    if not todo:
        return

    device = args.device if torch.cuda.is_available() else "cpu"
    backbone = build_backbone(args.backbone, image_size=args.image_size).to(device).eval()
    grid = backbone.grid
    print(f"[mosmed] {args.backbone} on {device}, grid {grid}x{grid}, dim {backbone.dim}", flush=True)

    n_ok = n_mask = n_fail = 0
    for i, uid in enumerate(todo):
        try:
            info = ix.volumes[uid]
            volume = load_volume(info["path"])

            mask = None
            if uid in ix.masks:
                mask = load_mask(ix.masks[uid])
                if mask.shape != volume.shape:
                    raise ValueError(
                        f"mask {mask.shape} != volume {volume.shape}"
                    )

            feats, slice_idx = extract_volume(
                backbone, volume, device=device,
                batch_size=args.extract_batch,
                image_size=args.image_size,
                restrict_to_lung=args.lung_filter,
            )
            write_series(
                args.cache, uid, feats, info["label"], slice_idx, grid,
                lesion_mask=None if mask is None else mask[slice_idx],
                extra_attrs={
                    "severity": info["severity"],
                    "backbone": args.backbone,
                    "has_mask": mask is not None,
                },
            )
            n_ok += 1
            n_mask += mask is not None
            if (i + 1) % 100 == 0 or mask is not None:
                print(
                    f"[mosmed] {i+1}/{len(todo)} {uid} {info['severity']} "
                    f"slices={len(slice_idx)}" + ("  +mask" if mask is not None else ""),
                    flush=True,
                )
        except Exception as e:
            n_fail += 1
            print(f"[mosmed] FAILED {uid}: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)

    print(f"\n[mosmed] done: {n_ok} cached ({n_mask} with masks), {n_fail} failed")
    print(f"[mosmed] {cache_size_report(args.cache)}")


if __name__ == "__main__":
    main()
