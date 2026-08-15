#!/usr/bin/env python
"""Stage 1 of the null-control battery: cache slot attention + patch masks to disk.

Running the model is the only part that needs a GPU and the 78GB feature cache.
Every null (random attention, label permutation, selection bias) is then a pure
numpy operation on these arrays, so they can be iterated on without re-reading
8GB of features per variant.

Writes, per (model, split), a single .npz holding attention concatenated along
the instance axis plus the per-bag lengths needed to slice it back apart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from slotmil.data.feature_cache import FeatureBagDataset, collate_bags
from slotmil.models.mil import build_model, slot_pooling_param_count


@torch.no_grad()
def collect(model, ds, device, workers=4, batch_size=4, dtype=np.float16):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                        collate_fn=collate_bags)
    model.eval().to(device)
    attns, masks, lens, uids = [], [], [], []
    for bi, b in enumerate(loader):
        a = model(b["features"].to(device), b["pad_mask"].to(device))["attn"].float().cpu().numpy()
        for i, n in enumerate(b["lengths"].tolist()):
            attns.append(a[i, :, :n].astype(dtype))
            masks.append((b["patch_target"][i, :n].numpy() > 0).astype(np.uint8))
            lens.append(n)
            uids.append(b["uid"][i])
        if bi % 10 == 0:
            print(f"  batch {bi} ({len(attns)} bags)", flush=True)
    return attns, masks, lens, uids


def save(path, attns, masks, lens, uids):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        attn=np.concatenate(attns, axis=1),      # K, sum(N)
        mask=np.concatenate(masks, axis=0),      # sum(N)
        lengths=np.asarray(lens, dtype=np.int64),
        uids=np.asarray(uids),
    )
    print(f"wrote {path}  K={attns[0].shape[0]} bags={len(lens)} total_N={sum(lens)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/lidc/features_dinov2_vitb14.h5")
    ap.add_argument("--splits", default="data/lidc/splits.json")
    ap.add_argument("--ckpt", default=None, help="omit for the untrained-model null")
    ap.add_argument("--pooling", default="slot",
                    choices=["slot", "mh_abmil", "centre_gaussian",
                             "normal_guidance"])
    ap.add_argument("--init-seed", type=int, default=1234,
                    help="torch seed for the untrained-model null")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="runs/nulls")
    ap.add_argument("--num-slots", type=int, default=8)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                    help="float16 halves the cache but quantises very peaked "
                         "attention; use float32 to check a seed whose AUC does "
                         "not reproduce the float32 reeval number")
    args = ap.parse_args()
    dtype = np.float16 if args.dtype == "float16" else np.float32

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = json.loads(Path(args.splits).read_text())
    lm = sp.get("labels")

    def sub(keys):
        p = FeatureBagDataset(args.cache, keys=keys, labels=lm, return_mask=True)
        return FeatureBagDataset(args.cache, keys=p.masked_keys(), labels=lm, return_mask=True)

    val_ds, test_ds = sub(sp["val"]), sub(sp["test"])
    print(f"val bags {len(val_ds)}  test bags {len(test_ds)}  dim {val_ds.dim}", flush=True)

    torch.manual_seed(args.init_seed)
    match_to = (slot_pooling_param_count(val_ds.dim, 256, args.num_slots)
                if args.pooling == "mh_abmil" else None)
    model = build_model(pooling=args.pooling, input_dim=val_ds.dim, dim=256,
                        num_classes=sp.get("num_classes", 2), num_slots=args.num_slots,
                        readout="gated", match_params_to=match_to)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        print(f"loaded {args.ckpt}", flush=True)
    else:
        print(f"UNTRAINED model, torch seed {args.init_seed}", flush=True)

    for split, ds in (("val", val_ds), ("test", test_ds)):
        print(f"[{args.tag}] collecting {split}", flush=True)
        save(Path(args.outdir) / f"{args.tag}_{split}.npz",
             *collect(model, ds, device, dtype=dtype))


if __name__ == "__main__":
    main()
