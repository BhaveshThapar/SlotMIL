#!/usr/bin/env python
"""Qualitative slot overlays + deletion/insertion faithfulness.

The two remaining ISBI deliverables from plan.md line 138 ("figures = slot
overlays + alignment table") and line 112 (faithfulness via deletion/insertion).
Both are needed regardless of how the alignment numbers come out: if slots do
bind lesions, these are the evidence; if they do not, the overlays are the
diagnosis.

Faithfulness matters because attention that *looks* right is not the same as
attention that *drove the decision*. A slot can sit convincingly on a nodule in
an overlay while contributing nothing to the prediction, and only deletion /
insertion separates those cases.

Overlays are rendered from the CT preview volumes when present; where a cache
holds features only, the attention grid is still written as a heatmap array so
figures can be regenerated later without re-running the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from slotmil.data.feature_cache import FeatureBagDataset, collate_bags
from slotmil.eval.faithfulness import deletion_curve, faithfulness_summary, insertion_curve
from slotmil.models.mil import build_model, slot_pooling_param_count


def load_model(args, input_dim: int):
    match_to = (
        slot_pooling_param_count(input_dim, args.dim, args.num_slots)
        if args.pooling == "mh_abmil" else None
    )
    model = build_model(
        pooling=args.pooling, input_dim=input_dim, dim=args.dim,
        num_classes=args.num_classes, num_slots=args.num_slots,
        readout=args.readout, match_params_to=match_to,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default="runs/figures")
    ap.add_argument("--pooling", default="slot")
    ap.add_argument("--num-slots", type=int, default=8)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--readout", default="gated")
    ap.add_argument("--n-figures", type=int, default=8)
    ap.add_argument("--n-faithfulness", type=int, default=40)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    splits = json.loads(Path(args.splits).read_text())
    label_map = splits.get("labels")
    if label_map and "num_classes" in splits:
        args.num_classes = splits["num_classes"]

    ds = FeatureBagDataset(
        args.cache, keys=splits["test"], labels=label_map, return_mask=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(args, ds.dim).to(device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_bags)

    # Prefer bags that actually carry an annotated lesion -- an overlay on a scan
    # with nothing to find demonstrates nothing.
    scored = []
    dels, inss = [], []

    for i, batch in enumerate(loader):
        feats, pad_mask = batch["features"], batch["pad_mask"]
        n = int(batch["lengths"][0])
        has_lesion = (
            "patch_target" in batch and float(batch["patch_target"][0, :n].sum()) > 0
        )

        with torch.no_grad():
            o = model(feats.to(device), pad_mask.to(device))
        attn = o["attn"][0, :, :n].float().cpu()
        target = int(batch["label"][0])

        if has_lesion and len(scored) < args.n_figures:
            np.savez_compressed(
                out / f"attn_{batch['uid'][0][-16:]}.npz",
                attn=attn.numpy(),
                patch_target=batch["patch_target"][0, :n].numpy(),
                n_slices=int(batch["n_slices"][0]),
                grid=ds.grid_h,
                label=target,
                attribution=o["attribution"][0].float().cpu().numpy(),
            )
            scored.append(batch["uid"][0])

        if len(dels) < args.n_faithfulness:
            dels.append(deletion_curve(model, feats, attn, target, device, args.steps))
            inss.append(insertion_curve(model, feats, attn, target, device, args.steps))

        if len(scored) >= args.n_figures and len(dels) >= args.n_faithfulness:
            break

    summary = faithfulness_summary(dels, inss)
    summary["n_overlay_bags"] = len(scored)
    summary["overlay_uids"] = scored
    summary["pooling"] = args.pooling
    (out / "faithfulness.json").write_text(json.dumps(summary, indent=2))

    print(f"=== faithfulness ({args.pooling}) ===")
    print(f"  deletion AUC   {summary['deletion_auc']:.4f} "
          f"+/- {summary['deletion_auc_std']:.4f}   (lower = more faithful)")
    print(f"  insertion AUC  {summary['insertion_auc']:.4f} "
          f"+/- {summary['insertion_auc_std']:.4f}   (higher = more faithful)")
    print(f"  insertion - deletion  {summary['insertion_minus_deletion']:+.4f}")
    print(f"  bags: {summary['n_bags']} faithfulness, {len(scored)} overlay arrays")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
