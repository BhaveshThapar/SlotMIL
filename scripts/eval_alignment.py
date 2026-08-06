#!/usr/bin/env python
"""Slot-to-finding alignment evaluation -- the paper's central experiment.

plan.md line 135 (W6-7, "the money experiment") and recommendation #1: the
defensible moat is quantitative slot-to-radiological-finding alignment measured
against public lesion masks, because Slot-MIL and INSIGHT already occupy the
neighbouring claims.

The protocol enforces the answer to reviewer objection #3 ("you post-hoc named
the slots"):

    1. compute slot-finding affinity on VALIDATION
    2. freeze the Hungarian slot -> finding assignment
    3. only then touch TEST, scoring under the frozen assignment

Because the assignment is an argument to the test-time call rather than
recomputed there, naming demonstrably precedes test.

The same numbers are produced for the parameter-matched multi-head-ABMIL control,
which is what separates "competition causes specialisation" from "extra capacity
causes specialisation" (objection #1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from slotmil.data.feature_cache import FeatureBagDataset, collate_bags
from slotmil.eval.alignment import (
    apply_slot_assignment,
    fit_slot_assignment,
    head_redundancy,
    slot_consistency,
    slot_finding_affinity,
    slot_purity,
)
from slotmil.eval.localization import evaluate_localization


@torch.no_grad()
def collect(model, ds, device: str, batch_size: int = 4, num_workers: int = 4):
    """Run the model over a split, returning per-bag attention and mask targets."""
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_bags,
    )
    model.eval().to(device)

    attns, masks, n_slices = [], [], []
    for batch in loader:
        out = model(batch["features"].to(device), batch["pad_mask"].to(device))
        attn = out["attn"].float().cpu().numpy()
        for i, length in enumerate(batch["lengths"].tolist()):
            attns.append(attn[i, :, :length])
            masks.append(batch["patch_target"][i, :length].numpy())
            n_slices.append(int(batch["n_slices"][i]))
    return attns, masks, n_slices


def binary_finding_masks(masks: list[np.ndarray], threshold: float = 0.0) -> list[np.ndarray]:
    """Patch-grid lesion fractions -> a 2-row (lesion / background) finding matrix.

    LIDC annotates one finding class (nodule), so the "findings" axis is
    lesion-vs-background. MosMed supplies two real classes (GGO, consolidation)
    and would pass its own stack here instead.
    """
    out = []
    for m in masks:
        lesion = (m > threshold).astype(float)
        out.append(np.stack([lesion, 1.0 - lesion]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="feature cache h5")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", required=True, help="json with {'val': [uids], 'test': [uids]}")
    ap.add_argument("--pooling", default="slot")
    ap.add_argument("--num-slots", type=int, default=8)
    ap.add_argument("--num-classes", type=int, default=2,
                    help="must match the checkpoint (MosMed severity is 4 after "
                         "folding CT-4 into CT-3); a mismatch fails the load")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--readout", default="gated")
    ap.add_argument("--out", default="runs/alignment")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    from slotmil.models.mil import build_model, slot_pooling_param_count

    splits = json.loads(Path(args.splits).read_text())
    device = args.device if torch.cuda.is_available() else "cpu"

    val_ds = FeatureBagDataset(args.cache, keys=splits["val"], return_mask=True)
    test_ds = FeatureBagDataset(args.cache, keys=splits["test"], return_mask=True)

    match_to = (
        slot_pooling_param_count(val_ds.dim, args.dim, args.num_slots)
        if args.pooling == "mh_abmil"
        else None
    )
    model = build_model(
        pooling=args.pooling, input_dim=val_ds.dim, dim=args.dim,
        num_classes=args.num_classes, num_slots=args.num_slots,
        readout=args.readout, match_params_to=match_to,
    )
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    except RuntimeError as e:
        raise SystemExit(
            f"could not load {args.checkpoint}: {e}\n"
            f"Check --num-classes ({args.num_classes}), --num-slots "
            f"({args.num_slots}), --dim ({args.dim}) and --pooling "
            f"({args.pooling}) match how the checkpoint was trained."
        ) from e

    # --- step 1: validation ------------------------------------------------
    val_attn, val_masks, val_ns = collect(model, val_ds, device, num_workers=args.num_workers)
    val_findings = binary_finding_masks(val_masks)
    val_affinity = slot_finding_affinity(val_attn, val_findings)

    # --- step 2: freeze the naming ----------------------------------------
    assignment = fit_slot_assignment(val_affinity)
    print(f"[align] assignment fit on VAL (frozen): {assignment}")

    # --- step 3: test, under the frozen assignment ------------------------
    test_attn, test_masks, test_ns = collect(model, test_ds, device, num_workers=args.num_workers)
    test_findings = binary_finding_masks(test_masks)
    test_affinity = slot_finding_affinity(test_attn, test_findings)

    results = {
        "pooling": args.pooling,
        "num_slots": args.num_slots,
        "assignment_fit_on": "val",
        "assignment": {str(k): v for k, v in assignment.items()},
        "val_affinity": val_affinity.tolist(),
        "test_alignment": apply_slot_assignment(test_affinity, assignment),
        "test_purity": slot_purity(test_attn, [(m > 0).astype(int) for m in test_masks]),
        "test_consistency": slot_consistency(test_attn, test_findings),
        "test_redundancy": head_redundancy(test_attn),
        "test_localization": evaluate_localization(
            test_attn, test_masks, test_ns, grid=val_ds.grid_h
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"alignment_{args.pooling}.json").write_text(json.dumps(results, indent=2))

    print(f"\n=== slot-to-finding alignment ({args.pooling}) ===")
    a = results["test_alignment"]
    print(f"  assigned affinity   {a['mean_assigned_affinity']:.4f}")
    print(f"  chance              {a['chance_affinity']:.4f}")
    print(f"  lift over chance    {a['lift_over_chance']:.2f}x")
    print(f"  purity / NMI        {results['test_purity']['purity']:.3f} / "
          f"{results['test_purity']['nmi']:.3f}")
    print(f"  consistency         {results['test_consistency']['mean_consistency']:.3f} "
          f"(chance {results['test_consistency']['chance']:.3f})")
    print(f"  head redundancy     {results['test_redundancy']['mean_pairwise_cosine']:.3f}")
    loc = results["test_localization"]
    if loc.get("n_bags"):
        print(f"  best-slot Dice      {loc['dice']:.3f} +/- {loc['dice_std']:.3f}")
        print(f"  pointing game       {loc['pointing_game']:.3f}")
        print(f"  instance AUC        {loc['instance_auc']:.3f}")
    print(f"\nwrote {out}/alignment_{args.pooling}.json")


if __name__ == "__main__":
    main()
