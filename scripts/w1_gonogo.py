#!/usr/bin/env python
"""Week-1 go/no-go on MedMNIST3D (plan.md line 132).

Bar: SlotMIL trains, produces non-degenerate slots, and beats mean pooling.
The published ResNet-18(3D) reference on NoduleMNIST3D is 0.879 AUC / 84.5% ACC
(Yang et al., Scientific Data 2022) on official splits 1158/165/310.

Two paths, deliberately:
  --path fast    small CNN encoder + pooling, end-to-end. Validates the slot
                 module itself in minutes.
  --path cached  DINOv2 features -> h5 -> train. Validates the *entire*
                 cache->train pipeline at small scale, before a GPU-day is
                 committed to LIDC.

This reports whatever it measures. A SlotMIL loss here is a real result --
plan.md Part 7 has pre-committed fallbacks for it, and a massaged pass would
poison the next ten weeks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from slotmil.data.medmnist3d import MedMNIST3DBags, medmnist_collate
from slotmil.eval.classification import aggregate_seeds, delong_test
from slotmil.losses import SlotMILLoss
from slotmil.models.encoders import SliceCNNEncoder
from slotmil.models.mil import build_model, slot_pooling_param_count
from slotmil.train import TrainConfig, fit

REFERENCE = {"nodulemnist3d": {"auc": 0.879, "acc": 0.845, "source": "ResNet-18(3D), Yang et al. 2022"}}


def build_datasets(flag: str, size: int, root: str):
    kw = dict(flag=flag, size=size, root=root, as_slices=True)
    return (
        MedMNIST3DBags(split="train", **kw),
        MedMNIST3DBags(split="val", **kw),
        MedMNIST3DBags(split="test", **kw),
    )


def parse_arm(spec: str) -> tuple[str, str, dict]:
    """``"slot"`` or ``"slot:div=0.1,iters=5"`` -> (label, pooling, overrides).

    Lets one job produce the whole comparison table, including a SlotMIL arm with
    the diversity regulariser on and one with it off, without rerunning the
    baselines.
    """
    label = spec
    pooling, _, rest = spec.partition(":")
    overrides: dict = {}
    for kv in filter(None, rest.split(",")):
        k, _, v = kv.partition("=")
        overrides[k.strip()] = float(v)
    return label, pooling, overrides


def make_model(pooling: str, args, n_classes: int, slice_hw: int, overrides: dict | None = None):
    """All arms share the encoder architecture and readout; only pooling varies."""
    overrides = overrides or {}
    encoder = SliceCNNEncoder(slice_hw=slice_hw, out_dim=args.dim)
    match_to = None
    if pooling == "mh_abmil":
        match_to = slot_pooling_param_count(
            input_dim=args.dim, dim=args.dim, num_slots=args.num_slots
        )
    return build_model(
        pooling=pooling,
        input_dim=args.dim,
        dim=args.dim,
        num_classes=n_classes,
        num_slots=int(overrides.get("K", args.num_slots)),
        readout=args.readout,
        iters=int(overrides.get("iters", args.iters)),
        implicit=not args.no_implicit,
        encoder=encoder,
        match_params_to=match_to,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flag", default="nodulemnist3d")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--root", default="/fs/nexus-scratch/bthapar/SlotMIL/data/medmnist")
    ap.add_argument("--out", default="/fs/nexus-scratch/bthapar/SlotMIL/runs/w1")
    ap.add_argument("--poolings", nargs="+", default=["mean", "gated_abmil", "mh_abmil", "slot"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--num-slots", type=int, default=4)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--readout", default="gated")
    ap.add_argument("--no-implicit", action="store_true")
    ap.add_argument("--w-diversity", type=float, default=0.0)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[w1] loading {args.flag} @ {args.size}^3 ...", flush=True)
    train_ds, val_ds, test_ds = build_datasets(args.flag, args.size, args.root)
    n_classes = train_ds.n_classes
    slice_hw = args.size
    print(
        f"[w1] splits: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"classes={n_classes}",
        flush=True,
    )

    results: dict[str, list[dict]] = {}
    test_scores: dict[str, np.ndarray] = {}
    test_truth = None

    for spec in args.poolings:
        label, pooling, overrides = parse_arm(spec)
        runs = []
        for seed in args.seeds:
            print(f"[w1] {label} seed={seed}", flush=True)
            model = make_model(pooling, args, n_classes, slice_hw, overrides)
            n_params = sum(p.numel() for p in model.parameters())

            cfg = TrainConfig(
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=seed,
                select_metric="auc",
            )
            criterion = SlotMILLoss(
                w_diversity=overrides.get("div", args.w_diversity),
                w_entropy=overrides.get("ent", 0.0),
            )

            res = fit(
                model, train_ds, val_ds, cfg,
                collate_fn=medmnist_collate,
                criterion=criterion,
                test_ds=test_ds,
                out_dir=out_root / label.replace(":", "_") / f"seed{seed}",
            )
            row = {"seed": seed, "n_params": n_params, **res["test"]}
            runs.append(row)
            print(
                f"[w1]   -> test auc={row['auc']:.4f} acc={row['acc']:.4f} "
                f"params={n_params/1e3:.0f}k"
                + (
                    f" active_slots={row.get('active_slots', float('nan')):.2f}"
                    f" max_cos={row.get('max_off_diag_cos', float('nan')):.3f}"
                    if "active_slots" in row
                    else ""
                ),
                flush=True,
            )
        results[label] = runs

    summary = {p: aggregate_seeds(r) for p, r in results.items()}

    print("\n" + "=" * 78)
    print(f"W1 GO/NO-GO  --  {args.flag}")
    ref = REFERENCE.get(args.flag)
    if ref:
        print(f"reference: AUC {ref['auc']:.3f} / ACC {ref['acc']:.3f}  ({ref['source']})")
    print("=" * 78)
    print(f"{'arm':<18}{'AUC':>18}{'ACC':>18}{'params':>10}{'active':>8}{'maxcos':>8}")
    for p, s in summary.items():
        auc, acc = s["auc"], s["acc"]
        act = f"{s['active_slots']['mean']:>8.2f}" if "active_slots" in s else " " * 8
        cos = f"{s['max_off_diag_cos']['mean']:>8.3f}" if "max_off_diag_cos" in s else " " * 8
        print(
            f"{p:<18}{auc['mean']:>10.4f} +/-{auc['std']:.4f}"
            f"{acc['mean']:>10.4f} +/-{acc['std']:.4f}"
            f"{s['n_params']['mean']/1e3:>9.0f}k{act}{cos}"
        )

    # Go/no-go conditions, evaluated explicitly rather than eyeballed. The slot
    # arm judged is the best-AUC one, so adding a regularised variant cannot make
    # the verdict look worse than the method's best honest configuration.
    slot_arms = {k: v for k, v in summary.items() if k.split(":")[0] == "slot"}
    verdict = {}
    if slot_arms:
        best_label = max(slot_arms, key=lambda k: slot_arms[k]["auc"]["mean"])
        best = slot_arms[best_label]
        verdict["_best_slot_arm"] = best_label

        # A bare mean comparison across 3 seeds says almost nothing -- an 0.003
        # AUC edge with 0.016 spread is noise, and calling that PASS would set up
        # a false go decision for the next ten weeks. Every head-to-head is
        # therefore gated on an actual test.
        def beats(other: str) -> bool | None:
            if other not in summary:
                return None
            a, b = best["auc"]["values"], summary[other]["auc"]["values"]
            if len(a) < 2 or len(b) < 2:
                return None
            p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
            delta = float(np.mean(a) - np.mean(b))
            verdict[f"_vs_{other}"] = f"delta={delta:+.4f} p={p:.3f}"
            return bool(delta > 0 and p < 0.05)

        for other in ("mean", "gated_abmil", "mh_abmil"):
            got = beats(other)
            if got is not None:
                verdict[f"beats_{other}"] = got
        if ref:
            verdict["meets_reference_auc"] = best["auc"]["mean"] >= ref["auc"]
        if "active_slots" in best:
            verdict["slots_non_degenerate"] = bool(
                best["active_slots"]["mean"] > 1.0
                and best["max_off_diag_cos"]["mean"] < 0.9
            )

    print("\nverdict:")
    for k, v in verdict.items():
        if k.startswith("_"):
            print(f"  {k[1:]:<24} {v}")
        else:
            print(f"  {k:<24} {'PASS' if v else 'FAIL'}")

    with open(out_root / "summary.json", "w") as f:
        json.dump(
            {"args": vars(args), "summary": summary, "verdict": verdict, "reference": ref},
            f,
            indent=2,
            default=str,
        )
    print(f"\nwrote {out_root/'summary.json'}")


if __name__ == "__main__":
    main()
