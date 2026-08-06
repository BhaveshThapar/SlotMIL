#!/usr/bin/env python
"""Train MIL arms on a cached feature bag dataset (LIDC / MosMed).

The counterpart to w1_gonogo.py for real CT: instead of a trainable CNN encoder
over MedMNIST slices, bags come from the frozen-DINOv2 HDF5 cache.

Memory note: at patch level a LIDC bag is n_slices x 256 tokens -- around 28k
instances for a typical scan, or ~86 MB in fp32. Training therefore subsamples
slices per epoch (``--max-slices``), which plan.md line 88 prescribes for long
volumes anyway; evaluation always uses the full bag, so no test-time information
is discarded.

Checkpoints are written per arm/seed because scripts/eval_alignment.py consumes
them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from slotmil.data.feature_cache import FeatureBagDataset, collate_bags
from slotmil.eval.classification import aggregate_seeds
from slotmil.losses import SlotMILLoss
from slotmil.models.mil import build_model, slot_pooling_param_count
from slotmil.train import TrainConfig, fit


def parse_arm(spec: str):
    """``"slot:div=0.1,K=8"`` -> (label, pooling, overrides)."""
    pooling, _, rest = spec.partition(":")
    overrides = {}
    for kv in filter(None, rest.split(",")):
        k, _, v = kv.partition("=")
        overrides[k.strip()] = float(v)
    return spec, pooling, overrides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default="runs/lidc")
    ap.add_argument("--arms", nargs="+",
                    default=["mean", "gated_abmil", "mh_abmil", "slot", "slot:div=0.1"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--num-slots", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--readout", default="gated")
    ap.add_argument("--instance-level", default="patch", choices=["patch", "slice"])
    ap.add_argument("--max-slices", type=int, default=48,
                    help="stochastic slice subsampling during training only")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-classes", type=int, default=2)
    args = ap.parse_args()

    splits = json.loads(Path(args.splits).read_text())
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # A split file carrying a label map (from --merge-classes) overrides the
    # labels stored in the cache. Without this the merge would apply to the
    # partitioning but not to the labels actually served.
    label_map = splits.get("labels")
    if label_map:
        n_cls = splits.get("num_classes", len(set(label_map.values())))
        if n_cls != args.num_classes:
            print(f"[train] splits define {n_cls} classes "
                  f"(merge_classes={splits.get('merge_classes')}); "
                  f"overriding --num-classes {args.num_classes} -> {n_cls}")
            args.num_classes = n_cls

    mk = lambda keys, train: FeatureBagDataset(  # noqa: E731
        args.cache, keys=keys, labels=label_map,
        instance_level=args.instance_level,
        max_slices=args.max_slices if train else None,
        train=train, return_mask=False,
    )
    train_ds, val_ds, test_ds = mk(splits["train"], True), mk(splits["val"], False), mk(splits["test"], False)
    input_dim = train_ds.dim
    print(f"[train] cache dim={input_dim} grid={train_ds.grid_h}x{train_ds.grid_w} "
          f"| train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    # Verify every served label is in range before touching the GPU. Out-of-range
    # labels surface as `nll_loss ... Assertion t >= 0 && t < n_classes failed`
    # deep in a CUDA kernel, with no indication of which dataset or which label.
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        labels = {
            (label_map[uid] if label_map else None) for uid in ds.keys
        } if label_map else None
        if labels is None:
            import h5py
            with h5py.File(args.cache, "r") as f:
                labels = {int(f[uid].attrs["label"]) for uid in ds.keys}
        bad = sorted(l for l in labels if not (0 <= l < args.num_classes))
        if bad:
            raise SystemExit(
                f"[train] split '{name}' contains label(s) {bad} outside "
                f"[0, {args.num_classes}). Either --num-classes is wrong, or the "
                f"split file needs a label map (make_splits.py --merge-classes "
                f"writes one)."
            )
    print(f"[train] labels validated against num_classes={args.num_classes}", flush=True)

    results = {}
    for spec in args.arms:
        label, pooling, ov = parse_arm(spec)
        runs = []
        for seed in args.seeds:
            match_to = (
                slot_pooling_param_count(input_dim, args.dim,
                                         int(ov.get("K", args.num_slots)))
                if pooling == "mh_abmil" else None
            )
            model = build_model(
                pooling=pooling, input_dim=input_dim, dim=args.dim,
                num_classes=args.num_classes,
                num_slots=int(ov.get("K", args.num_slots)),
                readout=args.readout, iters=int(ov.get("iters", args.iters)),
                match_params_to=match_to,
            )
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[train] {label} seed={seed} params={n_params/1e3:.0f}k", flush=True)

            cfg = TrainConfig(
                epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                num_workers=args.num_workers, seed=seed, select_metric="auc",
            )
            res = fit(
                model, train_ds, val_ds, cfg,
                collate_fn=collate_bags,
                criterion=SlotMILLoss(w_diversity=ov.get("div", 0.0),
                                      w_entropy=ov.get("ent", 0.0)),
                test_ds=test_ds,
                out_dir=out_root / label.replace(":", "_") / f"seed{seed}",
            )
            row = {"seed": seed, "n_params": n_params, **res["test"]}
            runs.append(row)
            msg = f"[train]   -> auc={row['auc']:.4f} acc={row['acc']:.4f}"
            if "active_slots" in row:
                msg += f" active={row['active_slots']:.2f} maxcos={row['max_off_diag_cos']:.3f}"
            print(msg, flush=True)
        results[label] = runs

    summary = {k: aggregate_seeds(v) for k, v in results.items()}

    print("\n" + "=" * 80)
    print(f"{'arm':<18}{'AUC':>18}{'ACC':>18}{'params':>10}{'active':>8}{'maxcos':>8}")
    for k, s in summary.items():
        act = f"{s['active_slots']['mean']:>8.2f}" if "active_slots" in s else " " * 8
        cos = f"{s['max_off_diag_cos']['mean']:>8.3f}" if "max_off_diag_cos" in s else " " * 8
        print(f"{k:<18}{s['auc']['mean']:>10.4f} +/-{s['auc']['std']:.4f}"
              f"{s['acc']['mean']:>10.4f} +/-{s['acc']['std']:.4f}"
              f"{s['n_params']['mean']/1e3:>9.0f}k{act}{cos}")

    # Head-to-head with significance, same gate as W1: a bare mean comparison
    # over a few seeds is not evidence.
    slot_arms = {k: v for k, v in summary.items() if k.split(":")[0] == "slot"}
    comparisons = {}
    if slot_arms:
        best_label = max(slot_arms, key=lambda k: slot_arms[k]["auc"]["mean"])
        a = slot_arms[best_label]["auc"]["values"]
        print(f"\nbest slot arm: {best_label}")
        for other in summary:
            if other == best_label:
                continue
            b = summary[other]["auc"]["values"]
            if len(a) > 1 and len(b) > 1:
                p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
                d = float(np.mean(a) - np.mean(b))
                comparisons[other] = {"delta": d, "p": p, "significant": bool(d > 0 and p < 0.05)}
                verdict = "SIGNIFICANT" if comparisons[other]["significant"] else "not significant"
                print(f"  vs {other:<16} delta={d:+.4f}  p={p:.3f}  {verdict}")

    (out_root / "summary.json").write_text(json.dumps(
        {"args": vars(args), "summary": summary, "comparisons": comparisons}, indent=2, default=str))
    print(f"\nwrote {out_root/'summary.json'}")


if __name__ == "__main__":
    main()
