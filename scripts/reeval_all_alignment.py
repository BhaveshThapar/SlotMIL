#!/usr/bin/env python
"""Re-evaluate localisation for every (condition, arm, seed) with the CORRECTED metric.

Every alignment number produced before this used ``attn.max(axis=0)``, which for
softmax-over-slots attention measures assignment confidence rather than lesion
saliency, and reported ~0.489 where the frozen lesion slot gives ~0.842. It also
only ever ran seed 0.

This recomputes, per seed:
  - the Hungarian slot->finding assignment fit on VALIDATION and frozen
  - instance AUC using that frozen lesion slot (the honest statistic)
  - instance AUC using max-over-slots (to quantify the size of the old error)
  - oracle best slot (an upper bound; NOT a reportable number)
  - average precision against prevalence, which unlike Dice/pointing-game retains
    power at 0.07% positive rate

The matched multi-head-ABMIL control gets the same treatment: its attention
normalises over instances, not slots, so max-over-heads was never wrong for it in
the same way -- which is precisely why the two must be recomputed on equal terms
before any comparison is claimed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from slotmil.data.feature_cache import FeatureBagDataset, collate_bags
from slotmil.eval.alignment import fit_slot_assignment, slot_finding_affinity
from slotmil.models.mil import build_model, slot_pooling_param_count

CONDITIONS = [
    ("nodule_present", "runs/lidc", "data/lidc/splits.json", 2),
    ("malignancy", "runs/lidc_malignancy", "data/lidc/splits_malignancy.json", 2),
    ("balanced_presence", "runs/lidc_balanced", "data/lidc/splits_balanced.json", 2),
]
# (run-directory name, pooling). train_cached.py writes the directory as the arm
# spec with ':' replaced by '_', which is why normal_guidance carries its lam.
ARMS = [
    ("slot_div=0.5", "slot"),
    ("mh_abmil", "mh_abmil"),
    ("centre_gaussian", "centre_gaussian"),
    ("normal_guidance_lam=0.1", "normal_guidance"),
]


def binary_findings(masks):
    return [np.stack([(m > 0).astype(float), 1.0 - (m > 0).astype(float)]) for m in masks]


@torch.no_grad()
def collect(model, ds, device, workers=4):
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=workers,
                        collate_fn=collate_bags)
    model.eval().to(device)
    attns, masks = [], []
    for b in loader:
        a = model(b["features"].to(device), b["pad_mask"].to(device))["attn"].float().cpu().numpy()
        for i, n in enumerate(b["lengths"].tolist()):
            attns.append(a[i, :, :n])
            masks.append(b["patch_target"][i, :n].numpy())
    return attns, masks


def score(attns, masks, lesion_slot):
    frozen, maxover, oracle, aps, prevs = [], [], [], [], []
    for a, m in zip(attns, masks):
        t = (m > 0).astype(int)
        if t.sum() == 0 or t.sum() == len(t):
            continue
        frozen.append(roc_auc_score(t, a[lesion_slot]))
        maxover.append(roc_auc_score(t, a.max(axis=0)))
        oracle.append(max(roc_auc_score(t, a[k]) for k in range(a.shape[0])))
        aps.append(average_precision_score(t, a[lesion_slot]))
        prevs.append(t.mean())
    if not frozen:
        return None
    return {
        "instance_auc_frozen_slot": float(np.mean(frozen)),
        "instance_auc_max_over_slots": float(np.mean(maxover)),
        "instance_auc_oracle_slot": float(np.mean(oracle)),
        "avg_precision": float(np.mean(aps)),
        "prevalence": float(np.mean(prevs)),
        "ap_lift": float(np.mean(aps) / max(np.mean(prevs), 1e-12)),
        "n_bags": len(frozen),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/lidc/features_dinov2_vitb14.h5")
    ap.add_argument("--out", default="runs/corrected_alignment.json")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    for cond, runs, splits_path, n_cls in CONDITIONS:
        if not Path(splits_path).exists():
            print(f"[reeval] skip {cond}: no splits"); continue
        sp = json.loads(Path(splits_path).read_text())
        lm = sp.get("labels")

        def sub(keys):
            p = FeatureBagDataset(args.cache, keys=keys, labels=lm, return_mask=True)
            return FeatureBagDataset(args.cache, keys=p.masked_keys(), labels=lm, return_mask=True)

        val_ds, test_ds = sub(sp["val"]), sub(sp["test"])

        for arm_dir, pooling in ARMS:
            for seed in args.seeds:
                ckpt = Path(runs) / arm_dir / f"seed{seed}" / "best.pt"
                if not ckpt.exists():
                    continue
                match_to = (slot_pooling_param_count(val_ds.dim, 256, 8)
                            if pooling == "mh_abmil" else None)
                m = build_model(pooling=pooling, input_dim=val_ds.dim, dim=256,
                                num_classes=sp.get("num_classes", n_cls),
                                num_slots=8, readout="gated", match_params_to=match_to)
                m.load_state_dict(torch.load(ckpt, map_location="cpu"))

                va, vm = collect(m, val_ds, device)
                assign = fit_slot_assignment(slot_finding_affinity(va, binary_findings(vm)))
                lesion_slot = next((k for k, v in assign.items() if v == 0), 0)

                ta, tm = collect(m, test_ds, device)
                s = score(ta, tm, lesion_slot)
                if s is None:
                    continue
                s["lesion_slot"] = int(lesion_slot)
                results.setdefault(cond, {}).setdefault(arm_dir, {})[f"seed{seed}"] = s
                print(f"[reeval] {cond:<18} {arm_dir:<13} seed{seed}  "
                      f"frozen={s['instance_auc_frozen_slot']:.4f}  "
                      f"max-over={s['instance_auc_max_over_slots']:.4f}  "
                      f"slot={lesion_slot}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 82)
    print(f"{'condition':<20}{'arm':<14}{'frozen slot':>14}{'max-over-slots':>16}{'n':>5}")
    for cond, arms in results.items():
        for arm, seeds in arms.items():
            fr = [v["instance_auc_frozen_slot"] for v in seeds.values()]
            mx = [v["instance_auc_max_over_slots"] for v in seeds.values()]
            print(f"{cond:<20}{arm:<14}{np.mean(fr):>8.4f}+/-{np.std(fr):.3f}"
                  f"{np.mean(mx):>10.4f}+/-{np.std(mx):.3f}{len(fr):>5}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
