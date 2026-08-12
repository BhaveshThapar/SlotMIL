#!/usr/bin/env python
"""Diagnose why the parameter-matched multi-head ABMIL control scores below chance.

Collects, for EVERY (condition, arm, seed) and BOTH splits (val, test), per-bag and
per-head:

  * instance-level ROC AUC of that head's attention against the patch lesion mask
  * the head's lesion attention-mass share (the quantity `slot_finding_affinity`
    averages -- i.e. the Hungarian objective), plus the same quantity computed
    WITHOUT the `a / a.sum(-1)` renormalisation, to test the claimed asymmetry
  * attention concentration (normalised entropy, top-0.1% mass)
  * bag prevalence and the readout gate mass per head

Plus two architecture-prior controls per condition: an UNTRAINED slot model and an
UNTRAINED mh_abmil model, to separate "training made it anti-correlated" from
"this attention functional on these features is anti-correlated before any
training".

Plus two geometry baselines per bag (in-plane radial position, axial position) to
test whether a positional prior alone can produce systematic below-chance AUC.

Nothing is aggregated here; everything is dumped to an npz for offline analysis.
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

CONDITIONS = [
    ("nodule_present", "runs/lidc", "data/lidc/splits.json", 2, [0, 1, 2]),
    ("malignancy", "runs/lidc_malignancy", "data/lidc/splits_malignancy.json", 2, [0, 1, 2]),
    ("balanced_presence", "runs/lidc_balanced", "data/lidc/splits_balanced.json", 2, [0, 1, 2, 3, 4]),
]
ARMS = [("slot_div=0.5", "slot"), ("mh_abmil", "mh_abmil")]


def auc_rows(scores: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """Exact ROC AUC (mid-rank tie handling) of each row of ``scores`` [K, N].

    Uses searchsorted over the sorted negatives: for each positive, (#neg < p +
    #neg <= p) / 2 is its tie-corrected concordance count.
    """
    pos = target > 0
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return np.full(scores.shape[0], np.nan)
    out = np.empty(scores.shape[0], dtype=np.float64)
    for k in range(scores.shape[0]):
        s = scores[k]
        neg = torch.sort(s[~pos]).values
        p = s[pos]
        lo = torch.searchsorted(neg, p, right=False).double()
        hi = torch.searchsorted(neg, p, right=True).double()
        out[k] = float(((lo + hi) / 2).sum() / (n_pos * n_neg))
    return out


def norm_entropy(a: torch.Tensor) -> np.ndarray:
    """Entropy of each row of a [K, N] attention map, normalised by log N."""
    p = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    h = -(p.clamp_min(1e-12).log() * p).sum(dim=-1)
    return (h / np.log(a.shape[-1])).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/lidc/features_dinov2_vitb14.h5")
    ap.add_argument("--out", default="runs/control_diagnosis.npz")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--conditions", nargs="*", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[diag] device={device}", flush=True)

    rec = []  # flat list of dict rows

    for cond, runs, splits_path, n_cls, seeds in CONDITIONS:
        if args.conditions and cond not in args.conditions:
            continue
        if not Path(splits_path).exists():
            print(f"[diag] skip {cond}: no splits", flush=True)
            continue
        sp = json.loads(Path(splits_path).read_text())
        lm = sp.get("labels")

        def sub(keys):
            p = FeatureBagDataset(args.cache, keys=keys, labels=lm, return_mask=True)
            return FeatureBagDataset(args.cache, keys=p.masked_keys(), labels=lm, return_mask=True)

        datasets = {"val": sub(sp["val"]), "test": sub(sp["test"])}
        dim_in = datasets["val"].dim
        grid = datasets["val"].grid_h

        # ---- build every model for this condition, plus untrained controls ----
        models = {}
        match_to = slot_pooling_param_count(dim_in, 256, 8)
        for arm_dir, pooling in ARMS:
            for seed in seeds:
                ckpt = Path(runs) / arm_dir / f"seed{seed}" / "best.pt"
                if not ckpt.exists():
                    continue
                m = build_model(
                    pooling=pooling, input_dim=dim_in, dim=256,
                    num_classes=sp.get("num_classes", n_cls), num_slots=8,
                    readout="gated",
                    match_params_to=match_to if pooling == "mh_abmil" else None,
                )
                m.load_state_dict(torch.load(ckpt, map_location="cpu"))
                models[(arm_dir, seed)] = m.eval().to(device)
        for pooling, tag in [("slot", "UNTRAINED_slot"), ("mh_abmil", "UNTRAINED_mh_abmil")]:
            torch.manual_seed(1234)
            m = build_model(
                pooling=pooling, input_dim=dim_in, dim=256,
                num_classes=sp.get("num_classes", n_cls), num_slots=8, readout="gated",
                match_params_to=match_to if pooling == "mh_abmil" else None,
            )
            models[(tag, -1)] = m.eval().to(device)
        print(f"[diag] {cond}: {len(models)} models, dim_in={dim_in}, grid={grid}", flush=True)

        for split, ds in datasets.items():
            loader = DataLoader(ds, batch_size=1, shuffle=False,
                                num_workers=args.workers, collate_fn=collate_bags)
            for bi, b in enumerate(loader):
                n = int(b["lengths"][0])
                feats = b["features"].to(device)
                pm = b["pad_mask"].to(device)
                tgt = (b["patch_target"][0, :n] > 0).to(device).float()
                n_pos = int(tgt.sum())
                if n_pos == 0 or n_pos == n:
                    continue
                ns = int(b["n_slices"][0])

                # geometry baselines: is a positional prior alone informative?
                zz = torch.arange(ns, device=device).float()
                zc = (zz - (ns - 1) / 2).abs() / max((ns - 1) / 2, 1)
                yy, xx = torch.meshgrid(
                    torch.arange(grid, device=device).float(),
                    torch.arange(grid, device=device).float(), indexing="ij")
                rr = (((yy - (grid - 1) / 2) ** 2 + (xx - (grid - 1) / 2) ** 2).sqrt()
                      / max((grid - 1) / 2, 1)).reshape(-1)
                npatch = n // ns
                geo = torch.stack([
                    (-zc[:, None].expand(ns, npatch)).reshape(-1)[:n],       # central slices
                    (-rr[None, :npatch].expand(ns, npatch)).reshape(-1)[:n],  # image centre
                ])
                geo_auc = auc_rows(geo, tgt)

                with torch.no_grad():
                    for (arm, seed), m in models.items():
                        out = m(feats, pm)
                        a = out["attn"][0, :, :n].float()
                        gate = out["attribution"][0].float().cpu().numpy()
                        aucs = auc_rows(a, tgt)
                        a_sum = a.sum(dim=-1)                       # per-head total mass
                        lesion_raw = (a * tgt).sum(dim=-1)          # mass on lesion, unnormalised
                        lesion_norm = lesion_raw / a_sum.clamp_min(1e-8)
                        rec.append(dict(
                            cond=cond, arm=arm, seed=seed, split=split, bag=bi,
                            n=n, n_pos=n_pos, prev=n_pos / n,
                            auc=aucs,
                            aff_norm=lesion_norm.cpu().numpy(),   # what slot_finding_affinity uses
                            aff_raw=lesion_raw.cpu().numpy(),     # without the renormalisation
                            mass=a_sum.cpu().numpy(),
                            ent=norm_entropy(a),
                            gate=gate,
                            geo_auc_z=geo_auc[0], geo_auc_r=geo_auc[1],
                        ))
                if bi % 25 == 0:
                    print(f"[diag] {cond}/{split} bag {bi} n={n} pos={n_pos}", flush=True)

        for m in models.values():
            m.cpu()
        del models
        torch.cuda.empty_cache()

    # ---- dump ----
    keys_scalar = ["cond", "arm", "seed", "split", "bag", "n", "n_pos", "prev",
                   "geo_auc_z", "geo_auc_r"]
    keys_vec = ["auc", "aff_norm", "aff_raw", "mass", "ent", "gate"]
    d = {k: np.array([r[k] for r in rec]) for k in keys_scalar}
    for k in keys_vec:
        d[k] = np.stack([r[k] for r in rec])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **d)
    print(f"[diag] wrote {args.out}  rows={len(rec)}", flush=True)


if __name__ == "__main__":
    main()
