#!/usr/bin/env python
"""Pass 2: is the lesion AUC (either sign) just a positional prior?

Lesion patches are not uniformly placed on the patch grid -- lungs sit near the
middle of the axial field of view and near the middle of the scanned volume. Any
attention map with a centre bias will score high lesion AUC, and any map with a
periphery bias will score symmetrically low, without either one having anything
to do with lesions.

For every (condition, arm, seed) and split this recomputes each head's instance
AUC four ways:

  raw     unrestricted (what the paper reports)
  slice   only instances on slices that contain a lesion  (removes the axial prior)
  rad     stratified over 6 in-plane radial bins          (removes the radial prior)
  joint   stratified over (slice x radial bin)            (removes both)

Stratified AUC is Mantel-Haenszel style: concordant pairs are only counted within
a stratum, so a score that merely reproduces the stratum variable scores 0.5.
The two model-free positional baselines are pushed through the identical code so
the collapse can be verified rather than assumed.

Also records mean attention per radial bin per head, which shows directly whether
a head is a centre detector or a periphery detector.
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
NBIN = 6


def strat_auc(scores: torch.Tensor, pos: torch.Tensor, strat: torch.Tensor) -> np.ndarray:
    """Mantel-Haenszel stratified AUC of each row of ``scores`` [K, N].

    Only strata containing at least one positive and one negative contribute; the
    result is total concordant pairs over total within-stratum pairs.
    """
    k = scores.shape[0]
    conc = torch.zeros(k, dtype=torch.float64, device=scores.device)
    pairs = 0.0
    for s in torch.unique(strat[pos]):
        m = strat == s
        pm, nm = m & pos, m & (~pos)
        npos, nneg = int(pm.sum()), int(nm.sum())
        if npos == 0 or nneg == 0:
            continue
        neg = torch.sort(scores[:, nm], dim=-1).values           # K, nneg
        p = scores[:, pm]                                        # K, npos
        lo = torch.searchsorted(neg.contiguous(), p.contiguous(), right=False).double()
        hi = torch.searchsorted(neg.contiguous(), p.contiguous(), right=True).double()
        conc += ((lo + hi) / 2).sum(dim=-1)
        pairs += npos * nneg
    if pairs == 0:
        return np.full(k, np.nan)
    return (conc / pairs).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/lidc/features_dinov2_vitb14.h5")
    ap.add_argument("--out", default="runs/control_confound.npz")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--untrained-seeds", type=int, default=1,
                    help="how many random inits to evaluate for the untrained baselines")
    ap.add_argument("--untrained-only", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rec = []

    for cond, runs, splits_path, n_cls, seeds in CONDITIONS:
        if args.conditions and cond not in args.conditions:
            continue
        sp = json.loads(Path(splits_path).read_text())
        lm = sp.get("labels")

        def sub(keys):
            p = FeatureBagDataset(args.cache, keys=keys, labels=lm, return_mask=True)
            return FeatureBagDataset(args.cache, keys=p.masked_keys(), labels=lm, return_mask=True)

        datasets = {"val": sub(sp["val"]), "test": sub(sp["test"])}
        dim_in, grid = datasets["val"].dim, datasets["val"].grid_h
        match_to = slot_pooling_param_count(dim_in, 256, 8)

        models = {}
        if not args.untrained_only:
            for arm_dir, pooling in ARMS:
                for seed in seeds:
                    ckpt = Path(runs) / arm_dir / f"seed{seed}" / "best.pt"
                    if not ckpt.exists():
                        continue
                    m = build_model(pooling=pooling, input_dim=dim_in, dim=256,
                                    num_classes=sp.get("num_classes", n_cls), num_slots=8,
                                    readout="gated",
                                    match_params_to=match_to if pooling == "mh_abmil" else None)
                    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
                    models[(arm_dir, seed)] = m.eval().to(device)
        for pooling, tag in [("slot", "UNTRAINED_slot"), ("mh_abmil", "UNTRAINED_mh_abmil")]:
            for us in range(args.untrained_seeds):
                torch.manual_seed(1234 + us)
                m = build_model(pooling=pooling, input_dim=dim_in, dim=256,
                                num_classes=sp.get("num_classes", n_cls), num_slots=8,
                                readout="gated",
                                match_params_to=match_to if pooling == "mh_abmil" else None)
                models[(tag, -1 - us)] = m.eval().to(device)
        print(f"[cf] {cond}: {len(models)} models", flush=True)

        # in-plane radius per patch position, binned into NBIN equal-count bins
        yy, xx = torch.meshgrid(torch.arange(grid).float(), torch.arange(grid).float(),
                                indexing="ij")
        r_pos = (((yy - (grid - 1) / 2) ** 2 + (xx - (grid - 1) / 2) ** 2).sqrt()).reshape(-1)
        edges = torch.quantile(r_pos, torch.linspace(0, 1, NBIN + 1)[1:-1])
        rbin_pos = torch.bucketize(r_pos, edges).to(device)
        r_pos = r_pos.to(device)

        for split, ds in datasets.items():
            loader = DataLoader(ds, batch_size=1, shuffle=False,
                                num_workers=args.workers, collate_fn=collate_bags)
            for bi, b in enumerate(loader):
                n = int(b["lengths"][0])
                ns = int(b["n_slices"][0])
                npatch = n // ns
                tgt = (b["patch_target"][0, :n] > 0).to(device)
                if int(tgt.sum()) in (0, n):
                    continue
                feats, pm = b["features"].to(device), b["pad_mask"].to(device)

                slice_id = torch.arange(ns, device=device).repeat_interleave(npatch)[:n]
                rbin = rbin_pos[:npatch].repeat(ns)[:n]
                rad = r_pos[:npatch].repeat(ns)[:n]
                lesion_slices = torch.unique(slice_id[tgt])
                on_lesion_slice = torch.isin(slice_id, lesion_slices)

                strata = {
                    "raw": torch.zeros(n, dtype=torch.long, device=device),
                    "rad": rbin.long(),
                    "slice": slice_id,
                    "joint": slice_id * NBIN + rbin.long(),
                }
                zc = (torch.arange(ns, device=device).float() - (ns - 1) / 2).abs()
                geo = torch.stack([-zc.repeat_interleave(npatch)[:n], -rad])

                row = dict(cond=cond, arm="__geo__", seed=-2, split=split, bag=bi,
                           n=n, n_pos=int(tgt.sum()))
                for name, s in strata.items():
                    row[f"geo_{name}"] = strat_auc(geo, tgt, s)
                # lesion density per radial bin (the confound itself)
                row["pos_by_rbin"] = np.array(
                    [float((tgt & (rbin == j)).sum()) for j in range(NBIN)])
                row["n_by_rbin"] = np.array([float((rbin == j).sum()) for j in range(NBIN)])
                rec.append(row)

                with torch.no_grad():
                    for (arm, seed), m in models.items():
                        a = m(feats, pm)["attn"][0, :, :n].float()
                        an = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                        r2 = dict(cond=cond, arm=arm, seed=seed, split=split, bag=bi,
                                  n=n, n_pos=int(tgt.sum()))
                        for name, s in strata.items():
                            r2[f"auc_{name}"] = strat_auc(a, tgt, s)
                        r2["aff_norm"] = (an * tgt).sum(dim=-1).cpu().numpy()
                        r2["attn_by_rbin"] = np.stack([
                            (an[:, rbin == j].sum(dim=-1)
                             / max(int((rbin == j).sum()), 1)).cpu().numpy()
                            for j in range(NBIN)], axis=1)  # K, NBIN
                        rec.append(r2)
                if bi % 40 == 0:
                    print(f"[cf] {cond}/{split} bag {bi}", flush=True)

        for m in models.values():
            m.cpu()
        del models
        torch.cuda.empty_cache()

    # Rows are ragged (geometry rows carry geo_*, model rows carry auc_*); pad every
    # missing numeric field with NaN so the dump is one rectangular table.
    keys = sorted({k for r in rec for k in r})
    proto = {}
    for k in keys:
        v = next(r[k] for r in rec if k in r)
        proto[k] = np.asarray(v)
    d = {}
    for k in keys:
        if proto[k].dtype.kind in "US":
            d[k] = np.array([r.get(k, "") for r in rec])
            continue
        fill = np.full(proto[k].shape, np.nan, dtype=np.float64)
        d[k] = np.stack([np.asarray(r[k], dtype=np.float64) if k in r else fill
                         for r in rec])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **d)
    print(f"[cf] wrote {args.out} rows={len(rec)}", flush=True)


if __name__ == "__main__":
    main()
