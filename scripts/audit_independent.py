#!/usr/bin/env python
"""INDEPENDENT audit. Does NOT import scripts/reeval_all_alignment.py.

Re-derives, from scratch:
  1. the headline frozen-lesion-slot test AUC for runs/lidc/slot_div=0.5/seed0
  2. what the Hungarian rule actually reduces to at F=2
  3. model-free positional baselines on the identical bags
  4. the supervised probe under BOTH estimands (pooled 20:1 vs per-bag macro)
  5. the sign-flip asymmetry between SlotMIL and MH-ABMIL selection
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from slotmil.data.feature_cache import FeatureBagDataset, collate_bags
from slotmil.models.mil import build_model, slot_pooling_param_count

CACHE = "data/lidc/features_dinov2_vitb14.h5"
SPLITS = "data/lidc/splits.json"
GRID = 16
NP_ = GRID * GRID
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = {}


def bag_auc(t, s):
    return roc_auc_score(t, s)


# ---------------------------------------------------------------- probe
def fit_probe(f, uids, max_scans=60, npp=20, seed=0):
    rng = np.random.default_rng(seed)
    X, y, used = [], [], 0
    for uid in uids:
        g = f[uid]
        if "mask" not in g:
            continue
        m = g["mask"][:]
        pos = np.argwhere(m > 0)
        if len(pos) == 0:
            continue
        feats = g["features"][:]
        for (s, r, c) in pos:
            X.append(feats[s, r * GRID + c]); y.append(1)
        flat = m.reshape(m.shape[0], -1)
        neg = np.argwhere(flat == 0)
        pick = rng.choice(len(neg), min(len(pos) * npp, len(neg)), replace=False)
        for j in pick:
            s, p = neg[j]; X.append(feats[s, p]); y.append(0)
        used += 1
        if used >= max_scans:
            break
    X = np.array(X, dtype=np.float32); y = np.array(y)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)
    return clf, X.shape, int(y.sum())


def probe_pooled_20to1(f, clf, uids, max_scans=40, npp=20, seed=0):
    """Exactly the estimand scripts/probe_ceiling.py reports."""
    rng = np.random.default_rng(seed)
    # probe_ceiling.py uses ONE rng for both train and test collect calls, so the
    # test draw is downstream of the train draw. Reproduce that by burning the
    # train draws first (done by the caller passing an advanced rng); here we use
    # a fresh rng and report both.
    X, y, used = [], [], 0
    for uid in uids:
        g = f[uid]
        if "mask" not in g:
            continue
        m = g["mask"][:]
        pos = np.argwhere(m > 0)
        if len(pos) == 0:
            continue
        feats = g["features"][:]
        for (s, r, c) in pos:
            X.append(feats[s, r * GRID + c]); y.append(1)
        flat = m.reshape(m.shape[0], -1)
        neg = np.argwhere(flat == 0)
        pick = rng.choice(len(neg), min(len(pos) * npp, len(neg)), replace=False)
        for j in pick:
            s, p = neg[j]; X.append(feats[s, p]); y.append(0)
        used += 1
        if used >= max_scans:
            break
    X = np.array(X, dtype=np.float32); y = np.array(y)
    sc = X @ clf.coef_[0] + clf.intercept_[0]
    return float(roc_auc_score(y, sc)), int(y.sum()), int(len(y) - y.sum()), used


# ---------------------------------------------------------------- collect
@torch.no_grad()
def collect(model, ds, probe_w=None, probe_b=0.0, workers=4):
    ld = DataLoader(ds, batch_size=2, shuffle=False, num_workers=workers,
                    collate_fn=collate_bags)
    model.eval().to(DEV)
    A, M, P, NS = [], [], [], []
    for b in ld:
        x = b["features"].to(DEV)
        a = model(x, b["pad_mask"].to(DEV))["attn"].float().cpu().numpy()
        if probe_w is not None:
            pw = torch.as_tensor(probe_w, dtype=torch.float32, device=DEV)
            ps = (x @ pw + probe_b).cpu().numpy()
        for i, n in enumerate(b["lengths"].tolist()):
            A.append(a[i, :, :n]); M.append(b["patch_target"][i, :n].numpy())
            NS.append(int(b["n_slices"][i]))
            P.append(ps[i, :n] if probe_w is not None else None)
    return A, M, P, NS


def affinity_mine(A, M):
    """aff[k,f] = mean over annotated bags of the share of slot k's attention
    mass falling inside finding f. f=0 lesion, f=1 background."""
    tot, cnt = None, 0
    for a, m in zip(A, M):
        t = (m > 0).astype(np.float64)
        if t.sum() == 0:
            continue
        an = a / np.clip(a.sum(axis=1, keepdims=True), 1e-8, None)
        aff = np.stack([an @ t, an @ (1.0 - t)], axis=1)  # K,2
        tot = aff if tot is None else tot + aff
        cnt += 1
    return tot / cnt, cnt


def geo_scores(ns, n):
    yy, xx = np.meshgrid(np.linspace(-1, 1, GRID), np.linspace(-1, 1, GRID), indexing="ij")
    ip = np.sqrt(yy.ravel() ** 2 + xx.ravel() ** 2)
    z = np.repeat(np.linspace(-1, 1, ns), NP_)[:n]
    d3 = np.sqrt(z ** 2 + np.tile(ip, ns)[:n] ** 2)
    return -d3, -np.tile(ip, ns)[:n]


def main():
    sp = json.loads(Path(SPLITS).read_text())
    print(f"[audit] splits val={len(sp['val'])} test={len(sp['test'])}", flush=True)
    assert not (set(sp["val"]) & set(sp["test"])), "val/test series overlap!"

    def sub(keys):
        p = FeatureBagDataset(CACHE, keys=keys, return_mask=True)
        return FeatureBagDataset(CACHE, keys=p.masked_keys(), return_mask=True)

    val_ds, test_ds = sub(sp["val"]), sub(sp["test"])
    print(f"[audit] masked val={len(val_ds)} test={len(test_ds)} dim={val_ds.dim}", flush=True)

    # ---- probe (fit on TRAIN only)
    with h5py.File(CACHE, "r") as f:
        clf, shp, npos = fit_probe(f, sp["train"])
        print(f"[audit] probe train X={shp} pos={npos}", flush=True)
        pooled, p_pos, p_neg, nsc = probe_pooled_20to1(f, clf, sp["test"])
        print(f"[audit] probe POOLED 20:1 on {nsc} test scans: AUC={pooled:.4f} "
              f"pos={p_pos} neg={p_neg}", flush=True)
    OUT["probe_pooled_20to1_40scans"] = pooled
    OUT["probe_pooled_n"] = [p_pos, p_neg, nsc]
    pw, pb = clf.coef_[0].astype(np.float32), float(clf.intercept_[0])

    match_to = slot_pooling_param_count(val_ds.dim, 256, 8)

    for arm, pooling in [("slot_div=0.5", "slot"), ("mh_abmil", "mh_abmil")]:
        ck = Path("runs/lidc") / arm / "seed0" / "best.pt"
        m = build_model(pooling=pooling, input_dim=val_ds.dim, dim=256,
                        num_classes=sp.get("num_classes", 2), num_slots=8,
                        readout="gated",
                        match_params_to=match_to if pooling == "mh_abmil" else None)
        m.load_state_dict(torch.load(ck, map_location="cpu"))

        VA, VM, _, _ = collect(m, val_ds)
        aff, ncnt = affinity_mine(VA, VM)
        row, col = linear_sum_assignment(-aff)
        assign = {int(r): int(c) for r, c in zip(row, col)}
        lesion = next((k for k, v in assign.items() if v == 0), 0)
        argmax_rule = int(np.argmax(aff[:, 0]))
        print(f"\n[{arm}] val bags used for affinity = {ncnt}", flush=True)
        print(f"[{arm}] aff[:,0] (lesion mass share) = {np.round(aff[:,0],5).tolist()}")
        print(f"[{arm}] hungarian assign = {assign}  lesion_slot={lesion}  "
              f"plain-argmax={argmax_rule}  IDENTICAL={lesion==argmax_rule}")

        TA, TM, TP, TNS = collect(m, test_ds, probe_w=pw if arm.startswith("slot") else None,
                                  probe_b=pb)

        K = TA[0].shape[0]
        per_slot, frozen, oracle, probe_bag, cen3, cen2 = [], [], [], [], [], []
        valauc = []
        for a, mk in zip(VA, VM):
            t = (mk > 0).astype(int)
            if t.sum() == 0 or t.sum() == len(t):
                continue
            valauc.append([bag_auc(t, a[k]) for k in range(K)])
        valauc = np.mean(valauc, axis=0)

        for i, (a, mk) in enumerate(zip(TA, TM)):
            t = (mk > 0).astype(int)
            if t.sum() == 0 or t.sum() == len(t):
                continue
            aucs = [bag_auc(t, a[k]) for k in range(K)]
            per_slot.append(aucs)
            frozen.append(aucs[lesion])
            oracle.append(max(aucs))
            d3, d2 = geo_scores(TNS[i], len(t))
            cen3.append(bag_auc(t, d3)); cen2.append(bag_auc(t, d2))
            if TP[i] is not None:
                probe_bag.append(bag_auc(t, TP[i]))
        per_slot = np.array(per_slot)
        frozen = np.array(frozen)

        print(f"[{arm}] n scored bags = {len(frozen)} of {len(TA)}")
        print(f"[{arm}] FROZEN slot {lesion} test AUC = {frozen.mean():.4f} "
              f"(sem {frozen.std(ddof=1)/np.sqrt(len(frozen)):.4f})")
        print(f"[{arm}] per-slot test AUC  = {np.round(per_slot.mean(0),4).tolist()}")
        print(f"[{arm}] per-slot VAL  AUC  = {np.round(valauc,4).tolist()}")
        print(f"[{arm}] oracle = {np.mean(oracle):.4f}")
        # sign-calibrated selection on VAL
        ksign = int(np.argmax(np.abs(valauc - 0.5)))
        sgn = 1.0 if valauc[ksign] >= 0.5 else -1.0
        sc = per_slot[:, ksign] if sgn > 0 else 1 - per_slot[:, ksign]
        print(f"[{arm}] SIGN-CALIBRATED val pick slot={ksign} sign={sgn:+.0f} "
              f"-> test AUC = {sc.mean():.4f}")
        OUT[arm] = {
            "lesion_slot": lesion, "argmax_slot": argmax_rule,
            "frozen_auc": float(frozen.mean()),
            "frozen_sem": float(frozen.std(ddof=1) / np.sqrt(len(frozen))),
            "per_slot_test": per_slot.mean(0).tolist(),
            "per_slot_val": valauc.tolist(),
            "oracle": float(np.mean(oracle)),
            "signcal_slot": ksign, "signcal_sign": sgn,
            "signcal_auc": float(sc.mean()),
            "n_bags": int(len(frozen)),
        }
        if arm.startswith("slot"):
            OUT["centre_prior_3d"] = float(np.mean(cen3))
            OUT["centre_prior_inplane"] = float(np.mean(cen2))
            OUT["probe_perbag_macro"] = float(np.mean(probe_bag))
            OUT["probe_perbag_n"] = len(probe_bag)
            print(f"\n[geo] centre-prior 3D  (model-free) = {np.mean(cen3):.4f}")
            print(f"[geo] centre-prior in-plane          = {np.mean(cen2):.4f}")
            print(f"[probe] PER-BAG MACRO over {len(probe_bag)} test bags, "
                  f"ALL negatives = {np.mean(probe_bag):.4f}")
            pb_ = np.array(probe_bag)
            d = frozen - pb_
            print(f"[probe] paired slot - probe = {d.mean():+.4f} "
                  f"(p={stats.ttest_rel(frozen, pb_).pvalue:.2g})")
            dc = frozen - np.array(cen3)
            print(f"[geo]   paired slot - centre = {dc.mean():+.4f} "
                  f"(p={stats.ttest_rel(frozen, np.array(cen3)).pvalue:.2g}, "
                  f"slot better in {(dc>0).mean():.2f} of bags)")
            OUT["paired_slot_minus_probe"] = float(d.mean())
            OUT["paired_slot_minus_centre"] = float(dc.mean())

    Path("runs/audit_independent.json").write_text(json.dumps(OUT, indent=2))
    print("\nwrote runs/audit_independent.json")


if __name__ == "__main__":
    main()
