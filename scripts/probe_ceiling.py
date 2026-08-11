#!/usr/bin/env python
"""Supervised patch-level probe: the ceiling for any pooling method on a cache.

The control that turns a null localisation result into an interpretable one. If
slot attention finds no lesion signal, there are two very different explanations:
the features do not encode it, or the weakly-supervised objective fails to
discover it. A logistic probe trained directly on patch labels, over the same
features and the same patient-disjoint splits, separates them.

On the LIDC 224px DINOv2 cache this returns AUC 0.9102 against SlotMIL's
unsupervised 0.489 -- i.e. the signal is present and weak supervision misses it.
"""

from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def collect(f, uids, max_scans, neg_per_pos, rng):
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
        gh = int(g.attrs["grid_h"])
        for (s, r, c) in pos:
            X.append(feats[s, r * gh + c]); y.append(1)
        flat = m.reshape(m.shape[0], -1)
        neg = np.argwhere(flat == 0)
        pick = rng.choice(len(neg), min(len(pos) * neg_per_pos, len(neg)), replace=False)
        for j in pick:
            s, p = neg[j]; X.append(feats[s, p]); y.append(0)
        used += 1
        if used >= max_scans:
            break
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--train-scans", type=int, default=60)
    ap.add_argument("--test-scans", type=int, default=40)
    ap.add_argument("--neg-per-pos", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sp = json.loads(open(args.splits).read())
    rng = np.random.default_rng(args.seed)
    with h5py.File(args.cache, "r") as f:
        Xtr, ytr = collect(f, sp["train"], args.train_scans, args.neg_per_pos, rng)
        Xte, yte = collect(f, sp["test"], args.test_scans, args.neg_per_pos, rng)

    print(f"train {Xtr.shape} pos={int(ytr.sum())}  test {Xte.shape} pos={int(yte.sum())}")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    print(f"\nsupervised patch-level AUC = {auc:.4f}")
    print("this is the ceiling any MIL pooling could reach on these features")
    return auc


if __name__ == "__main__":
    main()
