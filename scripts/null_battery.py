#!/usr/bin/env python
"""Stage 2: the null-control battery for the frozen-slot localisation protocol.

The protocol under test: compute slot->finding affinity on VAL, take the
Hungarian assignment, freeze it, read off the slot mapped to finding 0 (lesion),
and score ONLY that slot's attention on TEST by mean per-bag ROC AUC against the
patch-level mask. It reports 0.79-0.84 on LIDC. This asks what the same protocol
returns when the signal is removed, four ways.

One structural fact first, because it sets what the nulls have to rule out.
``binary_findings`` builds F=2 channels, [lesion, not-lesion], and affinity rows
are attention renormalised over instances, so aff[k,1] == 1 - aff[k,0] exactly.
Hungarian on a K x 2 matrix returns min(K,F) = 2 pairs and maximises
aff[k1,0] + aff[k2,1] = aff[k1,0] + 1 - aff[k2,0]. The "Hungarian assignment
over K=8 slots" is therefore *identical* to ``lesion_slot = argmax_k aff[k,0]``:
a best-of-8 selection on validation. That is exactly the operation control 4
has to price.

Nulls:
  1. random    -- real bag/mask geometry, attention replaced by noise softmaxed
                  over the slot axis, temperature swept and one setting tuned to
                  match the real model's per-instance slot entropy.
  1b. random with a persistent per-slot logit bias, so slots have a stable
                  identity across bags the way trained slots do (a stable
                  identity is what makes any val->test transfer possible).
  2. untrained -- real architecture, random weights, no training.
  3. permute   -- real checkpoint, real attention, masks randomised so the
                  attention<->mask correspondence dies while every marginal
                  survives.
  4. selection -- real checkpoint, real attention, REAL test masks, assignment
                  fit on RANDOMISED validation masks. Prices best-of-8 selection.

Two diagnostics that are not nulls but bound the interpretation:
  - the full per-slot test AUC distribution (what a uniformly random slot gets),
  - a fixed geometric centre-prior, to check whether a lesion-agnostic
    anatomical prior already reaches the headline AUC.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, roc_auc_score

N_PATCH = 256  # 16x16 DINOv2 ViT-B/14 grid per slice
GRID = 16


def load(path):
    z = np.load(path, allow_pickle=True)
    attn, mask, lengths = z["attn"], z["mask"], z["lengths"]
    off = np.concatenate([[0], np.cumsum(lengths)])
    attns = [np.ascontiguousarray(attn[:, off[i]:off[i + 1]], dtype=np.float32)
             for i in range(len(lengths))]
    masks = [mask[off[i]:off[i + 1]].astype(np.int8) for i in range(len(lengths))]
    return attns, masks


# ---------------------------------------------------------------- protocol
def binary_findings(masks):
    return [np.stack([(m > 0).astype(np.float32), 1.0 - (m > 0).astype(np.float32)])
            for m in masks]


def slot_finding_affinity(slot_attn, finding_masks):
    total, count = None, 0
    for attn, m in zip(slot_attn, finding_masks):
        if m.sum() == 0:
            continue
        a = attn / attn.sum(axis=-1, keepdims=True).clip(1e-8)
        aff = a @ m.T
        total = aff if total is None else total + aff
        count += 1
    return total / count


def fit_slot_assignment(affinity):
    row, col = linear_sum_assignment(-affinity)
    return {int(r): int(c) for r, c in zip(row, col)}


def pick_lesion_slot(attns, masks):
    aff = slot_finding_affinity(attns, binary_findings(masks))
    assign = fit_slot_assignment(aff)
    return next((k for k, v in assign.items() if v == 0), 0), aff


def score(attns, masks, lesion_slot, full=True):
    """Identical statistic to reeval_all_alignment.score: mean per-bag AUC."""
    frozen, maxover, oracle, aps, prevs = [], [], [], [], []
    for a, m in zip(attns, masks):
        t = (m > 0).astype(np.int8)
        s = int(t.sum())
        if s == 0 or s == len(t):
            continue
        frozen.append(roc_auc_score(t, a[lesion_slot]))
        if full:
            maxover.append(roc_auc_score(t, a.max(axis=0)))
            oracle.append(max(roc_auc_score(t, a[k]) for k in range(a.shape[0])))
            aps.append(average_precision_score(t, a[lesion_slot]))
            prevs.append(t.mean())
    if not frozen:
        return None
    out = {
        "instance_auc_frozen_slot": float(np.mean(frozen)),
        "auc_sem_across_bags": float(np.std(frozen) / np.sqrt(len(frozen))),
        "n_bags": len(frozen),
    }
    if full:
        out.update({
            "instance_auc_max_over_slots": float(np.mean(maxover)),
            "instance_auc_oracle_slot": float(np.mean(oracle)),
            "avg_precision": float(np.mean(aps)),
            "prevalence": float(np.mean(prevs)),
            "ap_lift": float(np.mean(aps) / max(np.mean(prevs), 1e-12)),
        })
    return out


def per_slot_auc(attns, masks):
    k = attns[0].shape[0]
    acc = [[] for _ in range(k)]
    for a, m in zip(attns, masks):
        t = (m > 0).astype(np.int8)
        s = int(t.sum())
        if s == 0 or s == len(t):
            continue
        for j in range(k):
            acc[j].append(roc_auc_score(t, a[j]))
    return [float(np.mean(v)) for v in acc]


def protocol(val_attns, val_masks, test_attns, test_masks, full=True):
    slot, aff = pick_lesion_slot(val_attns, val_masks)
    out = score(test_attns, test_masks, slot, full=full)
    out["lesion_slot"] = int(slot)
    out["val_affinity_lesion_col"] = [float(x) for x in aff[:, 0]]
    return out


# ---------------------------------------------------------- null 1: random
def slot_entropy(attns):
    tot, n = 0.0, 0
    for a in attns:
        p = np.clip(a, 1e-12, None)
        p = p / p.sum(axis=0, keepdims=True)
        tot += float((-(p * np.log(p)).sum(axis=0)).sum())
        n += a.shape[1]
    return tot / n


def random_attn(lengths, k, scale, rng, slot_bias=None):
    out = []
    for n in lengths:
        logits = rng.standard_normal((k, n)).astype(np.float32) * scale
        if slot_bias is not None:
            logits = logits + slot_bias[:, None]
        logits -= logits.max(axis=0, keepdims=True)
        e = np.exp(logits)
        out.append(e / e.sum(axis=0, keepdims=True))
    return out


def match_scale(target_H, k, lo=1e-3, hi=30.0, iters=40):
    """Bisect the logit scale so random attention matches the real slot entropy.
    Entropy per instance is independent of bag length, so a small probe suffices."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        h = slot_entropy(random_attn([20000], k, mid, np.random.default_rng(0)))
        if h > target_H:  # too uniform -> sharpen
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------- null 3/4: mask scrambles
def shuffle_masks_across_bags(masks, rng):
    """Give each bag another bag's mask (deranged), tiled/cropped to length."""
    b = len(masks)
    perm = rng.permutation(b)
    for i in range(b):
        if perm[i] == i:
            j = (i + 1) % b
            perm[i], perm[j] = perm[j], perm[i]
    out = []
    for i in range(b):
        src, n = masks[perm[i]], len(masks[i])
        out.append(src[:n].copy() if len(src) >= n else np.resize(src, n))
    return out


def roll_masks(masks, rng):
    """Circular shift by a random offset. Preserves per-bag prevalence EXACTLY
    and the mask's own spatial structure; kills only mask<->attention
    registration. The strictest form of the label-permutation null."""
    return [np.roll(m, int(rng.integers(1, len(m)))) for m in masks]


# ------------------------------------------------------------- diagnostics
def centre_prior_auc(masks):
    """Lesion-agnostic geometry: score every patch by -distance from the volume
    centre. If this alone scores ~0.8 the headline number is anatomy, not lesion."""
    yy, xx = np.meshgrid(np.linspace(-1, 1, GRID), np.linspace(-1, 1, GRID), indexing="ij")
    inplane = np.sqrt(yy.ravel() ** 2 + xx.ravel() ** 2)
    aucs = []
    for m in masks:
        t = (m > 0).astype(np.int8)
        s = int(t.sum())
        if s == 0 or s == len(t):
            continue
        ns = int(np.ceil(len(m) / N_PATCH))
        z = np.repeat(np.linspace(-1, 1, ns), N_PATCH)[: len(m)]
        d = np.sqrt(z ** 2 + np.tile(inplane, ns)[: len(m)] ** 2)
        aucs.append(roc_auc_score(t, -d))
    return float(np.mean(aucs)), len(aucs)


def agg(runs, key="instance_auc_frozen_slot"):
    v = [x[key] for x in runs]
    return {"mean": float(np.mean(v)), "std": float(np.std(v)),
            "min": float(np.min(v)), "max": float(np.max(v)), "n_reps": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/nulls")
    ap.add_argument("--real-tag", default="real_seed0")
    ap.add_argument("--untrained-tag", default="untrained_s1234")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--sweep-reps", type=int, default=6)
    ap.add_argument("--out", default="runs/nulls/null_battery.json")
    args = ap.parse_args()

    d, R = Path(args.dir), {}
    t0 = time.time()

    print("loading real attention ...", flush=True)
    va, vm = load(d / f"{args.real_tag}_val.npz")
    ta, tm = load(d / f"{args.real_tag}_test.npz")
    k = va[0].shape[0]
    vl = [a.shape[1] for a in va]
    tl = [a.shape[1] for a in ta]
    print(f"  K={k} val bags={len(va)} test bags={len(ta)} "
          f"total_N val={sum(vl)} test={sum(tl)}  ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- reference: the published protocol, recomputed ----------
    ref = protocol(va, vm, ta, tm)
    ref["per_slot_test_auc"] = per_slot_auc(ta, tm)
    ref["slot_entropy_val"] = slot_entropy(va)
    ref["slot_entropy_test"] = slot_entropy(ta)
    ref["max_possible_entropy"] = float(np.log(k))
    R["reference_real"] = ref
    print(f"REFERENCE   frozen={ref['instance_auc_frozen_slot']:.4f} "
          f"(sem {ref['auc_sem_across_bags']:.4f}) slot={ref['lesion_slot']} "
          f"oracle={ref['instance_auc_oracle_slot']:.4f} maxover={ref['instance_auc_max_over_slots']:.4f} "
          f"aplift={ref['ap_lift']:.2f} H={ref['slot_entropy_test']:.4f}/{np.log(k):.4f} "
          f"n={ref['n_bags']}", flush=True)
    print(f"  per-slot test AUC : {[round(x,3) for x in ref['per_slot_test_auc']]}", flush=True)
    print(f"  val aff[:,lesion] : {[round(x,5) for x in ref['val_affinity_lesion_col']]}", flush=True)

    # ---------------- diagnostic: geometric centre prior ---------------------
    cp, ncp = centre_prior_auc(tm)
    R["diagnostic_centre_prior"] = {"instance_auc": cp, "n_bags": ncp}
    print(f"CENTRE-PRIOR (lesion-agnostic geometry) auc={cp:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- null 1: random attention ------------------------------
    matched = match_scale(ref["slot_entropy_test"], k)
    scales = sorted({round(float(s), 4) for s in [0.25, 1.0, 4.0, matched]})
    R["null1_random_attention"] = {"matched_scale": float(matched),
                                   "target_entropy": ref["slot_entropy_test"],
                                   "by_scale": {}}
    for s in scales:
        runs = []
        for r in range(args.sweep_reps):
            rr = np.random.default_rng(1000 + r)
            rva = random_attn(vl, k, s, rr)
            rta = random_attn(tl, k, s, rr)
            o = protocol(rva, vm, rta, tm)
            o["slot_entropy_test"] = slot_entropy(rta)
            runs.append(o)
            del rva, rta
        a = agg(runs)
        a["oracle_mean"] = float(np.mean([x["instance_auc_oracle_slot"] for x in runs]))
        a["maxover_mean"] = float(np.mean([x["instance_auc_max_over_slots"] for x in runs]))
        a["ap_lift_mean"] = float(np.mean([x["ap_lift"] for x in runs]))
        a["slot_entropy_test"] = float(np.mean([x["slot_entropy_test"] for x in runs]))
        a["is_entropy_matched"] = bool(abs(s - matched) < 1e-6)
        R["null1_random_attention"]["by_scale"][f"{s}"] = a
        print(f"NULL1 random scale={s:<8} H={a['slot_entropy_test']:.4f} "
              f"frozen={a['mean']:.4f}+/-{a['std']:.4f} [{a['min']:.4f},{a['max']:.4f}] "
              f"oracle={a['oracle_mean']:.4f} maxover={a['maxover_mean']:.4f} "
              f"aplift={a['ap_lift_mean']:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    runs = []
    for r in range(args.sweep_reps):
        rr = np.random.default_rng(2000 + r)
        bias = rr.standard_normal(k).astype(np.float32)
        rva = random_attn(vl, k, matched, rr, slot_bias=bias)
        rta = random_attn(tl, k, matched, rr, slot_bias=bias)
        runs.append(protocol(rva, vm, rta, tm, full=False))
        del rva, rta
    R["null1b_random_attention_slot_bias"] = agg(runs)
    a = R["null1b_random_attention_slot_bias"]
    print(f"NULL1b random+persistent-slot-bias frozen={a['mean']:.4f}+/-{a['std']:.4f} "
          f"[{a['min']:.4f},{a['max']:.4f}]  ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- null 2: untrained model -------------------------------
    up = d / f"{args.untrained_tag}_val.npz"
    if up.exists():
        uva, uvm = load(up)
        uta, utm = load(d / f"{args.untrained_tag}_test.npz")
        o = protocol(uva, uvm, uta, utm)
        o["per_slot_test_auc"] = per_slot_auc(uta, utm)
        o["slot_entropy_test"] = slot_entropy(uta)
        R["null2_untrained"] = o
        print(f"NULL2 untrained frozen={o['instance_auc_frozen_slot']:.4f} "
              f"(sem {o['auc_sem_across_bags']:.4f}) slot={o['lesion_slot']} "
              f"oracle={o['instance_auc_oracle_slot']:.4f} H={o['slot_entropy_test']:.4f} "
              f"aplift={o['ap_lift']:.2f}", flush=True)
        print(f"  per-slot test AUC : {[round(x,3) for x in o['per_slot_test_auc']]}", flush=True)
        del uva, uvm, uta, utm
    else:
        print("NULL2 skipped: no untrained collection found", flush=True)

    # ---------------- null 3: label permutation -----------------------------
    for name, fn in (("shuffle_across_bags", shuffle_masks_across_bags),
                     ("circular_roll", roll_masks)):
        runs = []
        for r in range(args.reps):
            rr = np.random.default_rng(3000 + r)
            runs.append(protocol(va, fn(vm, rr), ta, fn(tm, rr)))
        a = agg(runs)
        a["oracle_mean"] = float(np.mean([x["instance_auc_oracle_slot"] for x in runs]))
        a["ap_lift_mean"] = float(np.mean([x["ap_lift"] for x in runs]))
        a["lesion_slots"] = [int(x["lesion_slot"]) for x in runs]
        a["n_bags"] = int(np.mean([x["n_bags"] for x in runs]))
        R[f"null3_permute_{name}"] = a
        print(f"NULL3 permute[{name}] frozen={a['mean']:.4f}+/-{a['std']:.4f} "
              f"[{a['min']:.4f},{a['max']:.4f}] oracle={a['oracle_mean']:.4f} "
              f"aplift={a['ap_lift_mean']:.2f} n={a['n_bags']}  ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- null 4: selection bias --------------------------------
    per_slot = ref["per_slot_test_auc"]
    for name, fn in (("shuffle_across_bags", shuffle_masks_across_bags),
                     ("circular_roll", roll_masks)):
        vals, slots = [], []
        for r in range(args.reps):
            rr = np.random.default_rng(4000 + r)
            slot, _ = pick_lesion_slot(va, fn(vm, rr))
            slots.append(int(slot))
            vals.append(score(ta, tm, slot, full=False)["instance_auc_frozen_slot"])
        R[f"null4_selection_{name}"] = {
            "instance_auc_mean": float(np.mean(vals)),
            "instance_auc_std": float(np.std(vals)),
            "instance_auc_min": float(np.min(vals)),
            "instance_auc_max": float(np.max(vals)),
            "selected_slots": slots, "n_reps": args.reps,
        }
        a = R[f"null4_selection_{name}"]
        print(f"NULL4 selection[{name}] test_auc={a['instance_auc_mean']:.4f}"
              f"+/-{a['instance_auc_std']:.4f} [{a['instance_auc_min']:.4f},"
              f"{a['instance_auc_max']:.4f}] slots={slots}  ({time.time()-t0:.0f}s)", flush=True)

    R["null4_uniform_random_slot"] = {
        "mean_over_all_slots": float(np.mean(per_slot)),
        "min_slot": float(np.min(per_slot)), "max_slot": float(np.max(per_slot)),
        "per_slot": per_slot,
        "note": "expected test AUC of a slot drawn uniformly from the K=8, i.e. selection with zero information",
    }
    print(f"NULL4 uniform-random-slot mean={np.mean(per_slot):.4f} "
          f"range=[{np.min(per_slot):.4f},{np.max(per_slot):.4f}]", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(R, indent=2))
    print(f"\nwrote {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
