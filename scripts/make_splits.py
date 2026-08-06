#!/usr/bin/env python
"""Build train/val/test splits over a feature cache.

Two things this gets right that a naive random split would not:

**Grouping by patient.** LIDC-IDRI holds 1,018 CT series from 1,010 patients, so
a few patients contribute more than one series. Splitting on series would put the
same patient's anatomy in both train and test, inflating every number. Splits are
therefore made over patients and then mapped back to series.

**Stratifying by label.** LIDC's nodule-present rate and MosMed's severity
distribution are both skewed (MosMed CT-4 has 2 volumes in 1,110), so an
unstratified split can leave a class absent from val or test entirely, which
silently breaks AUC.

Splits are written to JSON and version-controlled by hash, so every run and
ablation reads exactly the same partition (plan.md line 127).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


def read_cache_meta(h5_path: str) -> dict[str, dict]:
    out = {}
    with h5py.File(h5_path, "r") as f:
        for uid in f.keys():
            g = f[uid]
            if "features" not in g:
                continue
            out[uid] = {
                "label": int(g.attrs["label"]),
                # Fall back to the series UID so a cache without patient_id
                # degrades to a series-level split rather than silently grouping
                # everything together.
                "patient": str(g.attrs.get("patient_id", uid)),
                "has_mask": "mask" in g,
                "n_slices": int(g.attrs.get("n_slices", 0)),
            }
    return out


def grouped_stratified_split(
    meta: dict[str, dict], fracs=(0.7, 0.15, 0.15), seed: int = 0
) -> dict[str, list[str]]:
    """Assign whole patients to splits, keeping label balance close to target."""
    rng = np.random.default_rng(seed)

    by_patient: dict[str, list[str]] = defaultdict(list)
    for uid, m in meta.items():
        by_patient[m["patient"]].append(uid)

    # A patient's stratum is their majority series label.
    patients = sorted(by_patient)
    strata: dict[int, list[str]] = defaultdict(list)
    for p in patients:
        labels = [meta[u]["label"] for u in by_patient[p]]
        strata[max(set(labels), key=labels.count)].append(p)

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for label, plist in sorted(strata.items()):
        plist = list(plist)
        rng.shuffle(plist)
        n = len(plist)
        n_tr = int(round(fracs[0] * n))
        n_va = int(round(fracs[1] * n))
        chunks = {
            "train": plist[:n_tr],
            "val": plist[n_tr : n_tr + n_va],
            "test": plist[n_tr + n_va :],
        }
        for k, ps in chunks.items():
            for p in ps:
                splits[k].extend(by_patient[p])

    return {k: sorted(v) for k, v in splits.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fracs", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--masked-to-eval", action="store_true",
                    help="force every mask-carrying series into val/test, so the "
                         "localisation evaluation is never scored on training data")
    args = ap.parse_args()

    meta = read_cache_meta(args.cache)
    if not meta:
        raise SystemExit(f"no usable series in {args.cache}")
    print(f"[splits] {len(meta)} series, {len({m['patient'] for m in meta.values()})} patients")

    splits = grouped_stratified_split(meta, tuple(args.fracs), args.seed)

    if args.masked_to_eval:
        # Move masked series (and their patients) out of train.
        masked_patients = {meta[u]["patient"] for u in meta if meta[u]["has_mask"]}
        moved = [u for u in splits["train"] if meta[u]["patient"] in masked_patients]
        if moved:
            splits["train"] = [u for u in splits["train"] if u not in set(moved)]
            half = len(moved) // 2
            splits["val"] += moved[:half]
            splits["test"] += moved[half:]
            splits = {k: sorted(v) for k, v in splits.items()}
            print(f"[splits] moved {len(moved)} masked series out of train")

    # Leakage check -- the whole point of grouping.
    pat = {k: {meta[u]["patient"] for u in v} for k, v in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = pat[a] & pat[b]
        if overlap:
            raise SystemExit(f"patient leakage between {a} and {b}: {sorted(overlap)[:5]}")
    print("[splits] no patient overlap between splits")

    for k, v in splits.items():
        labels = [meta[u]["label"] for u in v]
        masked = sum(meta[u]["has_mask"] for u in v)
        dist = {int(c): int((np.array(labels) == c).sum()) for c in sorted(set(labels))}
        print(f"  {k:<6} {len(v):>5} series  {len(pat[k]):>5} patients  labels {dist}  masked {masked}")

    payload = {
        "cache": str(Path(args.cache).resolve()),
        "seed": args.seed,
        "fracs": args.fracs,
        "n_series": len(meta),
        **splits,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    payload["hash"] = hashlib.sha256(blob).hexdigest()[:16]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"[splits] wrote {args.out}  (hash {payload['hash']})")


if __name__ == "__main__":
    main()
