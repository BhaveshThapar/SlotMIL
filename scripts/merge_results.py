#!/usr/bin/env python
"""Merge per-arm/per-seed results into one summary with significance tests.

Training runs as a SLURM job array (one arm per task) so the six arms train in
parallel rather than serially -- the first LIDC attempt trained them in one
process and hit the 12 h wall clock partway through the third arm. Each task
writes runs/<arm>/seed<N>/result.json; this collects them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from slotmil.eval.classification import aggregate_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="directory holding <arm>/seed<N>/result.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.runs)
    results: dict[str, list[dict]] = {}

    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        rows = []
        for seed_dir in sorted(arm_dir.glob("seed*")):
            f = seed_dir / "result.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            if "test" not in r:
                continue
            rows.append({"seed": int(seed_dir.name.replace("seed", "")), **r["test"]})
        if rows:
            results[arm_dir.name] = rows

    if not results:
        raise SystemExit(f"no completed runs under {root}")

    summary = {k: aggregate_seeds(v) for k, v in results.items()}

    print("=" * 80)
    print(f"{'arm':<18}{'AUC':>18}{'ACC':>18}{'seeds':>7}{'active':>8}{'maxcos':>8}")
    for k, s in summary.items():
        act = f"{s['active_slots']['mean']:>8.2f}" if "active_slots" in s else " " * 8
        cos = f"{s['max_off_diag_cos']['mean']:>8.3f}" if "max_off_diag_cos" in s else " " * 8
        print(f"{k:<18}{s['auc']['mean']:>10.4f} +/-{s['auc']['std']:.4f}"
              f"{s['acc']['mean']:>10.4f} +/-{s['acc']['std']:.4f}"
              f"{s['auc']['n']:>7}{act}{cos}")

    # Same significance gate used everywhere else: a bare mean comparison over a
    # handful of seeds is not evidence of a difference.
    slot_arms = {k: v for k, v in summary.items() if k.split("_")[0] == "slot"}
    comparisons = {}
    if slot_arms:
        best = max(slot_arms, key=lambda k: slot_arms[k]["auc"]["mean"])
        a = slot_arms[best]["auc"]["values"]
        print(f"\nbest slot arm: {best}")
        for other in summary:
            if other == best:
                continue
            b = summary[other]["auc"]["values"]
            if len(a) > 1 and len(b) > 1:
                p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
                d = float(np.mean(a) - np.mean(b))
                # Report direction explicitly. A one-sided "significant only if
                # the slot arm wins" label prints "not significant" when the slot
                # arm significantly LOSES -- which is the opposite of the truth
                # and exactly the reading that flatters your own method.
                wins = bool(d > 0 and p < 0.05)
                loses = bool(d < 0 and p < 0.05)
                comparisons[other] = {"delta": d, "p": p,
                                      "slot_wins": wins, "slot_loses": loses}
                verdict = ("SIGNIFICANT WIN" if wins else
                           "SIGNIFICANT LOSS" if loses else "ns")
                print(f"  vs {other:<16} delta={d:+.4f}  p={p:.3f}  {verdict}")

        # Which slot arms stayed non-degenerate -- alignment must not be scored
        # on a collapsed model.
        healthy = [k for k, v in slot_arms.items()
                   if v.get("max_off_diag_cos", {}).get("mean", 1.0) < 0.9]
        print(f"\nslot arms with max cosine < 0.9: {healthy or 'NONE (all collapsed)'}")

    out = Path(args.out) if args.out else root / "summary.json"
    out.write_text(json.dumps({"summary": summary, "comparisons": comparisons}, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
