#!/usr/bin/env python
"""Does float16 storage of the attention change any conclusion?

The cached attention is written as float16 to keep the per-model cache at ~150MB
instead of ~500MB. For seed0 that is harmless (frozen AUC 0.8424 vs the float32
reeval's 0.8423) but seed2's attention is far more peaked -- its npz compresses
to 40MB where seed0's is 76MB -- so fp16 quantisation creates ties among the
small values and could move its number. This recomputes the headline statistic
and the static-template comparison from float32 attention for the affected seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from null_battery import load, pick_lesion_slot, score
from null_static_template import evaluate, global_template

D = Path("runs/nulls")
PAIRS = [("real_seed0", "f32_seed0"), ("real_seed2", "f32_seed2")]


def main():
    out = {}
    for fp16_tag, fp32_tag in PAIRS:
        row = {}
        for tag in (fp16_tag, fp32_tag):
            if not (D / f"{tag}_test.npz").exists():
                print(f"missing {tag}, skipping")
                continue
            va, vm = load(D / f"{tag}_val.npz")
            ta, tm = load(D / f"{tag}_test.npz")
            slot, _ = pick_lesion_slot(va, vm)
            s = score(ta, tm, slot, full=True)
            gt = global_template(va, vm, slot)
            r = evaluate(ta, tm, slot, gt)
            row[tag] = {
                "lesion_slot": int(slot),
                "frozen_auc": s["instance_auc_frozen_slot"],
                "oracle": s["instance_auc_oracle_slot"],
                "max_over_slots": s["instance_auc_max_over_slots"],
                "ap_lift": s["ap_lift"],
                "global_static": float(r["global_static"].mean()),
                "bag_static": float(r["bag_static"].mean()),
                "residual": float(r["residual"].mean()),
                "slot_minus_global_static": float((r["slot"] - r["global_static"]).mean()),
            }
            print(f"[{tag}] slot={slot} frozen={s['instance_auc_frozen_slot']:.4f} "
                  f"global_static={row[tag]['global_static']:.4f} "
                  f"slot-global={row[tag]['slot_minus_global_static']:+.4f}", flush=True)
            del va, vm, ta, tm
        out[f"{fp16_tag}_vs_{fp32_tag}"] = row

    Path(D / "fp16_check.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {D / 'fp16_check.json'}")


if __name__ == "__main__":
    main()
