#!/usr/bin/env python
"""Print every null-control number in one place, read back from the JSON."""

from __future__ import annotations

import json
from pathlib import Path

D = Path("runs/nulls")


def get(p):
    p = D / p
    return json.loads(p.read_text()) if p.exists() else None


def main():
    print("=" * 78)
    print("A. THE PROTOCOL UNDER TEST, RECOMPUTED  (LIDC nodule_present, K=8)")
    print("=" * 78)
    pub = json.loads(Path("runs/corrected_alignment.json").read_text())
    for s in ("seed0", "seed1", "seed2"):
        b = get(f"null_battery_{s}.json")
        v = pub["nodule_present"]["slot_div=0.5"].get(s, {})
        line = f"  {s}: published {v.get('instance_auc_frozen_slot', float('nan')):.4f}"
        if b:
            r = b["reference_real"]
            line += (f"   recomputed {r['instance_auc_frozen_slot']:.4f}"
                     f"  slot={r['lesion_slot']}  oracle={r['instance_auc_oracle_slot']:.4f}")
        print(line)

    print("\n" + "=" * 78)
    print("B. NULLS 1-4, full protocol (fit on val, freeze, score test)")
    print("=" * 78)
    for s in ("seed0", "seed1"):
        b = get(f"null_battery_{s}.json")
        if not b:
            continue
        r = b["reference_real"]
        print(f"\n  --- {s} (reference {r['instance_auc_frozen_slot']:.4f}, "
              f"slot {r['lesion_slot']}, slot-entropy {r['slot_entropy_test']:.4f} "
              f"of max {r['max_possible_entropy']:.4f}) ---")
        print(f"  per-slot test AUC          {[round(x,3) for x in r['per_slot_test_auc']]}")
        n1 = b["null1_random_attention"]
        print(f"  N1 random attention (entropy-matched scale {n1['matched_scale']:.3f}):")
        for sc, a in sorted(n1["by_scale"].items(), key=lambda kv: float(kv[0])):
            tag = "  <- entropy-matched" if a.get("is_entropy_matched") else ""
            print(f"       scale {float(sc):<9.4f} H={a['slot_entropy_test']:.4f}  "
                  f"AUC {a['mean']:.4f} +/- {a['std']:.4f}  "
                  f"[{a['min']:.4f},{a['max']:.4f}]  "
                  f"test-oracle {a['oracle_mean']:.4f}{tag}")
        a = b["null1b_random_attention_slot_bias"]
        print(f"       + persistent per-slot bias   AUC {a['mean']:.4f} +/- {a['std']:.4f}")
        if "null2_untrained" in b:
            u = b["null2_untrained"]
            print(f"  N2 untrained model           AUC {u['instance_auc_frozen_slot']:.4f} "
                  f"(sem {u['auc_sem_across_bags']:.4f}) slot={u['lesion_slot']} "
                  f"test-oracle {u['instance_auc_oracle_slot']:.4f} "
                  f"H={u['slot_entropy_test']:.4f}")
            print(f"       per-slot test AUC       {[round(x,3) for x in u['per_slot_test_auc']]}")
        for nm, lbl in (("shuffle_across_bags", "cross-patient mask swap"),
                        ("circular_roll", "circular roll (strict)")):
            a = b[f"null3_permute_{nm}"]
            print(f"  N3 permute {lbl:<24} AUC {a['mean']:.4f} +/- {a['std']:.4f} "
                  f"[{a['min']:.4f},{a['max']:.4f}]")
        for nm, lbl in (("shuffle_across_bags", "cross-patient mask swap"),
                        ("circular_roll", "circular roll (strict)")):
            a = b[f"null4_selection_{nm}"]
            print(f"  N4 name-slot-on {lbl:<21} AUC {a['instance_auc_mean']:.4f} +/- "
                  f"{a['instance_auc_std']:.4f} [{a['instance_auc_min']:.4f},"
                  f"{a['instance_auc_max']:.4f}] slots={sorted(set(a['selected_slots']))}")
        a = b["null4_uniform_random_slot"]
        print(f"  N4 uniformly random slot     AUC {a['mean_over_all_slots']:.4f} "
              f"[{a['min_slot']:.4f},{a['max_slot']:.4f}]")
        print(f"  diagnostic centre-prior      AUC {b['diagnostic_centre_prior']['instance_auc']:.4f} "
              f"(no model, no training, no fitting)")

    print("\n" + "=" * 78)
    print("C. IS THE FROZEN SLOT A DETECTOR OR A STATIC IN-PLANE TEMPLATE?")
    print("=" * 78)
    st = get("static_template.json")
    if st:
        print(f"  {'tag':<16}{'slot':>5}{'slot AUC':>11}{'bag_static':>12}"
              f"{'global_static':>15}{'residual':>10}{'slot-global':>13}{'p':>10}")
        for t, v in st.items():
            m = v["means"]
            p = v["slot_vs_global_static"]
            print(f"  {t:<16}{v['lesion_slot']:>5}{m['slot']:>11.4f}{m['bag_static']:>12.4f}"
                  f"{m['global_static']:>15.4f}{m['residual']:>10.4f}"
                  f"{p['mean_diff']:>+13.4f}{p['t_p']:>10.2g}")
        print("  global_static = one 256-number map fit on VAL, applied blind to every")
        print("  test patch. It never reads a test image.")

    print("\n" + "=" * 78)
    print("D. CROSSTAB: how much of the AUC is patient-specific?")
    print("=" * 78)
    ct = get("crosstab.json")
    if ct:
        for t, v in ct.items():
            print(f"\n  [{t}] slot={v['lesion_slot']} n_bags={v['n_bags']}")
            print(f"  {'scorer':<15}{'real':>9}{'shuffle':>9}{'roll':>9}"
                  f"{'real-shuf':>11}{'shuf-roll':>11}")
            for k, row in v["table"]["real"].items():
                re_ = row["mean"]
                sh = v["table"]["shuffle"][k]["mean"]
                ro = v["table"]["roll"][k]["mean"]
                print(f"  {k:<15}{re_:>9.4f}{sh:>9.4f}{ro:>9.4f}"
                      f"{re_-sh:>11.4f}{sh-ro:>11.4f}")
        print("\n  real-shuffle = patient-specific component")
        print("  shuffle-roll = population 'nodules live here' prior")
        print("  roll         = metric floor")

    print("\n" + "=" * 78)
    print("E. Z vs IN-PLANE DECOMPOSITION")
    print("=" * 78)
    dc = get("decompose_seed0.json")
    if dc:
        for t, v in dc.items():
            if t == "centre_prior":
                print(f"  centre_prior     flat {v['instance_auc_flat']:.4f}  "
                      f"slice {v['slice_auc']:.4f}  within-slice {v['within_slice_auc']:.4f}")
                continue
            r = v["real"]
            print(f"  {t} slot={r['lesion_slot']}   flat {r['instance_auc_flat']:.4f}  "
                  f"slice {r['slice_auc']:.4f}  within-slice {r['within_slice_auc']:.4f}")
            for nm in ("circular_roll", "shuffle_across_bags"):
                n = v.get(f"null_{nm}")
                if n:
                    print(f"     null[{nm:<20}] flat {n['instance_auc_flat']:.4f}  "
                          f"slice {n['slice_auc']:.4f}  within-slice {n['within_slice_auc']:.4f}")
            pv = v.get("paired_vs_centre_prior")
            if pv:
                print(f"     slot vs centre-prior {pv['slot_auc']:.4f} vs {pv['prior_auc']:.4f} "
                      f"diff {pv['mean_diff']:+.4f} +/- {pv['sem_diff']:.4f} "
                      f"(better in {pv['frac_bags_slot_better']:.2f} of bags, p={pv['t_p']:.2g})")
        print("  slice = AUC of per-slice mean attention vs 'this slice has a nodule'")
        print("  within-slice = AUC over patches of lesion-bearing slices only")

    fp = get("fp16_check.json")
    if fp:
        print("\n" + "=" * 78)
        print("F. fp16 STORAGE CHECK")
        print("=" * 78)
        for t, v in fp.items():
            print(f"  {t}: {v}")


if __name__ == "__main__":
    main()
